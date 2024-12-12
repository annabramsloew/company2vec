from pathlib import Path

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd

# path depending on user
if Path.home().name == "nikolaibeckjensen":
    DATA_ROOT = Path.home() / "Library" / "CloudStorage" / "OneDrive-DanmarksTekniskeUniversitet" / "Virk2Vec" / "data"
elif Path.home().name == "annabramslow":
    DATA_ROOT = Path.home() / "Library" / "CloudStorage" / "OneDrive-DanmarksTekniskeUniversitet(2)" / "Virk2Vec" / "data"

def fetch_population_data(task_name):

    # fetch data from populations folder
    df_population = pd.read_pickle(DATA_ROOT / "processed" / "populations" / task_name / "population" /"result.pkl")
    df_population = df_population[["TARGET"]]
    #create cvr column based on index
    df_population["CVR"] = df_population.index

    # fetch test cvr numbers from data split folder
    cvr_test = pd.read_pickle(DATA_ROOT / "processed" / "populations" / task_name / "data_split" / "result.pkl")
    cvr_test = cvr_test.test

    # return data only for the test cvr numbers
    df_test = df_population[df_population.CVR.isin(cvr_test)]

    return df_test
# ----------------------------------------------------- Capital -----------------------------------------------------
def capital_increase(min_date, max_date):

    #convert to datetime
    min_date = pd.to_datetime(min_date)
    max_date = pd.to_datetime(max_date)

    # fetch y true data and corresponding cvr numbers
    df_test = fetch_population_data("capital_increase")
    df_test = df_test.reset_index(drop=True)

    # fetch capital increase data to look up in 
    target_path = Path(DATA_ROOT / "Tables" / "CapitalChanges")
    target_csv = [file for file in target_path.iterdir() if file.is_file() and file.suffix == '.csv']
    df_target = dd.read_csv(
        target_csv,
        usecols=["CVR", "Date", "InvestmentDKK", "Rate", "PaymentType", "InvestmentType"],
        on_bad_lines="error",
        assume_missing=True,
        dtype={
            "CVR": int,
            "Date": str,
            "InvestmentDKK": float,
            "Rate": float,
            "PaymentType": str,
            "InvestmentType": str
        },
        blocksize="256MB",
        lineterminator='\n'
    ).compute()

    # filter on InvestmentType
    df_target = df_target.loc[df_target.InvestmentType == 'Kapitalforhøjelse']

    # filter on the period of interest
    df_target['Date'] = pd.to_datetime(df_target['Date'], errors='coerce')
    df_target = df_target.dropna(subset=['Date'])
    df_target = df_target.loc[(df_target.Date >= min_date) & (df_target.Date <= max_date)]

    #only include cvr numbers that are in the test set
    df_target = df_target[df_target.CVR.isin(df_test.CVR)]
    df_target = df_target.drop_duplicates(subset=['CVR'])
    df_target["PREDICTION"] = 1

    # result = pd.DataFrame(index=population_ids)
    # result = result.join(df_target.set_index('CVR')[target_columns], how='left')
    result = df_test.merge(df_target[["CVR", "PREDICTION"]], on="CVR", how="left")
    result = result.fillna(0)
    result = result.set_index("CVR")

    cat_type = pd.api.types.CategoricalDtype(categories=[0,1], ordered=False)
    result['PREDICTION'] = result['PREDICTION'].astype(cat_type)

    assert result.shape[0] == df_test.shape[0]
    assert isinstance(result, pd.DataFrame)
    return result
# ----------------------------------------------------- Moving -----------------------------------------------------
def moving(min_date, max_date):

    #convert to datetime
    min_date = pd.to_datetime(min_date)
    max_date = pd.to_datetime(max_date)

    # fetch y true data and corresponding cvr numbers
    df_test = fetch_population_data("moving")
    df_test = df_test.reset_index(drop=True)

    # fetch moving data to look up in
    target_path = Path(DATA_ROOT / "Tables" / "Registrations")
    target_csv = [file for file in target_path.iterdir() if file.is_file() and file.suffix == '.csv']
    df_target = dd.read_csv(
        target_csv,
        usecols=["CVR", "FromDate", "ChangeType", "NewValue"],
        on_bad_lines="error",
        assume_missing=True,
        dtype={
            "CVR": int,
            "FromDate": str,
            "ChangeType": str,
            "NewValue": str
        },
        blocksize="256MB"
    ).compute()

    # filter on the ChangeType (only include moving)
    df_target = df_target.loc[lambda x: x.ChangeType == 'Municipality']

    # filter on the period of interest
    df_target['FromDate'] = pd.to_datetime(df_target['FromDate'], errors='coerce')
    df_target = df_target.dropna(subset=['FromDate'])
    df_target = df_target.loc[(df_target.FromDate >= min_date) & (df_target.FromDate <= max_date)]

    #only include cvr numbers that are in the test set
    df_target = df_target[df_target.CVR.isin(df_test.CVR)]
    df_target = df_target.drop_duplicates(subset=['CVR'])
    df_target["PREDICTION"] = 1

    result = df_test.merge(df_target[["CVR", "PREDICTION"]], on="CVR", how="left")
    result = result.fillna(0)
    result = result.set_index("CVR")

    cat_type = pd.api.types.CategoricalDtype(categories=[0,1], ordered=False)
    result['PREDICTION'] = result['PREDICTION'].astype(cat_type)
    
    assert result.shape[0] == df_test.shape[0]
    assert isinstance(result, pd.DataFrame)
    return result
# ----------------------------------------------------- Employee -----------------------------------------------------
def employee_level(min_date, max_date):

    #convert to datetime
    min_date = pd.to_datetime(min_date)
    max_date = pd.to_datetime(max_date)

    # fetch y true data and corresponding cvr numbers
    df_test = fetch_population_data("employee_level")
    df_test = df_test.reset_index(drop=True)

    employee_status = ['DECREASE', 'STABLE', 'INCREASE']

    # fetch employee level data to look up in
    target_path = Path(DATA_ROOT / "Tables" / "EmployeeCounts")
    target_csv = [file for file in target_path.iterdir() if file.is_file() and file.suffix == '.csv']
    df_target = dd.read_csv(
        target_csv,
        usecols=["CVR", "FromDate", "ChangeType", "EmployeeCounts"],
        on_bad_lines="error",
        assume_missing=True,
        dtype={
            "CVR": int,
            "FromDate": str,
            "ChangeType": str,
            "EmployeeCounts": float
        },
        blocksize="256MB"
    ).compute()

    # filter on the period of interest in addition to finding previous year data
    df_target['FromDate'] = pd.to_datetime(df_target['FromDate'], errors='coerce')
    df_target['Year'] = df_target['FromDate'].dt.year
    df_target = df_target.dropna(subset=['FromDate'])
    df_previous = df_target.loc[(df_target.FromDate < min_date) & (df_target.FromDate >= min_date - pd.DateOffset(years=1))]
    df_target = df_target.loc[(df_target.FromDate >= min_date) & (df_target.FromDate <= max_date)]

    # filter on relevant test CVRs
    df_target = df_target[df_target.CVR.isin(df_test.CVR)]
    df_previous = df_previous[df_previous.CVR.isin(df_test.CVR)]

    #group by CVR and year and find last employee count
    df_previous = df_previous.sort_values(by=['CVR', 'FromDate'])
    df_previous = df_previous.drop_duplicates(subset=['CVR'], keep='last')
    df_target = df_target.sort_values(by=['CVR', 'FromDate'])
    df_target = df_target.drop_duplicates(subset=['CVR'], keep = 'last')

    # compute target column for df_target with three categories [DECREASE, STABLE, INCREASE]
    # if no previous year data, set to STABLE, if difference is more than 10% set to DECREASE/INCREASE else STABLE
    df_previous = df_previous.rename(columns={'EmployeeCounts': 'PreviousEmployeeCounts'})
    df_target = df_target.merge(df_previous[['CVR', 'PreviousEmployeeCounts']], on=['CVR'], how='left')
    df_target['EmployeeDiff'] = (df_target['EmployeeCounts'] - df_target['PreviousEmployeeCounts']) / df_target['PreviousEmployeeCounts']
    df_target['PREDICTION'] = pd.cut(df_target['EmployeeDiff'], bins=[-np.inf, -0.1, 0.1, np.inf], labels=employee_status)
    
    map_target = {
        'STABLE': 0,
        'INCREASE': 1,
        'DECREASE': 2
    }
    df_target['PREDICTION'] = df_target['PREDICTION'].map(map_target)

    result = df_test.merge(df_target[["CVR", "PREDICTION"]], on="CVR", how="left")
    result = result.fillna(0)
    result = result.set_index("CVR")

    cat_type = pd.api.types.CategoricalDtype(categories=[0,1,2], ordered=False)
    result['PREDICTION'] = result['PREDICTION'].astype(cat_type)

    assert result.shape[0] == df_test.shape[0]
    assert isinstance(result, pd.DataFrame)
    return result
# ----------------------------------------------------- Bankruptcy -----------------------------------------------------
def bankruptcy():

    #threshold for bankruptcy (computed from training data in bankruptcy_baseline_thresholds.ipynb)
    threshold = -0.985

    # fetch y true data and corresponding cvr numbers
    df_test = fetch_population_data("bankruptcy")
    df_test = df_test.reset_index(drop=True)

    #fetch bankruptcy data to look up in
    target_path = Path(DATA_ROOT/'processed'/'sources'/'financials'/'tokenized')
    annualreport_csv = [file for file in target_path.iterdir() if file.is_file() and file.suffix == '.parquet']
    df_annualreport = dd.read_parquet(annualreport_csv).compute()
    #turn cvr index to column
    df_annualreport['CVR'] = df_annualreport.index
    df_annualreport.reset_index(drop=True, inplace=True)

    # find latest financial data for each company
    df_financials = df_annualreport.sort_values(by=['CVR', 'FROM_DATE'])
    df_financials = df_financials.drop_duplicates(subset=['CVR'], keep = 'last')

    # add financial data to test set
    df = df_test.merge(df_financials, on='CVR', how='left')

    #compute working capital / total assets ratio
    df["WorkingCapital"] = df["CURRENT_ASSETS"] - df["SHORT_TERM_LIABILITIES"]
    df["WorkingAssetRatio"] = df["WorkingCapital"] / df["ASSETS"]

    #manual fix for division by zero
    df.loc[df.ASSETS == 0, "WorkingAssetRatio"] = 0
    df.loc[df.ASSETS == 0, "ProfitAssetRatio"] = 0
    #manual fix for nan values
    df = df.fillna(0)

    #remove unnecessary columns
    df = df[['CVR', 'TARGET','WorkingAssetRatio']]

    #bankruptcy prediction based on working capital / total assets ratio and threshold
    df['PREDICTION'] = 0
    df.loc[df.WorkingAssetRatio <= threshold, 'PREDICTION'] = 1
    df = df.drop(columns=['WorkingAssetRatio'])

    result = df.set_index("CVR")
    cat_type = pd.api.types.CategoricalDtype(categories=[0,1], ordered=False)
    result['PREDICTION'] = result['PREDICTION'].astype(cat_type)

    assert result.shape[0] == df_test.shape[0]
    assert isinstance(result, pd.DataFrame)
    return result
# ----------------------------------------------------- Saving -----------------------------------------------------

def save_dummy_predictions(task, dummy_predictions):

    folder = DATA_ROOT.parent / "predictions" / task / "dummy"

    # save dummy predictions as npy file, first target, then prediction
    np.save(folder / "trg.npy", dummy_predictions.TARGET.to_numpy())
    np.save(folder / "prb.npy", dummy_predictions.PREDICTION.to_numpy())
    np.save(folder / "id.npy", dummy_predictions.index.to_numpy())

    return f"Dummy predictions saved for {task}."

# ----------------------------------------------------- Main -----------------------------------------------------
#use for debugging
def main():

    #for capital_increase
    save_dummy_predictions('capital', capital_increase("2021-01-01", "2021-12-31"))
    #for moving
    save_dummy_predictions('moving', moving("2021-01-01", "2021-12-31"))
    #for employee_level
    save_dummy_predictions('employee', employee_level("2021-01-01", "2021-12-31"))
    #for bankruptcy
    save_dummy_predictions('bankruptcy', bankruptcy())

if __name__ == "__main__":
    main()
    print("Done!")