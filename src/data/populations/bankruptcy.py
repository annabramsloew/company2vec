from dataclasses import dataclass, field
from multiprocessing import resource_tracker
from pathlib import Path
from functools import reduce

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd

from ..decorators import save_pickle
from ..serialize import DATA_ROOT
from .base import DataSplit, Population
from .from_annualreports import FromAnnualReports, AnnualReportTokens


# HYDRA_FULL_ERROR=1 python -m src.prepare_data +data_new/population=survival_set target=\${data_new.population}
dateparser = lambda x: pd.to_datetime(x, format = '%d%b%Y:%X',  errors='coerce')

@dataclass
class BankruptcySubPopulation(Population):
    base_population: Population
    name: str = "bankruptcy"
    earliest_birthday: str = "01-01-1951" #TODO: Decide whether we want to filter on company age?
    latest_birthday: str = "31-12-1981"
    status_filters: list = field(default_factory=lambda: ['NORMAL', 'AKTIV'])
    employee_filters: list = field(default_factory=list) # [min, max]
    industry_filters: list = field(default_factory=list) # list of allowed industry codes of 4 digits
    company_type_filters: list = field(default_factory=lambda: ['A/S', 'APS', 'IVS']) # list of allowed company types
    financials_filters: dict = field(default_factory=dict) # {'financialskey1': [min, max], 'financialskey2': [min, max], ...} flexible number of keys

    target_path: Path =  DATA_ROOT / "Tables" / "Registrations"
    period_start: str = "01-01-2022"
    period_end: str = "31-12-2023"
    
    def __post_init__(self) -> None:
        self._earliest_birthday = pd.to_datetime(self.earliest_birthday, format="%d-%m-%Y")
        self._latest_birthday = pd.to_datetime(self.latest_birthday, format="%d-%m-%Y")
        self._period_start = pd.to_datetime(self.period_start, format="%d-%m-%Y")
        self._period_end = pd.to_datetime(self.period_end, format="%d-%m-%Y")
        self._industry_filters = [str(x) for x in self.industry_filters]

    @save_pickle(
        DATA_ROOT / "processed/populations/{self.name}/population",
        on_validation_error="error",
    )
    def population(self) -> pd.DataFrame:
        """Loads the joined sub_population and target dataframe 
           Inherits columns from TARGET:
            * EVENT_FINAL_DATE: date of the event (if event does not happen, it is set to the :attr:`period_end`)
            * TARGET: outcome feature (1 - event happend, 0 otherwise)
        """
        population = self.sub_population()
        target = self.target()
        assert population.shape[0] == target.shape[0]

        result = population.join(target)
        assert isinstance(result, pd.DataFrame)
        return result

    @save_pickle(
        DATA_ROOT / "interim/populations/{self.name}/population",
        on_validation_error="error",
    )
    def sub_population(self) -> pd.DataFrame:
        """
        Return the FILTERED population as a pandas dataframe with an index named CVR.
        Filters according to the parameters of the class.

        We probably want to filter on other things as well. To do so, we have to enrich the population with the relevant data - likely using asof-logic.
        e.g. filter away companies with less than 10 employees, or companies with a certain industry code at the time of filtration.
        """
        base_population = self.base_population.population()
        in_range_industries = self.cvrs_satisfying_registration(registration_type='Industry')
        in_range_types = self.cvrs_satisfying_registration(registration_type='CompanyType')
        in_range_status = self.cvrs_satisfying_registration(registration_type='Status')
        in_range_employees = self.cvrs_satisfying_employees()
        in_range_financials = self.cvrs_satisfying_financials()
        
        non_empty_lists = [lst for lst in [in_range_employees, in_range_industries, in_range_types, in_range_status, in_range_financials] if len(lst) > 0]
        cvrs_with_criteria = reduce(np.intersect1d, (non_empty_lists))
        
        result = base_population.loc[base_population.index.isin(cvrs_with_criteria)]
    
        assert isinstance(result, pd.DataFrame)
        return result


    @save_pickle(
        DATA_ROOT / "processed/populations/{self.name}/target",
        on_validation_error="error",
    )
    def target(self) -> pd.DataFrame:
        """
        Looks up bankruptcy status from Registrations table in the specified period. 
        Return pandas with [CVR, TARGET_UK, TARGET_UT] where (UK = under konkurs, UT = under tvangsopløsning)
        """
        
        bankruptcy_status = ['UNDER KONKURS', 'TVANGSOPLØST']
        target_columns = ["TARGET_UK", "TARGET_UT"]

        # load the target data
        target_csv = [file for file in self.target_path.iterdir() if file.is_file() and file.suffix == '.csv']
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

        # filter on the bankruptcy status
        df_target = df_target.loc[lambda x: x.ChangeType == 'Status']
        df_target = df_target.loc[df_target.NewValue.isin(bankruptcy_status)]

        # filter on the period of interest
        df_target['FromDate'] = pd.to_datetime(df_target['FromDate'], errors='coerce')
        df_target = df_target.dropna(subset=['FromDate'])
        df_target = df_target.loc[(df_target.FromDate >= self._period_start) & (df_target.FromDate <= self._period_end)]

        # filter on relevant CVRs
        population_ids = self.sub_population().index.to_numpy()
        df_target = df_target.loc[df_target.CVR.isin(population_ids)]
        df_target = df_target.drop_duplicates(subset=['CVR', 'NewValue'])
        
        # left join the relevant values on to the population
        result = pd.DataFrame(index=population_ids)

        for i, status in enumerate(bankruptcy_status):
            target = df_target.loc[df_target.NewValue == status]
            target[target_columns[i]] = 1
            result = result.join(target.set_index('CVR')[target_columns[i]], how='left')
        result = result.fillna(0)
        
        # target is one if either of the two columns are 1
        result['TARGET'] = result[target_columns].max(axis=1)
        cat_type = pd.api.types.CategoricalDtype(categories=[0,1], ordered=False)
        result['TARGET'] = result['TARGET'].astype(cat_type)

        assert result.shape[0] == population_ids.shape[0]
        assert isinstance(result, pd.DataFrame)
        return result
    



    def cvrs_satisfying_registration(self, registration_type) -> list:
        """ Returns a list of CVRs that satisfy a given registration filter at the cutoff date (self._period_start).
        Args:
            registration_type (str): The type of registration to filter on Either 'Status', 'Industry', 'CompanyType']
         """
        
        base_population = self.base_population.population()
        filter = {
            'Status': self.status_filters,
            'Industry': self._industry_filters,
            'CompanyType': self.company_type_filters
        }
        # return empty list if no filter is set
        if not filter[registration_type]:
            print(f"No filter set for registration type: {registration_type}")
            return []
        print(f"Filtering on registration type: {registration_type}")

        registrations_path = DATA_ROOT / "Tables" / "Registrations"
        registrations_csv = [file for file in registrations_path.iterdir() if file.is_file() and file.suffix == '.csv']
        df_registrations = dd.read_csv(
            registrations_csv,
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

        # filter to show industry changes
        df_registrations = df_registrations.loc[df_registrations.ChangeType == registration_type]
        df_registrations['FromDate'] = pd.to_datetime(df_registrations['FromDate'], errors='coerce')
        df_registrations = df_registrations.dropna(subset=['FromDate']).sort_values(by='FromDate')
        base_population['cutoff_date'] = self._period_start

        # merge asof values and filter
        merged = pd.merge_asof(base_population, df_registrations, left_on='cutoff_date', right_on='FromDate', by='CVR')
        
        if registration_type == 'Industry':
            merged['industry4digit'] = merged['NewValue'].str[:4]
            cvrs_in_criteria = merged.loc[merged.industry4digit.isin(self._industry_filters)].CVR.to_numpy()
        else:
            cvrs_in_criteria = merged.loc[merged.NewValue.isin(filter[registration_type])].CVR.to_numpy()

        return cvrs_in_criteria
    




    def cvrs_satisfying_employees(self) -> list:
        """ Returns a list of CVRs that satisfy the employee filter """
        
        if not self.employee_filters:
            print(f"No filter set for employees")
            return []
        print(f"Filtering on employees with [min, max]: {self.employee_filters}")
        # get the financials data 
        employee_path = DATA_ROOT / "Tables" / "EmployeeCounts"
        employees_csv = [file for file in employee_path.iterdir() if file.is_file() and file.suffix == '.csv']
        df_employees = dd.read_csv(
            employees_csv,
            usecols=['CVR', 'FromDate', 'EmployeeCounts'],
            on_bad_lines="error",
            assume_missing=True,
            dtype={
                    "CVR": int,
                    "FromDate": str,
                    "EmployeeCounts": float,
                },
            blocksize="256MB"
        ).compute()

        # apply filters on the annual reports in the year up until the cutoff date, take the most recent employee entry
        df_employees['FromDate'] = pd.to_datetime(df_employees['FromDate'], errors='coerce')
        df_employees = df_employees.loc[(df_employees.FromDate < self._period_start) & (df_employees.FromDate >= self._period_start - pd.DateOffset(years=1))]
        df_employees = df_employees.sort_values(by=['CVR', 'FromDate'], ascending=False).drop_duplicates(subset='CVR', keep='first')

        df_employees = df_employees.loc[(df_employees.EmployeeCounts >= self.employee_filters[0]) & \
                        (df_employees.EmployeeCounts <= self.employee_filters[1])]
        
        return df_employees.CVR.to_numpy()
    


    def cvrs_satisfying_financials(self) -> list:
        """ Returns a list of CVRs that satisfy the financials filters. Filters allowed ["ProfitLoss", "Equity", "Assets", "LiabilitiesOtherThanProvisions"] """
       
        if not self.financials_filters:
            print(f"No filter set for financials")
            return []
        assert all([key in ["ProfitLoss", "Equity", "Assets", "LiabilitiesOtherThanProvisions"] for key in self.financials_filters.keys()]), "Invalid financials key"
        print(f"Filtering on financial keys: {self.financials_filters.keys()}")
        # get the financials data 
        financials_path = DATA_ROOT / "Tables" / "Financials"
        financials_csv = [file for file in financials_path.iterdir() if file.is_file() and file.suffix == '.csv']
        df_financials = dd.read_csv(
            financials_csv,
            usecols=["CVR","PublicationDate", "Currency", "ProfitLoss", "Equity", "Assets", "LiabilitiesOtherThanProvisions"],
            on_bad_lines="error",
            assume_missing=True,
            dtype={
                    "CVR": int,
                    "PublicationDate": str,
                    "Currency": str,
                    "ProfitLoss": float,
                    "Equity": float,
                    "Assets": float,
                    "LiabilitiesOtherThanProvisions": float
                },
            blocksize="256MB"
        ).compute()
        
        # TODO: convert currency to DKK
        print("Warning: Currency conversion not implemented")

        # apply filters on the annual reports in the year up until the cutoff date, remove companies with nans in the specified financials
        df_financials['PublicationDate'] = pd.to_datetime(df_financials['PublicationDate'], errors='coerce')
        df_financials = df_financials.sort_values(by=['CVR', 'PublicationDate'], ascending=False).drop_duplicates(subset='CVR', keep='first')
        df_financials = df_financials.loc[(df_financials.PublicationDate <= self._period_start) & (df_financials.PublicationDate >= self._period_start - pd.DateOffset(years=1))]
        df_financials = df_financials.dropna(subset=self.financials_filters.keys())

        for key in self.financials_filters.keys():
            df_financials = df_financials.loc[(df_financials[key] >= self.financials_filters[key][0]) & \
                                                (df_financials[key] <= self.financials_filters[key][1])]
            
        return df_financials.CVR.to_numpy()


    @save_pickle(DATA_ROOT / "processed/populations/{self.name}/data_split")
    def data_split(self) -> DataSplit:
        """Split data based on subpopulation"""
        base_splits = self.base_population.data_split()
        current_population = self.population().index.to_numpy()

        train_ids = np.intersect1d(base_splits.train ,current_population, assume_unique=True)
        val_ids = np.intersect1d(base_splits.val ,current_population, assume_unique=True)
        test_ids = np.intersect1d(base_splits.test ,current_population, assume_unique=True)

        assert current_population.shape[0] == train_ids.shape[0] + test_ids.shape[0] + val_ids.shape[0]
        
        return DataSplit(
            train=train_ids,
            val=val_ids,
            test=test_ids,
        )

    def prepare(self) -> None:
        """
        Prepares the data by calling the :meth:`population` and :meth:`data_split`.
        """
        self.sub_population()
        self.target()
        self.population()
        self.data_split()




if __name__ == "__main__":
    global_population = FromAnnualReports(AnnualReportTokens())
    pop = BankruptcySubPopulation(global_population)
    pop.prepare()
