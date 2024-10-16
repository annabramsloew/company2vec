import pandas as pd
import os
import dask.dataframe as dd
from ..serialize import DATA_ROOT

# df_employee = pd.read_csv(r'/Users/nikolaibeckjensen/Dropbox/Virk2Vec/Tables/EmployeeCounts/chunk0.csv', index_col=0)
# df_registrations = pd.read_csv(r'/Users/nikolaibeckjensen/Dropbox/Virk2Vec/Tables/Registrations/chunk0.csv', index_col=0)

def enrich_with_asof_values(df: pd.DataFrame, df_registrations: pd.DataFrame, values=['Industry', 'CompanyType', "Municipality", 'Status']) -> pd.DataFrame:
    """ Adds the as-of values of df_registrations for the entries in df.
    Currently relies on a CVR and FromDate column in both dataframes."""

    for value in values:
        
        # load data
        df_value = df_registrations.loc[df_registrations['ChangeType'] == value].reset_index(drop=True)

        # Convert the 'Date' columns to datetime for proper comparison if they arent already datetime
        if value == 'Status':
            # replace nan values with 2000-01-01
            df_value['FromDate'] = df_value['FromDate'].fillna('2000-01-01')

        # convert to datetime
        if df.FromDate.dtype != 'datetime64[ns]':
            df['FromDate'] = pd.to_datetime(df['FromDate'])
        if df_value.FromDate.dtype != 'datetime64[ns]':
            df_value['FromDate'] = pd.to_datetime(df_value['FromDate'])
        
        # format columns
        df_value = df_value[['CVR', 'FromDate', 'NewValue']].rename(columns={'NewValue': value})

        df = pd.merge_asof(df.sort_values('FromDate'),
                    df_value.sort_values('FromDate'),
                    on='FromDate',
                    by='CVR',
                    direction='backward')
                

    return df

def dd_enrich_with_asof_values(df: dd.DataFrame, df_registrations: dd.DataFrame, values=['Industry', 'CompanyType', 'Municipality', 'Status'], 
                               date_col_df='FromDate', 
                               date_col_registrations='FromDate',
                               left_by_value='CVR',
                               right_by_value='CVR'
                               ) -> dd.DataFrame:
    """
    Adds the as-of values of df_registrations for the entries in df.
    Allows different date column names for both dataframes.
    Made for dask dataframes.
    """

    for value in values:
        # Load data by filtering df_registrations on ChangeType
        df_value = df_registrations.loc[df_registrations['ChangeType'] == value].reset_index(drop=True)

        # Special handling for 'Status' value: replace NaN dates with '2000-01-01'
        if value == 'Status':
            df_value[date_col_registrations] = df_value[date_col_registrations].fillna('2000-01-01')

        # Convert date columns to datetime if they are not already in datetime format
        df[date_col_df] = dd.to_datetime(df[date_col_df], errors='coerce')
        df_value[date_col_registrations] = dd.to_datetime(df_value[date_col_registrations], errors='coerce')

        # Select relevant columns and rename 'NewValue' to the current value
        df_value = df_value[[right_by_value, date_col_registrations, 'NewValue']].rename(columns={
            date_col_registrations: date_col_df,  # Align the date column names
            'NewValue': value  # Rename 'NewValue' to the specific value being processed
        })

        # sort the dataframes by the date column
        df = df.set_index(date_col_df, drop=False).reset_index(drop=True)
        df = df.sort_values(date_col_df)
        df_value = df_value.set_index(date_col_df, drop=False).reset_index(drop=True)
        df_value = df_value.sort_values(date_col_df)

        # Perform the asof merge using Dask's merge_asof function
        df = dd.merge_asof(
            df,
            df_value,
            on=date_col_df,
            left_by=left_by_value,
            right_by=right_by_value,
            direction='backward'
        )

    return df

def convert_currency( df: dd.DataFrame, lookup_table: dd.DataFrame, 
                     amount_cols=['amount'],  # List of columns to convert
                     currency_col='currency', 
                     date_col='PublicationDate',  # The datetime column in df
                     ) -> dd.DataFrame:
    """
    Convert multiple currency columns in the DataFrame based on a lookup table for rows where currency is not 'DKK'.
    
    Args:
        df (dd.DataFrame): The main dataframe with amounts to convert.
        lookup_table (dd.DataFrame): The lookup table containing conversion rates.
        amount_cols (list): List of column names in df containing amounts to be converted.
        currency_col (str): Column name in df containing currency information.
        date_col (str): Column name for datetime in df to extract year and month from.
    
    Returns:
        dd.DataFrame: DataFrame with currency conversion applied to the specified amount columns.
    """

    # 1. Filter rows where the currency is not 'DKK'
    non_dkk_df = df[df[currency_col] != 'DKK']
    dkk_df = df[df[currency_col] == 'DKK']  # Keep these rows unchanged

    # 2. Extract year and month from the PublicationDate column
    non_dkk_df['year'] = non_dkk_df[date_col].dt.year
    non_dkk_df['month'] = non_dkk_df[date_col].dt.month

    # 3. Perform a join to find the correct rate for each non-DKK row
    merged_df = dd.merge(
        non_dkk_df,
        lookup_table,
        left_on=[currency_col, 'year', 'month'],
        right_on=['from_currency', 'year', 'month'],
        how='left'
    )

    # 4. Apply currency conversion for each amount column
    for col in amount_cols:
        merged_df[col] = merged_df[col] * merged_df['rate']

    # 5. Drop unnecessary columns (from the lookup table like 'rate', 'from_currency')
    merged_df = merged_df.drop(columns=['rate', 'from_currency', 'year', 'month'])

    # 6. Concatenate the DKK and non-DKK dataframes
    final_df = dd.concat([dkk_df, merged_df])

    return final_df

def active_participants_per_year(df: dd.DataFrame) -> dd.DataFrame:
    """
    Computes a summary of active participants per company ('CVR') per year based on
    the provided DataFrame. The function groups records by company and iterates over
    years from 2013 to the latest available year (capped at 2023). It filters active
    entries per year based on entry and exit dates, and returns a summary DataFrame
    with experience data.
    
    Parameters:
    df (dd.DataFrame): A Dask DataFrame containing participant records with 
                       columns 'CVR', 'Date', 'Participation', 'EntityID', 
                       'RelationType', and 'Experience'.
    
    Returns:
    dd.DataFrame: A Dask DataFrame summarizing experience data per company ('CVR') 
                  and year, with details on participants' experience levels and 
                  relationship types. A row per company per year with lists of 
                  experience corresponding relation types.
    """

    # Compute input dd.DataFrame
    df = df.compute()

    # Create a DataFrame to store results
    results = pd.DataFrame(columns=['FromDate', 'CVR', 'EntityID', 'RelationType', 'Experience'])

    #for hvert år først regn experience
    for year in range(2013, 2024):
        print(year)
        counter = 0

        cut_off_date = pd.Timestamp(year=year, month=12, day=31)
        df_current_year = df.loc[df['Date'] <= cut_off_date]

        df_experience = df_current_year.groupby('EntityID')['CVR'].nunique().reset_index()
        df_experience.columns = ['EntityID', 'Experience']

        df_current_year = df_current_year.merge(df_experience, on='EntityID', how='left')


        # Group by CVR to process each company separately
        for cvr, group in df_current_year.groupby('CVR'):

            counter += 1
            # Print status update every 10,000 companies processed
            if counter % 10000 == 0:
                print(f"Processed {counter} companies...")
            
            
            # Filter participants who have 'entered' before or on the cutoff date
            df_entries = group[(group['Participation'] == 'entry')]
            
            # Get participants who have 'exited' before the cutoff date
            df_exits = group[(group['Participation'] == 'exit')]
            
            # Merge entries and exits on EntityID and RelationType to pair entries with exits
            merged = pd.merge(
                df_entries,
                df_exits,
                on=['EntityID', 'RelationType'],
                suffixes=('_entry', '_exit'),
                how='left'
            )

            # Keep only entries where either:
            #   1. There is no exit, or
            #   2. The entry date is after the exit date
            active_entries = merged.loc[(merged['Date_exit'].isna()) | (merged['Date_entry'] > merged['Date_exit'])]

            # check if there are any active entries else continue
            if active_entries.empty:
                continue

            # Select the relevant columns for active entries
            active_entries = active_entries[['CVR_entry','EntityID', 'RelationType', 'Experience_entry']]
            active_entries.columns = active_entries.columns.str.replace('_entry', '')

            # create a column with the cut off date
            active_entries['FromDate'] = cut_off_date
            
            # append to results
            results = pd.concat([results, active_entries[['FromDate', 'CVR', 'EntityID', 'RelationType', 'Experience']]])

        #save the results
        path = DATA_ROOT / "interim" / "leadership"
        #save to parquet
        results.to_parquet(path / f"active_participants_{year}.parquet", index=False)
        
        # Print final status update
        print(f"Processing complete. Total companies processed: {counter}")

    # Convert the results into a Dask DataFrame
    results_dd = dd.from_pandas(results, npartitions=1)

    return results_dd


def bin_share(share):
    if pd.isna(share):
        return "SHARE_NA"
    elif share < 0.1:
        return "SHARE_0_10"
    elif share < 0.2:
        return "SHARE_10_20"
    elif share < 0.3:
        return "SHARE_20_30"
    elif share < 0.4:
        return "SHARE_30_40"
    elif share < 0.5:
        return "SHARE_40_50"
    elif share < 0.6:
        return "SHARE_50_60"
    elif share < 0.7:
        return "SHARE_60_70"
    elif share < 0.8:
        return "SHARE_70_80"
    elif share < 0.9:
        return "SHARE_80_90"
    else:
        return "SHARE_90_100"