import pandas as pd
import os
import dask.dataframe as dd

# df_employee = pd.read_csv(r'/Users/nikolaibeckjensen/Dropbox/Virk2Vec/Tables/EmployeeCounts/chunk0.csv', index_col=0)
# df_registrations = pd.read_csv(r'/Users/nikolaibeckjensen/Dropbox/Virk2Vec/Tables/Registrations/chunk0.csv', index_col=0)

def enrich_with_asof_values(df: pd.DataFrame, df_registrations: pd.DataFrame, values=['Industry', 'CompanyType', "Address", 'Status']) -> pd.DataFrame:
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

def dd_enrich_with_asof_values(df: dd.DataFrame, df_registrations: dd.DataFrame, values=['Industry', 'CompanyType', 'Address', 'Status'], 
                               date_col_df='FromDate', 
                               date_col_registrations='FromDate'
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
        df_value = df_value[['CVR', date_col_registrations, 'NewValue']].rename(columns={
            date_col_registrations: date_col_df,  # Align the date column names
            'NewValue': value  # Rename 'NewValue' to the specific value being processed
        })

        # Perform the asof merge using Dask's merge_asof function
        df = dd.merge_asof(
            df.sort_values(date_col_df),
            df_value.sort_values(date_col_df),
            on=date_col_df,
            by='CVR',
            direction='backward'
        )

    return df