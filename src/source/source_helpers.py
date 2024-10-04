import pandas as pd
import os
import dask.dataframe as dd

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