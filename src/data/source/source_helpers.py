import pandas as pd
import os
import dask.dataframe as dd
from ..logging_config import DATA_ROOT

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

def enrich_with_asof_values_v2(df: pd.DataFrame, df_registrations: pd.DataFrame, values=['Industry', 'CompanyType', 'Municipality', 'Status'], 
                               date_col_df='FromDate', 
                               date_col_registrations='FromDate',
                               left_by_value='CVR',
                               right_by_value='CVR'
                               ) -> pd.DataFrame:
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
        df[date_col_df] = pd.to_datetime(df[date_col_df], errors='coerce')
        df_value[date_col_registrations] = pd.to_datetime(df_value[date_col_registrations], errors='coerce')

        # Select relevant columns and rename 'NewValue' to the current value
        df_value = df_value[[right_by_value, date_col_registrations, 'NewValue']].rename(columns={
            date_col_registrations: date_col_df,  # Align the date column names
            'NewValue': value  # Rename 'NewValue' to the specific value being processed
        })

        # sort the dataframes by the date column
        df = df.sort_values(date_col_df)
        df_value = df_value.sort_values(date_col_df)

        # Perform the asof merge using Dask's merge_asof function
        df = pd.merge_asof(
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

def generate_december_31_dates(min_date, max_date):
    """
    Generate all 31st December dates between min_date and max_date.
    
    Args:
        min_date (pd.Timestamp): The minimum date.
        max_date (pd.Timestamp): The maximum date.
    
    Returns:
        pd.DatetimeIndex: A DatetimeIndex containing all 31st December dates between min_date and max_date.
    """
    dates = pd.date_range(start=min_date, end=max_date, freq='A-DEC')
    return dates

def active_participants_per_year(df: dd.DataFrame) -> pd.DataFrame:
    """
    Computes a summary of active participants per company ('CVR') per year based on
    the provided DataFrame. Experience is computed as the time between entry in a given company
    and the yearly cutoff date.
    
    Parameters:
    df (dd.DataFrame): A Dask DataFrame containing participant records with 
                       columns 'CVR', 'Date', 'Participation', 'EntityID', 
                       'RelationType', and 'Experience'.
    
    Returns:
    dd.DataFrame: A Pandas DataFrame summarizing experience data per company ('CVR') 
                  and year, with details on participants' experience levels and 
                  relationship types.
    """
    
    #transition to pandas to compute experience
    df = df.compute()

    #sort by person, CVR, RelationType and Date
    df.sort_values(by=['EntityID','CVR','RelationType', 'Date'],inplace=True)
    df['identifier'] = df['CVR'].astype(str) + df['EntityID'].astype(str) + df['RelationType']
    #check if the previous row has the same identifier
    df['helper1']  = (df['identifier'] == df['identifier'].shift(1))

    # check if relation type is the same and the previous participation was an exit
    df['helper2'] = (df['RelationType'] == df['RelationType'].shift(1)) & (df['Participation'].shift(1) == 'exit')
    # find the first row of a new position for row identifier
    df['helper3'] = df['helper1'] * ~df['helper2']

    # add a row identifier which is incremented with one if the new_column is False
    df['row_identifier'] = (~df['helper3']).cumsum()

    # group by cvr, entity_id and relation type to get a column with entry and exit date
    df_grouped = df.groupby(['CVR','EntityID','RelationType','row_identifier']).agg({'Date':['min','max']}).reset_index()

    # if min == max, then set the max to todays date, as there is no exit date
    df_grouped.loc[df_grouped[('Date','min')] == df_grouped[('Date','max')],('Date','max')] = pd.to_datetime('today')

    #cut off date for experience calculation is 2013-01-01, all exits before this date are removed
    df_grouped = df_grouped.loc[df_grouped[('Date','max')] >= pd.to_datetime('2013-01-01')]

    # Assuming generate_december_31_dates is a function you defined earlier
    def generate_december_31_dates_vectorized(min_dates, max_dates):
        """Generate December 31 dates for vectorized min and max dates."""
        date_ranges = [list(pd.date_range(start=min_date, end=max_date, freq='A-DEC')) for min_date, max_date in zip(min_dates, max_dates)]
        return date_ranges

    # Vectorized generation of December 31 dates for all rows
    df_grouped['December31Dates'] = generate_december_31_dates_vectorized(
        df_grouped[('Date', 'min')],
        df_grouped[('Date', 'max')]
    )   
    # Remove rows with empty lists before exploding
    df_grouped = df_grouped[df_grouped['December31Dates'].apply(len) > 0]
    df_grouped.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_grouped.columns]
    df_expanded = df_grouped.explode('December31Dates_')
    df_expanded = df_expanded.dropna(subset=['December31Dates_'])
    df_expanded['Experience'] = (df_expanded.groupby(['CVR_', 'EntityID_','RelationType_', 'row_identifier_']).cumcount())


    # df_expanded = pd.DataFrame(columns=['Date','CVR', 'EntityID', 'RelationType','Experience'])
    # i = 0
    
    
    # # Iterate over each row in the dataframe
    # for _, row in df_grouped.iterrows():
    #     i += 1
    #     # Generate the 31st December dates
    #     dates = generate_december_31_dates(row[('Date', 'min')], row[('Date', 'max')])
    #     date_count = len(dates)
    #     # Create a new DataFrame for the expanded rows
    #     expanded_df = pd.DataFrame({
    #         'Date': dates,
    #         'CVR': [row['CVR'].item()] * date_count,
    #         'EntityID': [row['EntityID'].item()] * date_count,
    #         'RelationType': [row['RelationType'].item()] * date_count,
    #         'Experience' : list(range(date_count))
    #     })
    #     # Concatenate the expanded DataFrame to the main DataFrame
    #     df_expanded = pd.concat([df_expanded, expanded_df], ignore_index=True)

    #     if i % 10000 == 0:
    #         print(f'Processed {i} rows')

    #     # if i == 1000:
    #     #     break

    #filter away all rows where the date is before the cutoff date
    df_expanded = df_expanded.loc[df_expanded['December31Dates_'] >= pd.to_datetime('2013-01-01')].rename(columns={'December31Dates_': 'FromDate', 'CVR_': 'CVR', 'EntityID_': 'EntityID', 'RelationType_': 'RelationType'})
    df_expanded = df_expanded[['FromDate', 'CVR', 'RelationType', 'Experience']]
    # rename

    return df_expanded


def bin_share(share):
    if pd.isna(share):
        return "[UNK]"
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