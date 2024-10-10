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


def compute_experience_old(df: pd.DataFrame) -> pd.Series:
    """
    Computes an experience proxy for each entry in the provided DataFrame, representing the count of unique CVR 
    numbers that have been entered before the current entry for each EntityID.

    The function operates as follows:
    - Groups the DataFrame by 'EntityID' to ensure calculations are done for each participant independently.
    - Sorts the entries for each EntityID by 'Date' to maintain a chronological order of participation.
    - Identifies 'entry' rows as valid for counting experience, 
      while 'exit' rows do not contribute to the experience.
    - For each 'entry', the function tracks unique CVR numbers seen up to that point, ensuring that duplicate entries 
      of the same CVR do not increase the experience count.

    Parameters:
    df (pd.DataFrame): A DataFrame containing at least the following columns:
        - 'EntityID': Unique identifier for each participant.
        - 'Date': Date of entry or exit.
        - 'Participation': Indicates if the record is an 'entry' or 'exit'.
        - 'CVR': The unique CVR number associated with the entry.

    Returns:
    pd.Series: A Pandas Series where the index corresponds to the original DataFrame's index and the values represent 
    the computed experience proxy for each entry.
    """
    # List to store tuples of index and experience values
    experience_records = []
    
    # Group the dataframe by EntityID to compute the experience per participant
    for entity_id, group in df.groupby('EntityID'):
        # Sort the group by Date to maintain chronological order for this participant
        group = group.sort_values(by='Date')
        
        # Create a helper column to identify only the 'entry' rows (since exits don't count towards experience)
        group['is_entry'] = group['Participation'] == 'entry'

        # List to hold unique CVR numbers for this participant
        unique_cvr = set()
        
        # Iterate over rows in the group to calculate experience
        for index, row in group.iterrows():
            if row['is_entry']:
                
                # remove the current CVR from the set of unique CVRs if it is already in the set - we do not count entering the same CVR multiple times as double experience
                if row['CVR'] in unique_cvr:
                    unique_cvr.remove(row['CVR'])
                
                # Experience is the length of unique CVR numbers seen so far
                experience_value = len(unique_cvr)
                
                # Add the current CVR to the set of unique CVRs
                unique_cvr.add(row['CVR'])
            else:
                # For non-entry rows, experience is the current experience minus the current CVR
                experience_value = len(unique_cvr)-1
            
            # Append the index and experience value
            experience_records.append((index, experience_value))
    
    # Convert the records to a Pandas Series with the original index
    experience_series = pd.Series(
        {index: experience for index, experience in experience_records}, 
        index=df.index
    )
    
    # Return the experience series as a new column
    return experience_series


def active_participants_per_year_old(df: dd.DataFrame) -> dd.DataFrame:
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

    # Compute the experience proxy for each participant row
    df['Experience'] = compute_experience(df)

    # Create a DataFrame to store results
    results = []

    # Group by CVR to process each company separately
    for cvr, group in df.groupby('CVR'):

        #find the max year, but not more than 2023
        max_year = group['Date'].dt.year.max()
        max_year = min(2023, max_year)

        #years from 2013 to max year
        years = list(range(2013, max_year+1))
        
        # Iterate through each year
        for year in years:
            # Determine the cutoff date (December 31st of the given year)
            cutoff_date = pd.Timestamp(year=year, month=12, day=31)
            
            # Filter participants who have 'entered' before or on the cutoff date
            df_entries = group[(group['Participation'] == 'entry') & (group['Date'] <= cutoff_date)]
            
            # Get participants who have 'exited' before the cutoff date
            df_exits = group[(group['Participation'] == 'exit') & (group['Date'] < cutoff_date)]
            
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

            # Select the relevant columns for active entries
            active_entries = active_entries[['CVR_entry','EntityID', 'RelationType', 'Date_entry', 'Experience_entry']]
            active_entries.columns = active_entries.columns.str.replace('_entry', '')
            
            # Store the result as a dictionary
            results.append({
                'FromDate': cutoff_date,
                'CVR': cvr,
                'EntityID': active_entries['EntityID'].tolist(), # used for validation
                'RelationTypes': active_entries['RelationType'].tolist(),
                'Experience': active_entries['Experience'].tolist()
            })

    # Convert the results into a Dask DataFrame
    result_dd = dd.from_pandas(pd.DataFrame(results), npartitions=1)

    return result_dd

def compute_experience(df_participants, EntityID, cutoff_date):
    """
    Computes the experience proxy for a given participant on a given cutoff date.
    
    Parameters:
    EntityID (str): The unique identifier of the participant.
    cutoff_date (pd.Timestamp): The cutoff date for the experience computation.
    df_participants (pd.DataFrame): A DataFrame containing all participant records.
    
    Returns:
    int: The number of unique companies the participant has been associated with.
    """
    
    #filter df to only include the participant of interest before the cutoff date
    df_participant = df_participants.loc[(df_participants.EntityID == EntityID) & (df_participants.Date <= cutoff_date)]

    #count the unique companies
    return df_participant.CVR.nunique()

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
    results = []

    counter = 0

    # Group by CVR to process each company separately
    for cvr, group in df.groupby('CVR'):

        counter += 1
        # Print status update every 10,000 companies processed
        if counter % 10000 == 0:
            print(f"Processed {counter} companies...")

        #find the max year, but not more than 2023
        max_year = group['Date'].dt.year.max()
        max_year = min(2023, max_year)

        #years from 2013 to max year
        years = list(range(2013, max_year+1))
        
        # Iterate through each year
        for year in years:
            # Determine the cutoff date (December 31st of the given year)
            cutoff_date = pd.Timestamp(year=year, month=12, day=31)
            
            # Filter participants who have 'entered' before or on the cutoff date
            df_entries = group[(group['Participation'] == 'entry') & (group['Date'] <= cutoff_date)]
            
            # Get participants who have 'exited' before the cutoff date
            df_exits = group[(group['Participation'] == 'exit') & (group['Date'] < cutoff_date)]
            
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

            # Select the relevant columns for active entries
            active_entries = active_entries[['CVR_entry','EntityID', 'RelationType', 'Date_entry']]
            active_entries.columns = active_entries.columns.str.replace('_entry', '')

            #compute experience for each active participant at the cutoff date if active_entries is not empty else return None
            if not active_entries.empty:
                active_entries['Experience'] = active_entries.apply(lambda x: compute_experience(df, x['EntityID'], cutoff_date), axis=1)
            else:
                active_entries['Experience'] = None
            
            # Store the result as a dictionary
            results.append({
                'FromDate': cutoff_date,
                'CVR': cvr,
                'EntityID': active_entries['EntityID'].tolist(), # used for validation
                'RelationTypes': active_entries['RelationType'].tolist(),
                'Experience': active_entries['Experience'].tolist()
            })

    # Print final status update
    print(f"Processing complete. Total companies processed: {counter}")

    # Convert the results into a Dask DataFrame
    result_dd = dd.from_pandas(pd.DataFrame(results), npartitions=1)

    return result_dd
