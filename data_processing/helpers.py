import pandas as pd
import os

def date_chunks(date_start, date_end, days_per_query) -> list:
    """
    Creates a list of date intervals [date_interval_start, date_interval_end] starting from date_start and ending with date_end. Each interval is days_per_query long.
    Note: Both 'date_interval_start' and 'date_interval_end' is inclusive. Final interval may be shorter than days_per_query. 
    :param date_start: str, start date in the format "YYYY-MM-DD"
    :param date_end: str, end date in the format "YYYY-MM-DD"
    :param days_per_query: int, number of days in each query
    return: list of lists, each containing two strings [date_interval_start, date_interval_end]
    """

    # convert to pd.datetime to allow for date arithmetic
    query_date_start = pd.to_datetime(date_start)
    query_date_end = pd.to_datetime(date_end)

    # create a list of dates with [from_date, to_date] for each query. Both to_date and from_date is inclusive. 
    query_dates = []
    from_date = query_date_start
    while from_date <= query_date_end:
        to_date = from_date + pd.DateOffset(days=days_per_query-1)
        if to_date > query_date_end:
            to_date = query_date_end
        query_dates.append([from_date.strftime('%Y-%m-%d'), to_date.strftime('%Y-%m-%d')])
        from_date = to_date + pd.DateOffset(days=1)

    # convert to string
    query_dates = [[str(date) for date in dates] for dates in query_dates]
    
    return query_dates


def unique_cvr(xml_folder) -> list:
    """
    Fetches all unique CVR numbers from a folder containing csv files with all CVR numbers that have published a financial report in xml format.
    :param xml_folder: str, path to folder containing csv files with CVR numbers
    :return: set, containing all unique CVR numbers
    """

    unique_cvr = set()
    for file in os.listdir(xml_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(xml_folder, file)
            df = pd.read_csv(file_path, index_col=0)

            # get all unique values in the CVR column
            cvr_list = df.CVR.unique().astype(int).astype(str).tolist()
            unique_cvr.update(cvr_list)
    
    return list(unique_cvr)


#
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