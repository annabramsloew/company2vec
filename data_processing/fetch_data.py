import argparse
import os
import json
import pandas as pd
import time
from VirkConnection import VirkConnection
from queries import xbrl_reports_query, cvr_query
from helpers import date_chunks, unique_cvr
from xml_parser import fetch_financials
import datetime as dt
#from google.colab import drive


parser = argparse.ArgumentParser()
# query
parser.add_argument("--query", default='xml_reports', type=str)

# necessary xml reports query arguments
parser.add_argument("--query_date_start", default=None, type=str)
parser.add_argument("--query_date_end", default=None, type=str)
parser.add_argument("--days_per_query", default=30, type=int)
parser.add_argument("--local_save_path", default=None, type=str)

# necessary arguments for cvr query and xbrl parser
parser.add_argument("--xml_reports_folder", default=None, type=str) # folder containing csv files with CVR numbers
parser.add_argument("--table_folder") # folder to save the tables (must contain subfolders for each table: Registrations, EmployeeCounts, ProductionUnits, CompanyInfo, Participants, Financials)

# additional arguments for xbrl parser
parser.add_argument("--overwrite", default="False", type=str)  # overwrite existing chunks, otherwise continues from the latest chunk
parser.add_argument("--chunk_size", default=10000, type=int)  # chunk size for the xbrl parser

# paths

parser.add_argument("--virk_credentials_path", default=None, type=str)
args = parser.parse_args()


if args.query != 'xbrl_parser':
    # read credentials from file and assert that user and password is present
    with open(args.virk_credentials_path) as f:
        credentials = json.load(f)
    assert 'user' in credentials.keys(), 'User not found in credentials'
    assert 'password' in credentials.keys(), 'Password not found in credentials'



if args.query == 'xml_reports':

    # split the query date range into smaller chunks to avoid timeouts
    date_intervals = date_chunks(args.query_date_start, args.query_date_end, args.days_per_query)
    assert len(date_intervals) > 0, 'No date intervals found'
    
    # iterate through the query chunks:
    entries = []
    for date_interval in date_intervals:
        print()
        print("Querying data for date interval: ", date_interval)
        query = xbrl_reports_query(date_interval[0], date_interval[1], result_size=1000)

        connection = VirkConnection(
                credentials = credentials,
                query = query,
                endpoint = r"offentliggoerelser",  
                MAX_ITERATIONS = 100
            )   
        
        connection.execute_query()
        connection.parse_results()
        entries += connection.parsed_data
        time.sleep(10)
        
    results = pd.DataFrame(entries, columns = ["CVR", "PublicationDate", "UrlXML"])
    results.to_csv(args.local_save_path)
    


elif args.query == 'cvr_tables':
    cvr_list = unique_cvr(args.xml_reports_folder)
    print("CVR list length: ", len(cvr_list))

    # testing
    #cvr_list = ["35657339", "29140774", "37375675"]
    #df_cvr = pd.DataFrame(cvr_list, columns=['CVR'])



    # split the cvr list into chunks of 50,000 to avoid timeouts
    index_intervals = [i for i in range(0, len(cvr_list), 10000)]
    

    # iterate through the query chunks:
    for i in range(2):#range(len(index_intervals)):
        start_time = time.time()
        print("iteration: ", i) 

        # fetch the relevant cvr numbers
        if i == len(index_intervals)-1:
            cvr_list_current = cvr_list[index_intervals[i]:]
        else:
            cvr_list_current = cvr_list[index_intervals[i]:index_intervals[i+1]]

        print()
        print("Querying data from index: ", index_intervals[i])

        # create the query
        query = cvr_query(cvr_list_current, result_size=10000)

        connection = VirkConnection(
                credentials = credentials,
                query = query,
                endpoint = r"cvr-permanent/virksomhed",  
                MAX_ITERATIONS = 100
            ) 
        connection.execute_query()
        print("Query executed")
        connection.parse_results()
        print("Data parsed")

        # save the data
        print("Saving data")
        df_registrations = pd.DataFrame(connection.parsed_data[0], columns=['CVR', 'FromDate', 'ChangeType', 'NewValue']).to_csv(args.table_folder + "/Registrations" + f'/chunk{i}.csv')
        df_employees = pd.DataFrame(connection.parsed_data[1], columns=["CVR", "FromDate", "ChangeType", "EmployeeCounts"]).to_csv(args.table_folder + "/EmployeeCounts" + f'/chunk{i}.csv')
        df_production_units = pd.DataFrame(connection.parsed_data[2], columns=[ "CVR", "UnitNumber", "Date", "ChangeType"]).to_csv(args.table_folder + "/ProductionUnits" + f'/chunk{i}.csv')
        df_company_info = pd.DataFrame(connection.parsed_data[3], columns=["CVR", "Name", "StartDate", "EndDate", "CompanyType", "CompanyTypeCode", "ProductionUnits", "ZipCode", "Industry", "IndustryCode", "Status"]).to_csv(args.table_folder + "/CompanyInfo" + f'/chunk{i}.csv')
        df_participants = pd.DataFrame(connection.parsed_data[4], columns=['CVR','EntityID','Name', 'ParticipantType','RelationType', 'Participation', 'Date', 'EquityPct' ]).to_csv(args.table_folder + "/Participants" + f'/chunk{i}.csv')
        
        end_time = time.time()
        print("Time elapsed fetching and saving 50k results: ", end_time - start_time)

        time.sleep(10)





elif args.query == 'xbrl_parser':

    print("Loading data from xml_links folder")
    #load data from xml_links folder
    data = pd.DataFrame(columns=['CVR', 'PublicationDate', 'UrlXML'])
    for i in range(2013, 2024):
        path = args.xml_reports_folder + "/" + str(i) + '.csv'
        df = pd.read_csv(path, index_col=0)
        data = pd.concat([data, df])

    print("Filtering data")
    # filter out CVR numbers with less than 5 reports
    cvr_counts = data['CVR'].value_counts()
    cvr_counts = cvr_counts[cvr_counts >= 5].index
    data = data[data['CVR'].isin(cvr_counts)]
    del cvr_counts

    selected_keys = ['GrossProfitLoss', 'EmployeeBenefitsExpense', 'WagesAndSalaries', 'ProfitLoss', 'OtherFinanceIncome','OtherFinanceExpenses',
                 'NonCurrentAssets','CurrentAssets','CashAndCashEquivalents','Assets', 'Equity', 'ShorttermLiabilitiesOtherThanProvisions',
                 'LongtermLiabilitiesOtherThanProvisions','ShorttermDebtToBanks','LiabilitiesOtherThanProvisions','LiabilitiesAndEquity']
    
    # define the chunk sizes
    CHUNK_SIZE = args.chunk_size
    overwrite = args.overwrite
    index_intervals = [idx for idx in range(0, len(data), CHUNK_SIZE)]

    for i in range(0, len(index_intervals)):

        # check if the chunk already exists
        chunk_path = args.table_folder + "/Financials" + f'/chunk{i}.csv'
        if os.path.exists(chunk_path):
            if overwrite == 'True':
                print(f"Overwriting: Chunk {i} already exists at path {chunk_path}")
            else:
                print(f"Skipping: Chunk {i} already exists at path {chunk_path}")
                continue
                
        else:
            print(f"Chunk {i} does not exist at path {chunk_path}")

        print(f"Loading chunk {i}")
        start_time = time.time()

        # define chunk start and end
        chunk_start_index = index_intervals[i]
        if i == len(index_intervals)-1:
            chunk_end_index = len(data)
        else:
            chunk_end_index = index_intervals[i+1]

        # fetch the financials
        lines = []
        for j in range(chunk_start_index, chunk_end_index):
            cvr = data.iloc[j]['CVR']
            publicationdate = data.iloc[j]['PublicationDate']
            url = data.iloc[j]['UrlXML']

            line = fetch_financials(url, cvr, publicationdate, selected_keys)
            if line != None:
                lines.append(line)

        df_financials = pd.DataFrame(lines, columns=['CVR', "PublicationDate", "AuditClass", "ReportType", "Currency"] + selected_keys).to_csv(args.table_folder + "/Financials" + f'/chunk{i}.csv')
        end_time = time.time()
        print(f"Time elapsed fetching and saving {CHUNK_SIZE} results: ", end_time - start_time)

else:
    raise ValueError('Invalid query type')
