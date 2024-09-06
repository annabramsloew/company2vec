import argparse
import os
import json
import pandas as pd
import time
from VirkConnection import VirkConnection
from queries import xbrl_reports_query, cvr_query
from helpers import date_chunks, unique_cvr
#from google.colab import drive


parser = argparse.ArgumentParser()
# query
parser.add_argument("--query", default='xml_reports', type=str)

# necessary xml reports query arguments
parser.add_argument("--query_date_start", default=None, type=str)
parser.add_argument("--query_date_end", default=None, type=str)
parser.add_argument("--days_per_query", default=30, type=int)
parser.add_argument("--local_save_path", default=None, type=str)

# necessary arguments for cvr query
parser.add_argument("--xml_reports_folder", default=None, type=str) # folder containing csv files with CVR numbers
parser.add_argument("--table_folder") # folder to save the tables (must contain subfolders for each table: Registrations, EmployeeCounts, ProductionUnits, CompanyInfo, Participants)


# paths

parser.add_argument("--virk_credentials_path", default=None, type=str)
args = parser.parse_args()



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
        query = cvr_query(cvr_list_current, result_size=1000)

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
        

else:
    raise ValueError('Invalid query type')
