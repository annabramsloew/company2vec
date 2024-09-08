# ----------------------------- SET UP ----------------------------- #
import pandas as pd
import numpy as np
import requests
import xmltodict

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

#load data from xml_links folder

# SET YOUR PATH HERE
#main_path = r'/Users/annabramslow/Library/CloudStorage/Dropbox/DTU/Virk2Vec/xml_links/'

#load data from xml_links folder
def load_data(main_path):
    data = pd.DataFrame(columns=['CVR', 'PublicationDate', 'UrlXML'])
    for i in range(2013, 2024):
        path = main_path + str(i) + '.csv'
        print(path)
        df = pd.read_csv(path, index_col=0)
        data = pd.concat([data, df])

    #find out number of CVR with value counts >= 5
    cvr_counts = data['CVR'].value_counts()
    cvr_counts = cvr_counts[cvr_counts >= 5]
    data = data[data['CVR'].isin(cvr_counts)]

    return data


# ----------------------------- HELPER FUNCTIONS ----------------------------- #
#parse xml data from url
def parse_xml(url, timeout=10):
    try:
        response = requests.get(url, timeout=timeout)
        data = xmltodict.parse(response.content)
        return data
    except:
        return None
    
#find first key in xml data
def find_first_key(xml_data):
    if xml_data == None:
        return None
    else:
        for key in xml_data.keys():
            if 'xbrl' in key:
                first_key = key
                return first_key
            
#find audit class in xml data
def find_audit_class(xml_data, first_key):
    if xml_data == None:
        return None
    else:
        for key in xml_data[first_key].keys():
            if 'ClassOfReportingEntity' in key:
                try:
                    audit_class = xml_data[first_key][key]['#text']
                    return audit_class
                except:
                    audit_class = xml_data[first_key][key][0]['#text']
                    return audit_class


#find report type (annual, half-yearly, etc.)
def find_report_type(xml_data, first_key):    
    if xml_data == None:
        return None
    else:
        for key in xml_data[first_key].keys():
            if 'InformationOnTypeOfSubmittedReport' in key:
                try:
                    result = xml_data[first_key][key]
                    if isinstance(result, dict):
                        report_type = xml_data[first_key][key]['#text']
                        return report_type
                    elif isinstance(result, list):
                        report_type = xml_data[first_key][key][0]['#text']
                        return report_type
                    else:
                        report_type = None
                        return report_type
                    
                except:
                    report_type = None
                    return report_type


#find currency of the report
def find_currency(xml_data, first_key):    
    if xml_data is None:
        return None

    try:
        for key in xml_data[first_key].keys():
            if 'unit' in key:
                result = xml_data[first_key][key]
                
                # Handle dictionary case by converting it to a list
                if isinstance(result, dict):
                    result = [result]
                
                # Iterate through result list (handling both dict and list cases)
                for item in result:
                    if isinstance(item, dict):
                        for another_key in item.keys():
                            if 'measure' in another_key:
                                currency_info = item[another_key]

                                # Check if currency_info is a string and contains a colon
                                if isinstance(currency_info, str) and ':' in currency_info:
                                    parts = currency_info.split(':')
                                    # Ensure there are at least two parts after splitting
                                    if len(parts) > 1 and parts[1] in ['DKK', 'EUR', 'USD', 'CHF', 'GBP', 'SEK', 'NOK']:
                                        return parts[1]  # Return the found currency
                                
                                # Handle case where currency_info is a dictionary
                                elif isinstance(currency_info, dict):
                                    for subvalue in currency_info.values():
                                        # Check if subvalue is a string and contains a colon
                                        if isinstance(subvalue, str) and ':' in subvalue:
                                            parts = subvalue.split(':')
                                            # Ensure there are at least two parts after splitting
                                            if len(parts) > 1 and parts[1] in ['DKK', 'EUR', 'USD', 'CHF', 'GBP', 'SEK', 'NOK']:
                                                return parts[1]  # Return the found currency
        
        return None  # If no currency was found

    except:
        return None
        
# ----------------------------- FETCH FINANCIAL DATA ----------------------------- #

#selected keys
selected_keys = ['GrossProfitLoss', 'EmployeeBenefitsExpense', 'WagesAndSalaries', 'ProfitLoss', 'OtherFinanceIncome','OtherFinanceExpenses',
                 'NonCurrentAssets','CurrentAssets','CashAndCashEquivalents','Assets', 'Equity', 'ShorttermLiabilitiesOtherThanProvisions',
                 'LongtermLiabilitiesOtherThanProvisions','ShorttermDebtToBanks','LiabilitiesOtherThanProvisions','LiabilitiesAndEquity']

#fetch financial data from xml
def fetch_financials(url, cvr, publicationdate, selected_keys):
    
    columns=['CVR','PublicationDate', 'AuditClass','ReportType','Currency']+selected_keys
    
    xml = parse_xml(url)
    firstkey = find_first_key(xml)
    
    #find audit class
    audit_class = find_audit_class(xml, firstkey)

    #find report type
    report_type = find_report_type(xml, firstkey)

    #find currency in report
    currency = find_currency(xml, firstkey)

    if report_type == 'Årsrapport' or report_type == 'Annual report' or report_type == None:

        #find financial values, prepare lists to store values
        financial_values = []
        financial_keys = [None]*len(selected_keys)

        #find financial keys
        for i, value  in enumerate(selected_keys):
            for key in xml[firstkey].keys():
                try:
                    if key.split(':')[1] == value:
                        financial_keys[i] = key
                except:
                    pass
        
        #find financial values
        for value in financial_keys:
            if value == None:
                financial_values.append(None)
            else:
                try:
                    #take first value of the key ¯\_(ツ)_/¯
                    result = xml[firstkey][value]
                    if isinstance(result, dict):
                        financial_values.append(xml[firstkey][value]['#text'])
                    elif isinstance(result, list):
                        financial_values.append(xml[firstkey][value][0]['#text'])
                    else:
                        financial_values.append(None)
                except:
                    financial_values.append(None)

        line = [cvr, publicationdate, audit_class, report_type, currency]+financial_values
    
    else:
        #create list of None values
        #line = [None]*len(columns)
        line = None
    
    return line

# ----------------------------- CREATE DF ----------------------------- #
# load rows from flatfile and sort away none results, create df
"""
#test on df
financials = []

for i in range(len(df)):
    financials.append(fetch_financials(df['UrlXML'].iloc[i], df['CVR'].iloc[i], df['PublicationDate'].iloc[i], selected_keys))

#save financials in dataframe
columns=['CVR','PublicationDate', 'AuditClass','ReportType','Currency']+selected_keys
#pick only rows where line is not None in financials
financials = [line for line in financials if line is not None]
df_financials = pd.DataFrame(financials, columns=columns)
"""