
#load data from xml_links folder

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
                    report_type = xml_data[first_key][key]['#text']
                    return report_type
                    
                except:
                    report_type = None
                    return report_type


#find currency of the report
def find_currency(xml_data, first_key):    
    if xml_data == None:
        return None
    else:
        try:
            for measure in xml_data[first_key]['unit']:
                currency_info = measure['measure'].split(':')[1]
                if currency_info == 'DKK':
                    currency = currency_info
                elif currency_info == 'EUR':
                    currency = currency_info
                elif currency_info == 'USD':
                    currency = currency_info
                else:
                    currency = None
                return currency
        except:
            currency = None
            return currency
        
# ----------------------------- FETCH FINANCIAL DATA ----------------------------- #

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
        line = [None]*len(columns)
    
    return line