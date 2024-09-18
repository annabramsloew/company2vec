import pandas as pd
import json


class VirkResult():
    """ Class for parsing the resulting data from the VirkConnection class """

    def __init__(self, hit_list):
        """
        :param hit_list: list of dictionaries, containing the data to be parsed
        """
        self.hit_list: list[dict] = hit_list


    def parse(self, endpoint) -> pd.DataFrame:
        """Parses the data from the fetched response"""

        if endpoint == 'offentliggoerelser':
            return self.xml_reports()
        
        elif endpoint == 'cvr-permanent/virksomhed':
            
            # Initialize lists for each of the tables
            registration_lines = []
            employee_lines = []
            production_unit_lines = []
            company_info_lines = []
            participant_lines = []

            for hit in self.hit_list:
                hit = hit['_source']
                cvr = hit['Vrvirksomhed']['cvrNummer']
                registration_lines += self.registrations(hit, cvr)
                employee_lines += self.employee_counts(hit, cvr)
                production_unit_lines += self.production_units(hit,  cvr)
                company_info_lines += self.company_info(hit, cvr)
                participant_lines += self.participants(hit, cvr)
                
            return registration_lines, employee_lines, production_unit_lines, company_info_lines, participant_lines
        else:
            raise NotImplementedError('Parsing for this endpoint is not implemented yet.')



    def registrations(self, hit, cvr) -> list:
        """
        Fetches all registrations related to 'Address', 'Industry', 'CompanyType', 'Name', 'Capital' for a given company hit. 
        Returns a dataframe with CVR, FromDate, ChangeType, NewValue
        :param hit: dict
        :param cvr: str
        :return: pd.DataFrame
        """

        lines = []

        lines += self.name_changes(hit, cvr)
        lines += self.address_changes(hit, cvr)
        lines += self.company_type_changes(hit, cvr)
        lines += self.industry_changes(hit, cvr)
        lines += self.capital_changes(hit, cvr)
        lines += self.status_changes(hit, cvr)

        return lines



    def name_changes(self, hit, cvr) -> list:
        """ Fetches all name changes for a given company hit.
        Returns a list of lists with values CVR, FromDate, ChangeType='Name', NewValue
        :param hit: dict
        :param cvr: str
        :return: list
        """
        lines = []
        for name in hit['Vrvirksomhed']['navne']:
            from_date = name['periode']['gyldigFra']
            new_value = name['navn']
            lines.append([cvr, from_date, 'Name', new_value])
        
        return lines


    def address_changes(self, hit, cvr) -> list:
        """ Fetches all address changes for a given company hit.
        Returns a list of lists with values: CVR=cvr, FromDate, ChangeType='Address', NewValue
        :param hit: dict
        :param cvr: str
        :return: list
        """

        lines = []
        # fetch all address changes
        for address in hit['Vrvirksomhed']['beliggenhedsadresse']:
            from_date = address['periode']['gyldigFra']
            new_value = address['postnummer']
            lines.append([cvr, from_date, 'Address', new_value])

        return lines


    def company_type_changes(self, hit, cvr) -> list:
        """ Fetches all address changes for a given company hit.
        Returns a list of lists with values: CVR=cvr, FromDate, ChangeType='CompanyType', NewValue
        :param hit: dict
        :param cvr: str
        :return: list
        """
        lines = []
        for company_type in hit['Vrvirksomhed']['virksomhedsform']:
            from_date = company_type['periode']['gyldigFra']
            new_value = company_type['kortBeskrivelse']
            lines.append([cvr, from_date, 'CompanyType', new_value])
        
        return lines


    def industry_changes(self, hit, cvr) -> list:
        """ Fetches all industry code changes for a given company hit.
        Returns a list of lists with values: CVR=cvr, FromDate, ChangeType='Industry', NewValue
        :param hit: dict
        :param cvr: str
        :return: list
        """
        lines = []
        for industry in hit['Vrvirksomhed']['hovedbranche']:
            from_date = industry['periode']['gyldigFra']
            new_value = industry['branchekode']
            lines.append([cvr, from_date, 'Industry', new_value])
        
        return lines
    

    def capital_changes(self, hit, cvr) -> list:
        """ Fetches all changes in capital structure (investments) for a given company hit.
        Returns a list of lists with values: CVR=cvr, FromDate, ChangeType='Capital', NewValue
        :param hit: dict
        :param cvr: str
        :return: list
        """
        # TODO: Enrich with the investment 'kurs'
        lines = []
        for attribute in hit['Vrvirksomhed']['attributter']:
            if attribute['type'] == 'KAPITAL':
                for values in attribute['vaerdier']:
                    new_value = float(values['vaerdi'].replace(',',''))
                    from_date = values['periode']['gyldigFra']
                    lines.append([cvr, from_date, 'Capital', new_value])
        return lines


    def status_changes(self, hit, cvr) -> list:
        """ Fetches all changes in company status for a given company hit.
        Returns a list of lists with values: CVR=cvr, FromDate, ChangeType='Status', NewValue
        :param hit: dict
        :param cvr: str
        :return: list
        """
        lines = []
        for status in hit['Vrvirksomhed']['virksomhedsstatus']:
            from_date = status['periode']['gyldigFra']
            new_value = status['status']
            lines.append([cvr, from_date, 'Status', new_value])
        
        return lines

    def employee_counts(self, hit, cvr) -> list:
        """
        Fetches all employee counts for a given company hit. 
        Returns a list of lists with CVR, FromDate, ChangeType='EmployeeCount', EmployeeCounts
        :param hit: dict
        :param cvr: str
        :return: list
        """

        kvartal_mapping = {
            1: "01-01",
            2: "04-01",
            3: "07-01",
            4: "10-01"
        }

        dates = []
        employee_counts = []

        # fetch all employee counts at the most granular level (monthly). maanedsbeskaeftigelse in some cases only holds values until 2019. afterwards use erstMaanedsbeskaeftigelse
        for entry in hit['Vrvirksomhed']['maanedsbeskaeftigelse']:
            employee_count = entry['antalAarsvaerk']
            date = str(entry['aar']) + "-" + str(entry['maaned']).zfill(2) + "-01"

            if employee_count != None:
                dates.append(date)
                employee_counts.append(int(employee_count))

        if 'erstMaanedsbeskaeftigelse' in hit['Vrvirksomhed'].keys():
            for entry in hit['Vrvirksomhed']['erstMaanedsbeskaeftigelse']:
                employee_count = entry['antalAarsvaerk']
                date = str(entry['aar']) + "-" + str(entry['maaned']).zfill(2) + "-01"

                if date not in dates and employee_count != None:
                    dates.append(date)
                    employee_counts.append(int(employee_count))

        # fill in missing dates from quarterly data
        for entry in hit['Vrvirksomhed']['kvartalsbeskaeftigelse']:
            employee_count = entry['antalAarsvaerk']
            date = str(entry['aar']) + "-" + kvartal_mapping[entry['kvartal']]

            if date not in dates and employee_count != None:
                dates.append(date)
                employee_counts.append(int(employee_count))

        # fill in missing dates from yearly data
        for entry in hit['Vrvirksomhed']['aarsbeskaeftigelse']:
            employee_count = entry['antalAarsvaerk']
            date = str(entry['aar']) + "-01-01"

            if date not in dates and employee_count != None:
                dates.append(date)
                employee_counts.append(int(employee_count))
        
        return list(zip([cvr]*len(dates), dates, ['EmployeeCount']*len(dates), employee_counts))
    
    def production_units(self, hit, cvr) -> list:
        """ Fetches all creations/deletions of production units for a given company hit.
        Returns a list of lists with values: CVR=cvr, UnitNumber, Date, ChangeType (Start/End)
        """

        lines = []

        for p_unit in hit['Vrvirksomhed']['penheder']:
            unit_number = p_unit['pNummer']
            date_start = p_unit['periode']['gyldigFra']
            date_end = p_unit['periode']['gyldigTil']

            lines.append([cvr, unit_number, date_start, 'Start'])

            if date_end != None:
                lines.append([cvr, unit_number, date_end, 'End'])

        return lines
    
    def xml_reports(self) -> list:
        """ Fetches the CVR, PublicationDate and XML URL for all entries in a given financial API hit.
        Returns a list of lists with values in the following order: CVR, PublicationDate, UrlXML,
        """

        lines = []

        for hit in self.hit_list:
            cvr = hit['_source']['cvrNummer']
            publication_date = hit['_source']['offentliggoerelsesTidspunkt'].split('T')[0]

            for document in hit['_source']['dokumenter']:
                if document['dokumentMimeType'] == 'application/xml':
                    url = document['dokumentUrl']
                    lines.append([cvr, publication_date, url])
        return lines
    

    def company_info(self, hit, cvr) -> list:
        """ Fetches all the latest company information for a given company hit.
        Returns a list of lists with values: CVR, Name, StartDate, EndDate, CompanyType, CompanyTypeCode, ProductionUnits, ZipCode, Industry, IndustryCode, Status"""

        try:
            name = hit['Vrvirksomhed']['virksomhedMetadata']['nyesteNavn']['navn']
            start_date = hit['Vrvirksomhed']['virksomhedMetadata']['stiftelsesDato']
            end_date = hit['Vrvirksomhed']['virksomhedMetadata']['nyesteNavn']['periode']['gyldigTil']
            company_type = hit['Vrvirksomhed']['virksomhedMetadata']['nyesteVirksomhedsform']['kortBeskrivelse']
            company_type_code = hit['Vrvirksomhed']['virksomhedMetadata']['nyesteVirksomhedsform']['virksomhedsformkode']
            p_units = int(hit['Vrvirksomhed']['virksomhedMetadata']['antalPenheder'])
            zipcode = int(hit['Vrvirksomhed']['virksomhedMetadata']['nyesteBeliggenhedsadresse']['postnummer'])
            industry = hit['Vrvirksomhed']['virksomhedMetadata']['nyesteHovedbranche']['branchetekst']
            industry_code = int(hit['Vrvirksomhed']['virksomhedMetadata']['nyesteHovedbranche']['branchekode'])
            status = hit['Vrvirksomhed']['virksomhedMetadata']['sammensatStatus']
        except:
            name, start_date, end_date, company_type, company_type_code, p_units, zipcode, industry, industry_code, status = None, None, None, None, None, None, None, None, None, None
            # save the hit as json for later inspection
            save_folder = r'/Users/nikolaibeckjensen/Desktop/inspections'
            with open(save_folder + f'/{cvr}.json', 'w') as f:
                json.dump(hit, f)

        return [[cvr, name, start_date, end_date, company_type, company_type_code, p_units, zipcode, industry, industry_code, status]]



    def add_participant(self, entity, value, cvr, relationtype, equity) -> list:
        """Adds a participant according to their type to the list of participants
        :param entity: dict, containing the entity information
        :param value: dict, containing the value information
        :param cvr: str, the cvr number of the company
        :param relationtype: str, the type of relation
        
        """
        lines_add = []

        #dates                
        time = value['periode']
        startdate = time['gyldigFra']
        enddate = time['gyldigTil']
        
        #entity ID - take cvr for businesses, otherwise 'unit id'
        if entity['deltager']['forretningsnoegle'] is not None:
            entity_id = entity['deltager']['forretningsnoegle']
        else:
            entity_id = entity['deltager']['enhedsNummer']
        
        #entity name (take most recent name)
        name = entity['deltager']['navne'][0]['navn']
        
        #participant type (company or person)
        participanttype = entity['deltager']['enhedstype']
        
        
        #append entry and exit lines
        lines_add.append([cvr, entity_id, name, participanttype, relationtype,'entry', startdate, equity])
        #print([cvr, entity_id, name, participanttype, relationtype,'entry', startdate, equity])

        if enddate is not None:
            lines_add.append([cvr, entity_id, name, participanttype, relationtype, 'exit', enddate, equity])
            #print([cvr, entity_id, name, participanttype, relationtype,'entry', startdate, equity])
        
        return lines_add



    def participants(self, hit, cvr) -> list:
        
        company = hit['Vrvirksomhed']
        
        lines_hit = []
        #loop through participants
        total_relations = len(company['deltagerRelation'])
        for entity in company['deltagerRelation']:
            for role in entity['organisationer']:
                for item in role['medlemsData']:
                    for function in item['attributter']:
                        
                        if function['type'] == 'EJERANDEL_PROCENT':
                            
                            #loop through roles of participant
                            for value in function['vaerdier']:
                                
                                #relation type
                                relationtype = "EJERANDEL"
                                
                                #equity
                                try:
                                    equity = float(value['vaerdi'].replace(',','.'))
                                except:
                                    equity = None
                                
                                #call participant parser for owners
                                lines_hit += self.add_participant(entity, value, cvr, relationtype, equity)

                        
                        if function['type'] == 'FUNKTION':
                            #call person function parser
                            
                            #loop through roles of participant
                            for value in function['vaerdier']:
                                
                                #relation type
                                relationtype = value['vaerdi']
                                
                                #equity
                                equity = None
                                
                                #do not record auditing relationships
                                if relationtype in ['REVISION','Reel ejer']:
                                    continue 
                                else:
                                    
                                    #call participant parser for non-owners
                                    lines_hit += self.add_participant(entity, value, cvr, relationtype, equity)
                                    
                        else:
                            continue
        
        return lines_hit