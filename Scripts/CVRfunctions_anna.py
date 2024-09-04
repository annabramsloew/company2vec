#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 13:08:34 2024

@author: annabramslow
"""
import pandas as pd
import re
import urllib.request as request
import requests
from requests.auth import HTTPBasicAuth
import json 
import contextlib
from datetime import datetime
from elasticsearch import Elasticsearch , RequestsHttpConnection
from elasticsearch_dsl import Search, Q
from datetime import datetime
import json

######################### SETUP #########################
cvr = '35657339'

# with open('/Users/annabramslow/Downloads/29140774.json') as f:
#     hit = json.load(f)
    
with open('/Users/annabramslow/Downloads/virksomheder.json') as f:
    virksomheder = json.load(f)
    hit = virksomheder[0]['_source']

######################### FUNCTIONS #########################


def add_participant(lines, entity, value, cvr, relationtype, equity):
    
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
    lines.append([cvr, entity_id, name, participanttype,
                      relationtype,'entry', startdate, equity])

    if enddate is not None:
        lines.append([cvr, entity_id, name, participanttype,
                      relationtype, 'exit', enddate, equity])
    

def participants(hit, cvr):
    
    company = hit['Vrvirksomhed']
    
    lines = []
    
    #loop through participants
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
                            equity = value['vaerdi']
                            
                            #call participant parser for owners
                            add_participant(lines, entity, value, cvr, relationtype, equity)

                    
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
                                add_participant(lines, entity, value, cvr, relationtype, equity)
                                
                    else:
                        continue
    
    df = pd.DataFrame(lines,columns=['CVR','EntityID','Name', 'ParticipantType',
                                     'RelationType', 'Participation', 'Date', 'EquityPct' ])
            
    return df