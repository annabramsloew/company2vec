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

from CVR_script import connectToCvr

######################### SETUP #########################
cvr_input = '29140774'

cvr = cvr_input
es = connectToCvr()
s = Search(using=es, index="virksomhed") \
    .source(['Vrvirksomhed']) \
    .filter("term", Vrvirksomhed__cvrNummer=cvr)
response = s.execute()
hit = response[0]

with open('/Users/annabramslow/Downloads/29140774.json') as f:
    hit = json.load(f)

######################### FUNCTIONS #########################


def add_participant(entity, value, cvr):
    
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
    
    return [cvr, entity_id, name, participanttype]
    

def participants(hit, cvr):
    
    company = hit['Vrvirksomhed']
    
    lines = []
    
    #loop through participants
    for entity in company['deltagerRelation']:
        
        for role in entity['organisationer']:
            
            for item in role['medlemsData']:
                
                for function in item['attributter']:
                    
                    if function['type'] == 'EJERANDEL_PROCENT':
                        
                        # call owner function parser
                        
                        #loop through roles of participant
                        for value in function['vaerdier']:
                            
                            #relation type
                            relationtype = "EJERANDEL"
                            
                            #equity
                            equity = value['vaerdi']
                            
                            entryline = [relationtype, 'entry', value['periode']['gyldigFra'], equity]
                            exitline = [relationtype, 'exit', value['periode']['gyldigTil'], equity]
                            
                            line = add_participant(entity, value, cvr) + entryline
                            
                            lines.append(line)
                            
                            if value['periode']['gyldigTil'] is not None:
                                line = add_participant(entity, value, cvr) + exitline 
                                lines.append(line)

                    
                    elif function['type'] == 'FUNKTION':
                        
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
                                
                                entryline = ['entry', value['periode']['gyldigFra'], equity]
                                exitline = ['exit', value['periode']['gyldigTil'], equity]
                                
                                line = add_participant(entity, value, cvr) + entryline
                                
                                lines.append(line)
                                
                                if value['periode']['gyldigTil'] is not None:
                                    line = add_participant(entity, value, cvr) + exitline 
                                    lines.append(line)
                    
                    else:
                        continue
    
    df = pd.DataFrame(lines,columns=['CVR','EntityID','Name', 'ParticipantType',
                                     'RelationType', 'Participation', 'Date', 'Equity Pct' ])
            
    return df