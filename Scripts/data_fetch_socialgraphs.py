#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 13:34:25 2023

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
import gender_guesser.detector as gender

from CVR_script import connectToCvr

#Retrieve all active companies (ApS, A/S, P/S) with employees of 100 and above

q = Q("match",Vrvirksomhed__virksomhedMetadata__sammensatStatus="Aktiv") | Q("match",Vrvirksomhed__virksomhedMetadata__sammensatStatus="Normal")
es = connectToCvr()
s = Search(using=es, index="virksomhed") \
    .filter(q) \
    .filter("range",Vrvirksomhed__virksomhedMetadata__nyesteAarsbeskaeftigelse__antalAarsvaerk={'gte': 100, 'lte':20000}) \
    .source(["Vrvirksomhed"])
        
response = s[0:3000].execute()

df = pd.DataFrame(columns=['CVR','Firma','Ansatte (Mindst angivet)','Mail','Tlf','Adresse','Postnummer','By'])

#substract cvr no
cvr_no = [] 
for hit in response:
    cvr = hit.Vrvirksomhed.cvrNummer       
    cvr_no.append(cvr)

#-----------------------------------------------------------------------------

def fetch_data(cvr_input):
    cvr = cvr_input
    es = connectToCvr()
    s = Search(using=es, index="virksomhed") \
        .source(['Vrvirksomhed']) \
        .filter("term", Vrvirksomhed__cvrNummer=cvr)
    response = s.execute()
    hit = response[0]
    company = hit.Vrvirksomhed
    
    #constant values
    cvrno = company.cvrNummer
    company_name = company.virksomhedMetadata.nyesteNavn.navn
    #latest industry values
    industry_code = company.hovedbranche[-1]['branchekode']
    industry_description = company.hovedbranche[-1]['branchetekst']
    #latest purpose
    attributes = company.attributter
    purpose = ''
    for i in range(len(attributes)):
        if attributes[i]['type']=='FORMÅL':
            purpose = attributes[i]['vaerdier'][-1]['vaerdi']
            
    #latest FTE numbers
    FTE = company.aarsbeskaeftigelse[-1]['antalAarsvaerk']
    FTE_category = company.aarsbeskaeftigelse[-1]['intervalKodeAntalAarsvaerk']
    
    
    names = []
    positions = []
    period_starts = []
    period_ends = []
    person_ids = []
    
    for person in company.deltagerRelation:
        #check for humans
        #if person['deltager']['enhedstype'] not in ['PERSON','ANDEN_DELTAGER']:
           # continue
        
        # ensure that we are looking at a boardmember
        organisation_idx = None
        for i in range(len(person.organisationer)):
            if person.organisationer[i]['hovedtype'] == 'LEDELSESORGAN':
                organisation_idx = i
                continue
        
        # skip if a board member type has not been found
        if organisation_idx == None:
            continue
        
        #find name, position, period
        #for attributes in person.organisationer[0]['medlemsData'][0]['attributter']:
        #    if attributes['type'] == 'FUNKTION':
        position = person.organisationer[organisation_idx]['medlemsData'][0]['attributter'][0]['vaerdier'][0]['vaerdi']
        period_start = person.organisationer[organisation_idx]['medlemsData'][0]['attributter'][0]['vaerdier'][0]['periode']['gyldigFra']
        period_end = person.organisationer[organisation_idx]['medlemsData'][0]['attributter'][0]['vaerdier'][0]['periode']['gyldigTil']
        name = person['deltager']['navne'][0]['navn']   
        person_id = person['deltager']['enhedsNummer']
    
        positions.append(position)
        names.append(name)
        period_starts.append(period_start)
        period_ends.append(period_end)
        person_ids.append(person_id)
    
    df = pd.DataFrame({'personid':person_ids,'name':names,'position':positions,
                       'period_start':period_starts,'period_end':period_ends,
                       'cvr':cvrno,'company_name':company_name,'purpose':purpose,
                       'industry_code':industry_code,'industry_desc':industry_description,
                       'FTE':FTE,'FTE_category':FTE_category})
    return df

# -----------------------------------------------------------------------------

def compute_gender(name, d):
    
    #based on name predict gender
    sex = d.get_gender(name)
    
    return sex


def assign_gender(df,d):
    
    #use compute_gender function to create sex column in input df
    df['sex'] = df['name'].apply(lambda x: compute_gender(x.split(' ')[0],d))
    
    return df

# -----------------------------------------------------------------------------

companys = pd.read_csv('cvr.csv')
companys = companys['cvr'].to_list()

#intialize empty df
df_final = pd.DataFrame()

#using gender guesser module
d = gender.Detector()

for company in companys:
    df = fetch_data(company)
    df = assign_gender(df,d)
    
    df.to_csv(r'/Users/annabramslow/Library/CloudStorage/Dropbox/DTU/Social Graphs and Interactions/Projekt/'+str(company)+'.csv')
    
    #append to df_final
    df_final = pd.concat([df_final, df], ignore_index=True)
    
    
