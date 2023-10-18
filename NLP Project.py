# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:15:23 2023

@author: ziedk
"""
import pandas as pd 
import re 
import numpy as np


import os
os.chdir('C:/Users/ziedk/OneDrive/Bureau/Strasbourg/Master data Strasbourg/Neural Network')


df = pd.read_csv("clean_data_tweet_sample.csv")

df['text'][2]


df['text'][3]
x = re.findall(r'#\w+',df['text'][3])
x

x = re.findall(r'https.*',df['text'][2])






def extract_tags(text:str)->dict(): 
    d = {}
    try:
        tags = re.findall(r'#\w+',text)
        d['hash_tags'] = tags
    except: 
        d['hash_tags'] = []
    try:
        d['links'] = re.findall(r'(https://t.co/\w+|http://t.co/\w+)',text)
        for link in d['links']:
            text = text.replace(link,'')
    except:
        d['links'] = []
    
    try:
        for tag in tags:
            text = text.replace(tag,'')
        text = text.strip()
        d['text'] = text
    except:
        d['text'] = text
    
    return d
    

extract_tags(df['text'][3])




def clean_text_col(df):
    text = list()
    links = list()
    tags = list()
    d_clean = dict()
    for r in range(0,len(df)):
        text.append(extract_tags(df['text'][r])['text'])
        tags.append(extract_tags(df['text'][r])['hash_tags'])
        links.append(extract_tags(df['text'][r])['links'])
    
    df['text_clean'] = text
    df['links'] = links
    df['hash_tags'] = tags        
    return df 

df_new = clean_text_col(df)

s = df_new[['text_clean','links','hash_tags']]