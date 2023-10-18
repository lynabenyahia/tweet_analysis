#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:03:29 2023

@author: lyna
"""

import pandas as pd
import re
import nltk.corpus 
nltk.download('stopwords')
from nltk.corpus import stopwords

# Importing the dataframe
df = pd.read_csv('clean_data_tweet_sample.csv')  
  
# =============================================================================
# Cleaning the data
# =============================================================================
stop = stopwords.words('english')
stop.append('rt')

def cleaning(text):
    text = text.lower() # putting the text in lowercase
    text = re.sub("\[.*?\]","",text) # Remove text enclosed in square brackets
    text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
    # remove @, non-alphanumeric chars, tabs, url, rt and http
    text = " ".join([word for word in text.split() if word not in (stop)]) # remove stopwords
    return text

df['text'] = df['text'].apply(lambda x: cleaning(x))

df = df.drop_duplicates('text') # drop the duplicates

# Selecting only the columns that we interested in
selected_columns = ['id', 'created_at', 'text', 'author.name', 'label', 'public_metrics.like_count', 'public_metrics.retweet_count']
df = df.loc[:, selected_columns]





