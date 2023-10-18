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
from googletrans import Translator

df = pd.read_csv('data_tweet_sample_challenge.csv')  
  
# =============================================================================
# Translate the tweets in English
# =============================================================================
tr = Translator()

for i, row in df.iterrows():
    if row["label"] != "en":
        translated_text = tr.translate(row["text"], dest='en').text
        df.at[i, "text"] = translated_text


# =============================================================================
# Cleaning the data
# =============================================================================
stop = stopwords.words('english')
stop.append('rt')

def cleaning(text):
    text = text.lower()
    text = re.sub("\[.*?\]","",text)
    text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
    text = " ".join([word for word in text.split() if word not in (stop)])
    return text

df['clean_text'] = df['text'].apply(lambda x: cleaning(x))

df = df.drop_duplicates('clean_text')
df.reset_index(inplace=True, drop= True)






