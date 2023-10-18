#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:35:08 2023

@author: lyna
"""

import pandas as pd
import nltk.corpus 
nltk.download('stopwords')
from googletrans import Translator # pip install googletrans==3.1.0a0
from tqdm import tqdm

# Importing the dataframe
df = pd.read_csv('data_tweet_sample_challenge.csv')  
  
# =============================================================================
# Translate the tweets in English
# =============================================================================
tr = Translator()

# Creating a progression bar
progress_bar = tqdm(total=len(df), desc="Translation Progress")

for i, row in df.iterrows():
    if row["label"] != "en":
        translated_text = tr.translate(row["text"], dest='en').text
        df.at[i, "text"] = translated_text

    # Updating the progression bar
    progress_bar.update(1)

progress_bar.close()

# Saving in a csv file
csv_file_path = 'clean_data_tweet_sample.csv'
df.to_csv(csv_file_path, index=False)