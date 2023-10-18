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
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

# Importing the dataframe
df = pd.read_csv('clean_data_tweet_sample.csv')  
  

# =============================================================================
# Extracting the hashtags and links
# =============================================================================
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
    
    df['links'] = links
    df['hash_tags'] = tags        
    return df 

df = clean_text_col(df)


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
selected_columns = ['id', 'created_at', 'text', 'author.name', 'label', 'public_metrics.like_count', 'public_metrics.retweet_count', 'links', 'hash_tags']
df = df.loc[:, selected_columns]



# =============================================================================
# Data visualisation
# =============================================================================

# Create a new column with the date in format: YYYY-MM
df['date_M'] = df['created_at'].apply(lambda x: x[:7])

fig = px.line(df.groupby('date_M').count(), 
              y=df.groupby('date_M').count()['author.name'].values, 
              x=df.groupby('date_M').count().index)
fig.update_layout(plot_bgcolor = "white")
fig.update_layout(
    title="Number of Tweets per months",
    xaxis_title="",
    yaxis_title="Number of Tweets",
    font=dict(
         size=12,
        color="Black"
    ),
    plot_bgcolor = "white"
)
fig.show()

# Plot of the number of tweets per month
fig = make_subplots(rows=2, cols=3)

fig.add_trace(
    go.Line(x=df.loc[df['label'] == 'en'].groupby('date_M').count().index, 
            y=df.loc[df['label'] == 'en'].groupby('date_M').count()['author.name'].values,
            name='England'),
    row=1, col=1
)

fig.add_trace(
    go.Line(x=df.loc[df['label'] == 'es'].groupby('date_M').count().index, 
            y=df.loc[df['label'] == 'es'].groupby('date_M').count()['author.name'].values,
            name='Spain'),
    row=1, col=2
)

fig.add_trace(
    go.Line(x=df.loc[df['label'] == 'de'].groupby('date_M').count().index, 
            y=df.loc[df['label'] == 'de'].groupby('date_M').count()['author.name'].values,
            name='Germany'),
    row=1, col=3
)

fig.add_trace(
    go.Line(x=df.loc[df['label'] == 'nl'].groupby('date_M').count().index, 
            y=df.loc[df['label'] == 'nl'].groupby('date_M').count()['author.name'].values,
            name='Netherlands'),
    row=2, col=1
)

fig.add_trace(
    go.Line(x=df.loc[df['label'] == 'fr'].groupby('date_M').count().index, 
            y=df.loc[df['label'] == 'fr'].groupby('date_M').count()['author.name'].values,
            name='France'),
    row=2, col=2
)

fig.add_trace(
    go.Line(x=df.loc[df['label'] == 'it'].groupby('date_M').count().index, 
            y=df.loc[df['label'] == 'it'].groupby('date_M').count()['author.name'].values,
            name='Italy'),
    row=2, col=3
)

fig.update_layout(height=600, width=800, title_text="Number of Tweets per months")
fig.show()


# Number of tweets per author
fig = px.bar(df.groupby('author.name').count().sort_values(by='id', ascending=False).head(20), 
              y=df.groupby('author.name').count().sort_values(by='id', ascending=False).head(20)['id'].values, 
              x=df.groupby('author.name').count().sort_values(by='id', ascending=False).head(20).index)
fig.update_layout(plot_bgcolor = "white")
fig.update_layout(
    title="Number of Tweets per author (top 20)",
    xaxis_title="",
    yaxis_title="Number of Tweets",
    font=dict(
         size=12,
        color="Black"
    ),
    plot_bgcolor = "white"
)
fig.show()

# Top 5 per country
fig = make_subplots(rows=2, cols=3)

fig.add_trace(
    go.Bar(x=df.loc[df['label'] == 'en'].groupby('author.name').count().sort_values(by='id', ascending=False).head(5).index, 
            y=df.loc[df['label'] == 'en'].groupby('author.name').count().sort_values(by='id', ascending=False).head(5)['id'].values,
            name='England'),
    row=1, col=1
)

fig.add_trace(
    go.Bar(x=df.loc[df['label'] == 'es'].groupby('author.name').count().sort_values(by='id', ascending=False).head(5).index, 
            y=df.loc[df['label'] == 'es'].groupby('author.name').count().sort_values(by='id', ascending=False).head(5)['id'].values,
            name='Spain'),
    row=1, col=2
)

fig.add_trace(
    go.Bar(x=df.loc[df['label'] == 'de'].groupby('author.name').count().sort_values(by='id', ascending=False).head(5).index, 
            y=df.loc[df['label'] == 'de'].groupby('author.name').count().sort_values(by='id', ascending=False).head(5)['id'].values,
            name='Germany'),
    row=1, col=3
)

fig.add_trace(
    go.Bar(x=df.loc[df['label'] == 'nl'].groupby('author.name').count().sort_values(by='id', ascending=False).head(5).index, 
            y=df.loc[df['label'] == 'nl'].groupby('author.name').count().sort_values(by='id', ascending=False).head(5)['id'].values,
            name='Netherlands'),
    row=2, col=1
)

fig.add_trace(
    go.Bar(x=df.loc[df['label'] == 'fr'].groupby('author.name').count().sort_values(by='id', ascending=False).head(5).index, 
            y=df.loc[df['label'] == 'fr'].groupby('author.name').count().sort_values(by='id', ascending=False).head(5)['id'].values,
            name='France'),
    row=2, col=2
)

fig.add_trace(
    go.Bar(x=df.loc[df['label'] == 'it'].groupby('author.name').count().sort_values(by='id', ascending=False).head(5).index, 
            y=df.loc[df['label'] == 'it'].groupby('author.name').count().sort_values(by='id', ascending=False).head(5)['id'].values,
            name='Italy'),
    row=2, col=3
)

fig.update_layout(height=600, width=800, title_text="Number of Tweets per author (top 5)")
fig.show()


# Number of likes per author
fig = px.bar(df.groupby('author.name')['public_metrics.like_count'].sum().sort_values(ascending=False).head(20), 
              y=df.groupby('author.name')['public_metrics.like_count'].sum().sort_values(ascending=False).head(20).values, 
              x=df.groupby('author.name')['public_metrics.like_count'].sum().sort_values(ascending=False).head(20).index)
fig.update_layout(plot_bgcolor = "white")
fig.update_layout(
    title="Total number of likes per author (top 20)",
    xaxis_title="",
    yaxis_title="Total number of likes",
    font=dict(
         size=12,
        color="Black"
    ),
    plot_bgcolor = "white"
)
fig.show()

# Top 5 per country
fig = make_subplots(rows=2, cols=3)

fig.add_trace(
    go.Bar(x=df.loc[df['label'] == 'en'].groupby('author.name')['public_metrics.like_count'].sum().sort_values(ascending=False).head(5).index, 
            y=df.loc[df['label'] == 'en'].groupby('author.name')['public_metrics.like_count'].sum().sort_values(ascending=False).head(5).values,
            name='England'),
    row=1, col=1
)

fig.add_trace(
    go.Bar(x=df.loc[df['label'] == 'es'].groupby('author.name')['public_metrics.like_count'].sum().sort_values(ascending=False).head(5).index, 
            y=df.loc[df['label'] == 'es'].groupby('author.name')['public_metrics.like_count'].sum().sort_values(ascending=False).head(5).values,
            name='Spain'),
    row=1, col=2
)

fig.add_trace(
    go.Bar(x=df.loc[df['label'] == 'de'].groupby('author.name')['public_metrics.like_count'].sum().sort_values(ascending=False).head(5).index, 
            y=df.loc[df['label'] == 'de'].groupby('author.name')['public_metrics.like_count'].sum().sort_values(ascending=False).head(5).values,
            name='Germany'),
    row=1, col=3
)

fig.add_trace(
    go.Bar(x=df.loc[df['label'] == 'nl'].groupby('author.name')['public_metrics.like_count'].sum().sort_values(ascending=False).head(5).index, 
            y=df.loc[df['label'] == 'nl'].groupby('author.name')['public_metrics.like_count'].sum().sort_values(ascending=False).head(5).values,
            name='Netherlands'),
    row=2, col=1
)

fig.add_trace(
    go.Bar(x=df.loc[df['label'] == 'fr'].groupby('author.name')['public_metrics.like_count'].sum().sort_values(ascending=False).head(5).index, 
            y=df.loc[df['label'] == 'fr'].groupby('author.name')['public_metrics.like_count'].sum().sort_values(ascending=False).head(5).values,
            name='France'),
    row=2, col=2
)

fig.add_trace(
    go.Bar(x=df.loc[df['label'] == 'it'].groupby('author.name')['public_metrics.like_count'].sum().sort_values(ascending=False).head(5).index, 
            y=df.loc[df['label'] == 'it'].groupby('author.name')['public_metrics.like_count'].sum().sort_values(ascending=False).head(5).values,
            name='Italy'),
    row=2, col=3
)

fig.update_layout(height=600, width=800, title_text="Total number of likes per author (top 5)")
fig.show()

# Total retweets per author
fig = px.bar(df.groupby('author.name')['public_metrics.retweet_count'].sum().sort_values(ascending=False).head(20), 
              y=df.groupby('author.name')['public_metrics.retweet_count'].sum().sort_values(ascending=False).head(20).values, 
              x=df.groupby('author.name')['public_metrics.retweet_count'].sum().sort_values(ascending=False).head(20).index)
fig.update_layout(plot_bgcolor = "white")
fig.update_layout(
    title="Total number of retweets per author (top 20)",
    xaxis_title="",
    yaxis_title="Total number of retweets",
    font=dict(
         size=12,
        color="Black"
    ),
    plot_bgcolor = "white"
)
fig.show()

# Top 5 per country
fig = make_subplots(rows=2, cols=3)

fig.add_trace(
    go.Bar(x=df.loc[df['label'] == 'en'].groupby('author.name')['public_metrics.retweet_count'].sum().sort_values(ascending=False).head(5).index, 
            y=df.loc[df['label'] == 'en'].groupby('author.name')['public_metrics.retweet_count'].sum().sort_values(ascending=False).head(5).values,
            name='England'),
    row=1, col=1
)

fig.add_trace(
    go.Bar(x=df.loc[df['label'] == 'es'].groupby('author.name')['public_metrics.retweet_count'].sum().sort_values(ascending=False).head(5).index, 
            y=df.loc[df['label'] == 'es'].groupby('author.name')['public_metrics.retweet_count'].sum().sort_values(ascending=False).head(5).values,
            name='Spain'),
    row=1, col=2
)

fig.add_trace(
    go.Bar(x=df.loc[df['label'] == 'de'].groupby('author.name')['public_metrics.retweet_count'].sum().sort_values(ascending=False).head(5).index, 
            y=df.loc[df['label'] == 'de'].groupby('author.name')['public_metrics.retweet_count'].sum().sort_values(ascending=False).head(5).values,
            name='Germany'),
    row=1, col=3
)

fig.add_trace(
    go.Bar(x=df.loc[df['label'] == 'nl'].groupby('author.name')['public_metrics.retweet_count'].sum().sort_values(ascending=False).head(5).index, 
            y=df.loc[df['label'] == 'nl'].groupby('author.name')['public_metrics.retweet_count'].sum().sort_values(ascending=False).head(5).values,
            name='Netherlands'),
    row=2, col=1
)

fig.add_trace(
    go.Bar(x=df.loc[df['label'] == 'fr'].groupby('author.name')['public_metrics.retweet_count'].sum().sort_values(ascending=False).head(5).index, 
            y=df.loc[df['label'] == 'fr'].groupby('author.name')['public_metrics.retweet_count'].sum().sort_values(ascending=False).head(5).values,
            name='France'),
    row=2, col=2
)

fig.add_trace(
    go.Bar(x=df.loc[df['label'] == 'it'].groupby('author.name')['public_metrics.retweet_count'].sum().sort_values(ascending=False).head(5).index, 
            y=df.loc[df['label'] == 'it'].groupby('author.name')['public_metrics.retweet_count'].sum().sort_values(ascending=False).head(5).values,
            name='Italy'),
    row=2, col=3
)

fig.update_layout(height=600, width=800, title_text="Total number of retweets per author (top 5)")
fig.show()