# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:54:30 2023

@author: ziedk
"""

# =============================================================================
# Sentiment Analysis
# =============================================================================

from transformers import pipeline

classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

sentences = ["I am not having a great day"]

model_outputs = classifier(sentences)

model_outputs

print(model_outputs[0])
# produces a list of dicts for each of the labels






x = [{'label': 'disappointment', 'score': 0.46669575572013855}, {'label': 'sadness', 'score': 0.3984946012496948}, {'label': 'annoyance', 'score': 0.0680660828948021}, {'label': 'neutral', 'score': 0.05703022703528404}, {'label': 'disapproval', 'score': 0.044239409267902374}, {'label': 'nervousness', 'score': 0.014850739389657974}, {'label': 'realization', 'score': 0.014059904962778091}, {'label': 'approval', 'score': 0.011267448775470257}, {'label': 'joy', 'score': 0.0063033816404640675}, {'label': 'remorse', 'score': 0.006221492309123278}, {'label': 'caring', 'score': 0.0060293967835605145}, {'label': 'embarrassment', 'score': 0.0052654929459095}, {'label': 'anger', 'score': 0.00498143769800663}, {'label': 'disgust', 'score': 0.004259039647877216}, {'label': 'grief', 'score': 0.004002132453024387}, {'label': 'confusion', 'score': 0.003382926108315587}, {'label': 'relief', 'score': 0.003140496090054512}, {'label': 'desire', 'score': 0.0028274671640247107}, {'label': 'admiration', 'score': 0.002815789543092251}, {'label': 'fear', 'score': 0.0027075267862528563}, {'label': 'optimism', 'score': 0.0026164911687374115}, {'label': 'love', 'score': 0.002488391939550638}, {'label': 'excitement', 'score': 0.0024494787212461233}, {'label': 'curiosity', 'score': 0.002374362898990512}, {'label': 'amusement', 'score': 0.001746692811138928}, {'label': 'surprise', 'score': 0.001452987315133214}, {'label': 'gratitude', 'score': 0.0006464761681854725}, {'label': 'pride', 'score': 0.0005542492726817727}]

def find_sentiment(tweet):
    score = 0
    sentiment =''
    model_outputs = classifier(tweet)
    #print(model_outputs)
    for i in model_outputs[0]:
        if (i['score'] > score):
            score = i['score']
            sentiment = i['label']
    return {'label':sentiment,'score':score}
        



    
def extract_sentiment_df(df):
    sentiment = list()
    for i in range(0,len(df)):
        print(sentiment)
        sentiment.append(find_sentiment(df['text'][i]))
    df['sentiment'] = sentiment
    return df 


df_sentiment = extract_sentiment_df(df)


df_sentiment
        


csv_file_path = 'sentiment_df.csv'
df.to_csv(csv_file_path, index=False)
