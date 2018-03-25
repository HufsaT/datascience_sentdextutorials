# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 22:25:40 2018

@author: Hufsa
"""
import tweepy
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
# We import our access keys:
from credentialsTwitter import *    # This will allow us to use the keys as variables
from textblob import TextBlob
import re

#Import Library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#assigning predictor and target variables
x= np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])
Y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])

# API's setup:
def twitter_setup():
    """
    Utility function to setup the Twitter's API
    with our access keys provided.
    """
    # Authentication and access using keys:
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

    # Return API with authentication:
    api = tweepy.API(auth)
    return api

# crate an extraction obj
    
extractor = twitter_setup()

# create tweet list from account
#tweets = extractor.user_timeline(screen_name="KoboWritingLife", count=40)
q = '"kobo writing life" OR #kobowritinglife'
tweets = extractor.search(q,rpp=100,show_user=True)

print('Number of tweets extracted:{} .\n'.format(len(tweets)))

# print top x tweets
#print("Top {} tweets:".format(5))

#[print(tweet.text) for tweet in tweets[:5]]

# put iot into a pd
data = pd.DataFrame(data=[tweet.text for tweet in tweets],columns=['Tweets'])


# test print metadata from a tweet
#print(tweets[0].id,tweets[0].favorite_count,tweets[0].entities)

data['Len'] = np.array([len(tweet.text) for tweet in tweets])
#data['ID'] = np.array([tweet.id for tweet in tweets])
data['Date'] = np.array([tweet.created_at for tweet in tweets])
data['Source'] = np.array([tweet.source for tweet in tweets])
data['Likes'] = np.array([tweet.favorite_count for tweet in tweets])
data['RTs'] = np.array([tweet.retweet_count for tweet in tweets])



# now we grab some basic stats like mean etc

avg_len = np.mean(data['Len'])

# highest faves count and rts:
max_faves = np.max(data['Likes'])
max_RTs = np.max(data['RTs'])

# get first value that appears (in case of duplicate fave numbers)
fave_tweet = data[data['Likes']==max_faves].index[0]
rt_tweet = data[data['RTs']==max_RTs].index[0]

print('Most Liked tweet is: {}'.format(data['Tweets'][fave_tweet]))
print('This had {} likes.'.format(max_faves))

print('Most RT\'d tweet is: {}'.format(data['Tweets'][rt_tweet]))
print('This had {} RTs.'.format(max_RTs))

def clean_tweet(tweet):
    '''
    Utility function to clean the text in a tweet by removing 
    links and special characters using regex.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


def analize_sentiment(tweet):
    '''
    Utility function to classify the polarity of a tweet
    using textblob.
    '''
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1

data['Sentiment'] = np.array([analize_sentiment(tweet) for tweet in data['Tweets']]) 

display(data.head(10))
