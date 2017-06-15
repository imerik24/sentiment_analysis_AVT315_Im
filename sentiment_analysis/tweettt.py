#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tweepy
import sys
import re
import os
import numpy as np
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import json, time, sys
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
#This is a basic listener that just prints received tweets to stdout.

class StdOutListener(StreamListener):
    ''' Handles data received from the stream. '''

    def __init__(self, api=None):
        super(StdOutListener, self).__init__()
        self.num_tweets = 0
        self.tweet = []

    def on_status(self, status):
        # Prints the text of the tweet
        record = {'Text': status.text, 'Created At': status.created_at}
        print (record)  # See Tweepy documentation to learn how to access other fields
        tweet=status.text
        #print(u''.join(tweet))
        self.num_tweets += 1
        self.tweet.append(tweet)
        #self.label.addItem(QListWidgetItem(u''.join(tweet)))
        if self.num_tweets < 2:
            return True
        else:
            return False

    def on_error(self, status_code):
        print('Got an error with status code: ' + str(status_code))
        return True  # To continue listening

    def on_timeout(self):
        print('Timeout...')
        return True  # To continue listening

consumer_key = 'gjcS8K636rhQJpNubbQtACpLM'
consumer_secret = 'CLjM3tO56McJKrQKOll39htLtU12nvN3dBpjnnjoup2YjUk3r2'
access_token = '870367290260369408-Tbb0UTcYmn1pcYPIqDJpBdf2PMpgYH5'
access_secret = 'WcdXNeBQ44lwFi702SnocR2jTS7x7908bgS5v77qMpJs6'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
# #
api = tweepy.API(auth)
#This handles Twitter authetification and the connection to Twitter Streaming API
l = StdOutListener()
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
#stream = Stream(auth, l)
#
# #This line filter Twitter Streams to capture data by the keywords: 'python', 'javascript', 'ruby'
#stream.filter(track=['#python'])



a=[]
a=l.tweet
data=np.array(a)
print(data)

f = open('predobrabot.txt', 'r', encoding='utf8', errors='ignore')
k = 0
data = []
for line in f.readlines():
    data.append(line)
data = np.array(data)
f.close()

print(data)

tokenizer = Tokenizer(num_words=6000)
tokenizer.fit_on_texts(data)
print(tokenizer)
sequences = tokenizer.texts_to_sequences(data)
# word_index = tokenizer.word_index
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=10)
print(data)