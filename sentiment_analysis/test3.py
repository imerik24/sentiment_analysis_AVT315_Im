#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from numpy import vstack, row_stack, asarray
from sklearn.cross_validation import train_test_split
from pandas import read_csv
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json
# Устанавливаем seed для повторяемости результатов
np.random.seed(42)
# Максимальное количество слов (по частоте использования)
max_features = 5000
# Максимальная длина рецензии в словах
maxlen = 80

# Загружаем данные
def get_sequences(data, values):
    return train_test_split(data, values, train_size=0.8, test_size=0.2)
def load_twitter_msgs():
    names = ['id', 'tdate', 'tname', 'ttext', 'ttype', 'trep',
             'tfav', 'tstcount', 'tfol', 'tfrien', 'listcount', 'basename']
    data_negative = read_csv('./twitter/negative.csv', delimiter=';', names=names)
    data_positive = read_csv('./twitter/positive.csv', delimiter=';', names=names)
    data = row_stack((data_negative[['ttext']], data_positive[['ttext']])).ravel()
    values = row_stack((data_negative[['ttype']], data_positive[['ttype']])).ravel()
    values = [v == 1 for v in values]
    data = asarray(data)
    return data, values


dat, values = load_twitter_msgs()

f=open('predobrabot.txt','r',encoding='utf8')
k=0
data=[]
for line in f.readlines():
    data.append(line)
data=np.array(data)
print(data)
f.close()

tokenizer = Tokenizer(num_words=6000)
tokenizer.fit_on_texts(data)
print(tokenizer)
sequences = tokenizer.texts_to_sequences(data)
# word_index = tokenizer.word_index
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=10)
print(data)

x_train, x_test, y_train, y_test = get_sequences(data, values)
y_train = asarray(y_train, dtype=bool)
y_test = asarray(y_test, dtype=bool)

print(x_test)
print(x_test[1])
# Создаем сеть
print(y_test.size)
json_file = open("test6.json","r")
loaded_model_json=json_file.read()
json_file.close()

loaded_model=model_from_json(loaded_model_json)
loaded_model.load_weights("test6.h5")
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
scores = loaded_model.evaluate(x_test, y_test, verbose=0)
print("%.2f%%" %(scores[1]*100))
print(x_test[105])
print(y_test[105])
print(loaded_model.predict_classes(x_test[105]))