#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from numpy import vstack, row_stack, asarray
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import  Dense, Activation, Embedding
from keras.layers import LSTM, SpatialDropout1D
from keras.datasets import imdb
from pandas import read_csv
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Dropout
from collections import Counter
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import model_from_json
from keras.utils import plot_model

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
f=open('prepdata.txt','r',encoding='utf8')
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
print(sequences)
# word_index = tokenizer.word_index
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=15)
print(data)

x_train, x_test, y_train, y_test = get_sequences(data, values)
y_train = asarray(y_train, dtype=bool)
y_test = asarray(y_test, dtype=bool)

top_words=6000
max_features=6000
# create the model
embedding_vecor_length = 32
# model = Sequential()
# # # # Слой для векторного представления слов
# model.add(Embedding(max_features, 10))
# model.add(SpatialDropout1D(0.2))
# # # # Слой долго-краткосрочной памяти
# model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
# # # # Полносвязный слой
# model.add(Dense(1, activation="sigmoid"))
# #
# # # Копмилируем модель

# model = Sequential()
# model.add(Embedding(top_words, 32))
# model.add(Dropout(0.2))
# model.add(LSTM(100))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

model = Sequential()
model.add(Embedding(top_words, 32))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])
print(model.summary())
plot_model(model, to_file='model3.png', show_shapes=True)
# # Обучаем модель
model.fit(x_train, y_train, batch_size=64, epochs=10,
          validation_data=(x_test, y_test), verbose=2)
# # Проверяем качество обучения на тестовых данных
scores = model.evaluate(x_test, y_test,
                        batch_size=64)
print("Точность на тестовых данных: %.2f%%" % (scores[1] * 100))
print("Сохраняем сеть")

#
# print("Сохраняем сеть")
# # Сохраняем сеть для последующего использования
# # Генерируем описание модели в формате json
# model_json = model.to_json()
# json_file = open("t1.json", "w")
# # Записываем архитектуру сети в файл
# json_file.write(model_json)
# json_file.close()
# # Записываем данные о весах в файл
# model.save_weights("t1.h5")
# print("Сохранение сети завершено")
# # print('Found %s unique tokens.' % len(word_index))