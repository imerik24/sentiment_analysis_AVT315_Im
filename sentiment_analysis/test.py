#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from sklearn import metrics
from numpy import vstack, row_stack, asarray
from nltk.stem.snowball import RussianStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import  Dense, Activation, Embedding
from keras.layers import LSTM, SpatialDropout1D
from keras.datasets import imdb
from pandas import read_csv
from pymystem3 import Mystem
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from collections import Counter
import re
def preprocess(sentence):
    mystem = Mystem()
    sentence=''.join(mystem.lemmatize(sentence))
    sentence=sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    sentence=re.sub(r'@\w+',' ', sentence)
    sentence = re.sub(r'(?:https?:\/\/)?(?:[\w\.]+)\.(?:[a-z]{2,6}\.?)(?:\/[\w\.]*)*\/?', ' ', sentence)
    sentence = re.sub(r'r\w+', ' ', sentence)
    tokens = tokenizer.tokenize(sentence)
    filt_word = [word for word in tokens if word not in stopwords.words('russian')]
    return " ".join(filt_word)

def dict(sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    filt_word = [word for word in tokens]
    cntr=Counter(filt_word)
    return cntr

def dict_toarray(a,sentence,cntr):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    filt_word = [cntr[word] for word in tokens]
    a.append(filt_word)
    return a

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

def get_bag_of_words(data, max_features):
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=max_features)
    bag_of_words = vectorizer.fit_transform(data)
    bag_of_words = bag_of_words.toarray()
    return bag_of_words

def get_sequences(data, values):
    return train_test_split(data, values, train_size=0.9, test_size=0.1)

def classification(train_x, train_y, method=BernoulliNB):
    clf = method()
    return clf.fit(train_x, train_y)

def test(clf, test_x, test_y):
    res = clf.predict(test_x)
    return metrics.accuracy_score(test_y, res)

def tokenize_data(X_raw):
    tokenizer = Tokenizer(nb_words=1000)
    tokenizer.fit_on_texts(X_raw)
    sequences = tokenizer.texts_to_sequences(X_raw)
    word_index = tokenizer.word_index
    X_processed = pad_sequences(sequences, maxlen=100)
    return X_processed, word_index

data, values = load_twitter_msgs()
print (data)
cntr=Counter()
k=0
a=[]
for i in data:
     i = preprocess(i)
     cntr=cntr+dict(i)
     data[k] = i
     k += 1
     print(k)
print(data)
print(cntr)

f = open('prepdata.txt', 'w', encoding='utf8')
for index in data:
   f.write(index + '\n')
f.close()

tokenizer = Tokenizer(num_words=6000)
tokenizer.fit_on_texts(data)
print(tokenizer)
sequences = tokenizer.texts_to_sequences(data)
# word_index = tokenizer.word_index
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=16)

x_train, x_test, y_train, y_test = get_sequences(data, values)
y_train = asarray(y_train, dtype=bool)
y_test = asarray(y_test, dtype=bool)
# for i in x_train:
#     a=dict_toarray(a,i,cntr)
# b=np.array((a))
# b=sequence.pad_sequences(b, maxlen=20)
# c=[]
# for i in x_test:
#     c=dict_toarray(c,i,cntr)
# d=np.array((c))
# d=sequence.pad_sequences(d, maxlen=20)
# x_train=b
# x_test=d
# y_train = asarray(y_train, dtype=bool)
# y_test = asarray(y_test, dtype=bool)

print('обучение')
max_features=6000
# # Создаем сеть
model = Sequential()
# # Слой для векторного представления слов
model.add(Embedding(max_features, 32))
model.add(SpatialDropout1D(0.2))
# # Слой долго-краткосрочной памяти
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
# # Полносвязный слой
model.add(Dense(1, activation="sigmoid"))
#
# # Копмилируем модель
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# # Обучаем модель
model.fit(x_train, y_train, batch_size=64, epochs=20,
          validation_data=(x_test, y_test), verbose=2)
# # Проверяем качество обучения на тестовых данных
scores = model.evaluate(x_test, y_test,
                        batch_size=64)
print("Точность на тестовых данных: %.2f%%" % (scores[1] * 100))
print("Сохраняем сеть")

# Сохраняем сеть для последующего использования
# Генерируем описание модели в формате json
model_json = model.to_json()
json_file = open("mnist_model.json", "w")
# Записываем архитектуру сети в файл
json_file.write(model_json)
json_file.close()
# Записываем данные о весах в файл
model.save_weights("mnist_model.h5")
print("Сохранение сети завершено")


