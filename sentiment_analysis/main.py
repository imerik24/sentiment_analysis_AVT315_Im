# -*- coding: utf-8 -*-
import tweepy
import sys
import re
from tweepy.streaming import StreamListener
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from keras.models import model_from_json
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from pymystem3 import Mystem
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


consumer_key = 'gjcS8K636rhQJpNubbQtACpLM'
consumer_secret = 'CLjM3tO56McJKrQKOll39htLtU12nvN3dBpjnnjoup2YjUk3r2'
access_token = '870367290260369408-Tbb0UTcYmn1pcYPIqDJpBdf2PMpgYH5'
access_secret = 'WcdXNeBQ44lwFi702SnocR2jTS7x7908bgS5v77qMpJs6'

# # auth = OAuthHandler(consumer_key, consumer_secret)
# # auth.set_access_token(access_token, access_secret)
# #
# # api = tweepy.API(auth)
# #This handles Twitter authetification and the connection to Twitter Streaming API
# l = StdOutListener()
# auth = OAuthHandler(consumer_key, consumer_secret)
# auth.set_access_token(access_token, access_secret)
# stream = Stream(auth, l)
#
# #This line filter Twitter Streams to capture data by the keywords: 'python', 'javascript', 'ruby'
# stream.filter(track=['#love'])

class StdOutListener(StreamListener):
    ''' Handles data received from the stream. '''



    def on_status(self, status):
        # Prints the text of the tweet
        app = QApplication(sys.argv)
        ex = App()
        sys.exit(app.exec_())
        tweet=status.text
        print(u''.join(tweet))
        #self.label.addItem(QListWidgetItem(u''.join(tweet)))
        return True

    def on_error(self, status_code):
        print('Got an error with status code: ' + str(status_code))
        return True  # To continue listening

    def on_timeout(self):
        print('Timeout...')
        return True  # To continue listening

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Анализатор'
        self.left = 100
        self.top = 100
        self.width = 1800
        self.height = 600
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.button = QPushButton('Скачать твиты', self)
        self.button.setToolTip('This is an example button')
        self.button.move(100, 70)
        self.button.clicked.connect(self.on_click)

        self.edit = QLineEdit(self)
        self.edit.setText('#хэштег')
        self.edit.move(100,30)

        self.label = QListWidget(self)
        self.label.resize(800, 400)
        self.label.move(100, 120)

        self.label1 = QListWidget(self)
        self.label1.resize(800, 400)
        self.label1.move(960, 120)

        self.l = QLabel('отрицательные',self)
        self.l.move(100, 100)

        self.l1 = QLabel('положительные',self)
        self.l1.move(960, 100)

        self.show()

    @pyqtSlot()
    def on_click(self):

        self.label.clear()
        self.label1.clear()
        def preprocess(sentence):
            mystem = Mystem()
            sentence = ''.join(mystem.lemmatize(sentence))
            sentence = sentence.lower()
            tokenizer = RegexpTokenizer(r'\w+')
            sentence = re.sub(r'@\w+', ' ', sentence)
            sentence = re.sub(r'(?:https?:\/\/)?(?:[\w\.]+)\.(?:[a-z]{2,6}\.?)(?:\/[\w\.]*)*\/?', ' ', sentence)
            sentence = re.sub(r'r\w+', ' ', sentence)
            tokens = tokenizer.tokenize(sentence)
            filt_word = [word for word in tokens if word not in stopwords.words('russian')]
            #   contr=contr+Counter(filt_word)
            return " ".join(filt_word)

        print('PyQt5 button click')
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_secret)
        api = tweepy.API(auth)
        search_text = self.edit.text()
        search_number = 100
        search_result = api.search(search_text, count=20)
        query = 'python'
        max_tweets = 20
        #search_result = [status for status in tweepy.Cursor(api.search, q=search_text, since='2015-06-26 10:01:25').items(max_tweets)]
        #search_result = api.user_timeline(search_text, count=20)
        #for tweet in tweepy.Cursor(api.search, q='test', since='2015-06-26 10:01:25',
        #                          until='2015-06-26 10:15:10').items():
        print(search_result)
        json_file = open("test6.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("test6.h5")
        loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        f = open('predobrabot.txt', 'r', encoding='utf8', errors='ignore')
        k = 0
        data = []
        for line in f.readlines():
            data.append(line)
        data = np.array(data)
        f.close()
        tokenizer = Tokenizer(num_words=6000)
        tokenizer.fit_on_texts(data)
        a=0
        b=0
        for i in search_result:
            #print(i.text)
            sentence=preprocess(i.text)
            #print(sentence)
            sequences = tokenizer.texts_to_sequences([sentence])
            data=pad_sequences(sequences, maxlen=10)
            pred = loaded_model.predict_classes(data)
            #print(pred)
            p=loaded_model.predict_proba(data)

            if pred == 0:
                self.label.addItem(QListWidgetItem(np.array_str(p[0])))
                self.label.addItem(QListWidgetItem(i.text))
                a=a+1
            else:
                b=b+1
                self.label1.addItem(QListWidgetItem(np.array_str(p[0])))
                self.label1.addItem(QListWidgetItem(i.text))
        print(a,b)
        self.label.addItem(QListWidgetItem('Процент отрицательных:'+str(a*100/(a+b))))
        self.label1.addItem(QListWidgetItem('Процент положительных:'+str(b*100/(a + b))))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())