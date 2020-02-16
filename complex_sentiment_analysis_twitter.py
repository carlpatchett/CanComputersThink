import spacy
import pandas as pd
import pickle
import string
import os
import os.path

from spacy import displacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import datasets
from joblib import dump, load
from os import path

nlp = spacy.load('en_core_web_sm')

dir_path = os.path.dirname(os.path.realpath(__file__))

# This should be changed to the local file path of the training dataset.
data_loc = 'C:\\Users\\carlp\Desktop\\trainingandtestdata\\training.1600000.processed.noemoticon.csv'

clf_file_path = dir_path + '\\complex_sentiment_analysis_twitter_clf.joblib'
vect_file_path = dir_path + '\\complex_sentiment_analysis_twitter_vect.joblib'
clf = None
vectorizer = None

def CheckModelExistence():
    if (path.exists(clf_file_path)):
        print('Model exists! Checking vectorizer...')

        if (path.exists(vect_file_path)):
            print('Vectorizer exists! Loading...')

            global clf
            global vectorizer

            clf = load(clf_file_path)
            vectorizer = load(vect_file_path)

            GetUserInput(vectorizer, clf)
        else:
            print('Vectorizer doesn\'t exist. Creating new model...')
            CreateModel(vectorizer, clf)
    else:
        print('Model doesn\'t exist. Creating new model...')
        CreateModel(vectorizer, clf)

def CreateModel(vectorizer, clf):
    print('Creating DataFrame from CSV...')
    data = pd.read_csv(data_loc, names=['target','id','date','flag','user','text'],  encoding="ISO-8859-1")

    sentences = data['text'].values
    y = data['target'].values

    print('Training model...')

    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=0.05, random_state=1000
    )

    vectorizer = CountVectorizer()
    vectorizer.fit(sentences_train)
    X_train = vectorizer.transform(sentences_train)
    X_test  = vectorizer.transform(sentences_test)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print('Accuracy for {} data: {:.4f}'.format('Test', score))

    dump(clf, clf_file_path)
    dump(vectorizer, vect_file_path)

    clf = load(clf_file_path)
    vectorizer = load(vect_file_path)

    GetUserInput()

def GetUserInput(vectorizer, clf):
    while True:
        print('##########')
        print('Awaiting User Input for Testing...')
        print('##########')

        doc = nlp(input())
        X_test  = vectorizer.transform([doc.text])
        predicted_score = clf.predict(X_test)

        print('Predicated sentiment score: ' + str(predicted_score))
        print('##########')

        if (predicted_score == 0):
            print('Negative Sentiment Detected :(')
        elif (predicted_score == 2):
            print('Neutral Sentiment Detected :|')
        else:
            print('Positive Sentiment Detected :)')

CheckModelExistence()