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

# NOTE:
##     Not updated to latest version, as imtermediate replaces this with better training
##     Please use that script intead.

nlp = spacy.load('en_core_web_sm')

dir_path = os.path.dirname(os.path.realpath(__file__))

# This should be changed to the local directories of the training dataset.
data_loc_dict = {'yelp':   'C:\\Users\\carlp\\Desktop\\sentiment labelled sentences\\sentiment labelled sentences\\yelp_labelled.txt',
                 'amazon': 'C:\\Users\\carlp\\Desktop\\sentiment labelled sentences\\sentiment labelled sentences\\amazon_cells_labelled.txt',
                 'imdb':   'C:\\Users\\carlp\Desktop\\sentiment labelled sentences\\sentiment labelled sentences\\imdb_labelled.txt'}

clf_file_path = dir_path + '\\simple_sentiment_analysis_clf.joblib'
vect_file_path = dir_path + '\\simple_sentiment_analysis_vect.joblib'

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
    df_list = []
    for source, filepath in data_loc_dict.items():
        df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
        df['source'] = source  # Add another column filled with the source name
        df_list.append(df)

    df = pd.concat(df_list)

    sentences = []
    y = []
    for source in df['source'].unique():
        df_source = df[df['source'] == source]
        sentences  = df_source['sentence'].values
        y = df_source['label'].values

        sentences_train, sentences_test, y_train, y_test = train_test_split(
            sentences, y, test_size=0.25, random_state=1000)

        vectorizer = CountVectorizer()
        vectorizer.fit(sentences_train)
        X_train = vectorizer.transform(sentences_train)
        X_test  = vectorizer.transform(sentences_test)

        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print('Accuracy for {} data: {:.4f}'.format(source, score))

        dump(clf, clf_file_path)
        dump(vectorizer, vect_file_path)

        clf = load(clf_file_path)
        vectorizer = load(vect_file_path)

    GetUserInput(vectorizer, clf)

def GetUserInput(vectorizer, clf):

    while True:
        doc = nlp(input())

        X_test  = vectorizer.transform([doc.text])
        
        print('##########')
        if (clf.predict(X_test[0:1]) == 0):
            print('Negative Sentiment Detected :(')
        else:
            print('Positive Sentiment Detected :)')
        print('##########')

CheckModelExistence()