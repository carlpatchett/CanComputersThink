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

# Requires the dataset from https://ai.stanford.edu/~amaas/data/sentiment/
# This should be changed to the local directories of the training dataset.
data_loc = 'C:\\Users\\carlp\Desktop\\aclImdb\\'
data_loc_pos = data_loc + 'train\\pos\\'
data_loc_neg = data_loc + 'train\\neg\\'

clf_file_path = dir_path + '\\intermediate_sentiment_analysis_clf.joblib'
vect_file_path = dir_path + '\\intermediate_sentiment_analysis_vect.joblib'
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
    print('Collecting dataset into readable format...')

    counter = 0
    positive_train = []
    for files in os.walk(data_loc_pos):
            for name in files[2]:
                if counter < 5000: 
                    x = name.split('_')
                    label = 1
                    f = open(data_loc_pos + name, encoding="utf8")
                    sentences = f.read()
                    positive_train.append([sentences, label])
                    counter += 1

    counter = 0
    negative_train = []
    for files in os.walk(data_loc_neg):
            for name in files[2]:
                if counter < 5000:
                    x = name.split('_')
                    label = 0
                    f = open(data_loc_neg + name, encoding="utf8")
                    sentences = f.read()
                    negative_train.append([sentences, label])
                    counter += 1

    df_list = positive_train + negative_train

    df = pd.DataFrame(df_list, columns=['sentence', 'label'])
    sentences = df['sentence'].values
    y = df['label'].values

    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=0.25, random_state=1000
    )

    print('Training Model...')
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

    GetUserInput(vectorizer, clf)

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
        else:
            print('Positive Sentiment Detected :)')

CheckModelExistence()