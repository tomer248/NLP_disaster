# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 12:03:49 2020

@author: tomer
"""

# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')

dataset_test = pd.read_csv('test.csv')

# Cleaning the train texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 7613):
  review = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)

# Cleaning the test texts
corpus_test = []
for i in range(0, 3263):
  review = re.sub('[^a-zA-Z]', ' ', dataset_test['text'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus_test.append(review)


# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1700)
X_train = cv.fit_transform(corpus).toarray()
y_train = dataset.iloc[:, -1].values
X_test = cv.transform(corpus_test).toarray()


#creating a valid trining set
#Taking care of missing data training
dataset_key = dataset['keyword']
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'constant', verbose = 0)
missingvalues = missingvalues.fit([dataset_key])
[dataset_key]=missingvalues.transform([dataset_key])
#taking care of miising data test
dataset_key_test = dataset_test['keyword']
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'constant', verbose = 0)
missingvalues = missingvalues.fit([dataset_key_test])
[dataset_key_test]=missingvalues.transform([dataset_key_test])

#Encoding categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#for training and test set
labelencoder_X = LabelEncoder()
dataset_key= labelencoder_X.fit_transform(dataset_key)
dataset_key_test = labelencoder_X.transform(dataset_key_test)

onehotencoder = OneHotEncoder()
dataset_key=dataset_key.reshape(-1,1)
dataset_key_test = dataset_key_test.reshape(-1,1)

dataset_key = onehotencoder.fit_transform(dataset_key).toarray()
dataset_key_test = onehotencoder.transform(dataset_key_test).toarray()
dataset_key = dataset_key[:, 0:146]
dataset_key_test = dataset_key_test[:, 0:146]

#adding the keywords to the bag of word model
X_train = np.concatenate((dataset_key,X_train),axis = 1)
X_test = np.concatenate((dataset_key_test,X_test),axis = 1)
# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
dataset_test_id = dataset_test['id']
data = {'id': dataset_test_id, "target":y_pred}
results = pd.DataFrame(data)
results.to_csv("results_1600_r.csv",index=False)

#with xgboost
from xgboost import XGBClassifier
classifier_b = XGBClassifier()
classifier_b.fit(X_train, y_train)
y_pred_b = classifier_b.predict(X_test)
data = {'id': dataset_test_id, "target":y_pred_b}
results_b = pd.DataFrame(data)
results_b.to_csv("results_1700_b.csv",index=False)