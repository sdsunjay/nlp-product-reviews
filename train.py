
import numpy as np
import pandas as pd
import torch
import transformers as ppb  # pytorch transformers
# Training the model and Testing Accuracy on Validation data
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import os.path
import sys, traceback
import random
import re
from nltk.corpus import stopwords
import nltk.classify.util
import nltk.metrics

from nltk.classify import NaiveBayesClassifier
from nltk.classify.scikitlearn import SklearnClassifier
import string

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier, SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier

import string

from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from common import save_model, load_model, train_classifier as trainClassifier

def something():
    y_final_pred = model.predict(testing_features)
    df1["human_tag"] = y_final_pred
    header = ["ID", "human_tag"]
    output_path = 'result/' + model_name
    print('Output: ' + output_path)
    df1.to_csv(output_path, columns = header)


def trainClassifiers(features, labels, testing_features, df1):
    print('Starting training')
    # create training and testing vars
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1)

    # Create a simple Logistic Regression classifier
    # model_name = 'Logistic Regression (solver=lbfgs)'
    # model = LogisticRegression(max_iter=200)
    # trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # Create a simple Logistic Regression classifier
    # model_name = 'Logistic Regression (penalty elasticnet)'
    # model = LogisticRegression(max_iter=200, solver='saga')
    # trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # Create a simple Linear SVC classifier
    # model_name = 'Linear SVC (max_iter=5000)'
    # model = LinearSVC(random_state=42, max_iter=5000)
    # trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # XGBoost
    model = XGBClassifier()
    model_name = 'XGB'
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # TODO - Fix this and make it work
    # Naive Bayes Classifier
    # model = NaiveBayesClassifier()
    # model_name = 'Naive Bayes Classifier'
    # trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # GaussianNB Classifier
    # model = GaussianNB()
    # model_name = 'Gaussian Classifier'
    # trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # TODO - Fix this and make it work
    # MultinomialNB Classifier
    # model = MultinomialNB()
    # model_name = 'Multinomial Classifier'
    # trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # Multi-Layer Perceptron Classifier (3 layers)
    model = MLPClassifier(hidden_layer_sizes=(30,30,30))
    model_name = 'MLP3'
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    model = MLPClassifier(hidden_layer_sizes=(50,50,50,50,50))
    # model_name = 'Multi-Layer Perceptron Classifier (5 layers)'
    model_name = 'MLP5'
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    model = MLPClassifier(hidden_layer_sizes=(20,20,20,20,20,20,20,20,20,20))
    model_name = 'MLP10'
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    model = MLPClassifier(hidden_layer_sizes=(100,100,100,100,100,100,100,100,100,100))
    model_name = 'MLP100'
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # Sparse Support Vector Classifier
    model = SklearnClassifier(SVC(),sparse=False)
    model_name = 'Sparse_SVC'
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # Linear Support Vector Classifier
    model_name = 'Linear_SVC'
    model = SklearnClassifier(SVC(kernel='linear', probability=True, tol=1e-3))
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # l1 Support Vector Classifier
    model_name = 'Linear_SVC_l1'
    model = SklearnClassifier(LinearSVC("l1", dual=False, tol=1e-3))
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # l2 Support Vector Classifier
    model_name = 'Linear_SVC_l2'
    model = SklearnClassifier(LinearSVC("l2", dual=False, tol=1e-3))
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # Train SGD with hinge penalty
    # model_name = "Stochastic Gradient Descent Classifier (hinge loss)"
    model_name = 'SGD_hinge_loss'
    model = SklearnClassifier(SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5000, tol=None))
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # Train SGD with Elastic Net penalty
    model = SklearnClassifier(SGDClassifier(alpha=1e-3, random_state=42, penalty="elasticnet", max_iter=5000, tol=None))
    # model_name = "Stochastic Gradient Descent Classifier (elasticnet)"
    model_name = 'SGD_elasticnet'
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # Ridge Classifier
    model = SklearnClassifier(RidgeClassifier(alpha=0.5, tol=1e-2, solver="sag"))
    model_name = "Ridge"
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # Perceptron Classifier
    model = SklearnClassifier(Perceptron(max_iter=5000))
    model_name = "Perceptron"
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # Passive-Aggressive Classifier
    model = SklearnClassifier(PassiveAggressiveClassifier(max_iter=1000))
    model_name = "Passive-Aggressive"
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

from common import createTensor, padding, tokenizeText1, tokenizeText2

from common import read_data

def main():
    """Main function of the program."""
    # Specify path
    training_filepath = 'data/clean_training1.csv'
    # Check whether the specified path exists or not
    isExist = os.path.exists(training_filepath)
    if(isExist):
        print('Reading from ' + training_filepath)
    else:
        print('Training file not found in the app path.')
        exit()
    df = read_data(training_filepath)
    # import pdb; pdb.set_trace()
    # df = df.sample(frac=0.5).reset_index(drop=True)
    df = df.dropna()

    # split into training, validation, and test sets
    training, test = np.array_split(df.head(500), 2)
    labels = training['human_tag']

    # When we have more time
    # model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-large-uncased')
    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    features  = tokenizeText1(training, 'clean_text', model_class, tokenizer_class, pretrained_weights)

    testing_filepath = 'data/clean_testing1.csv'
    # Check whether the specified path exists or not
    isExist = os.path.exists(testing_filepath)
    if(isExist):
        print('Reading from ' + testing_filepath)
    else:
        print('Testing file not found in the app path.')
        exit()
    df1 = read_data(testing_filepath)
    df1 = df1.dropna()
    testing_features  = tokenizeText1(df1, 'clean_text', model_class, tokenizer_class, pretrained_weights)


    trainClassifiers(features, labels, testing_features, df1)


    # features = tokenizeText2(df, 'clean_text', model_class)
    # features  = tokenizeText2(training, 'clean_text', model_class, tokenizer_class, pretrained_weights)
    # trainClassifiers(features, labels)
    # model_class, tokenizer_class, pretrained_weights = (ppb.RobertaModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    # features  = tokenizeText1(training, 'clean_text', model_class, tokenizer_class, pretrained_weights)
    # trainClassifiers(features, labels)

if __name__ == "__main__":
    main()
