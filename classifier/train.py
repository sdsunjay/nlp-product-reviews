import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common import tokenize_text1, save_model, load_model, train_classifier as trainClassifier
import bert_model_training

from datetime import datetime

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.linear_model import RidgeClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def trainClassifiers(features, labels):
    begin_time_train = datetime.now()
    print("Begin training models: ", begin_time_train.strftime("%m/%d/%Y, %H:%M:%S"))
    # create training and testing vars
    X_train, X_test, y_train, y_test = train_test_split(features, labels,test_size=0.2)

    # Create a simple Logistic Regression classifier
    model_name = 'Logistic Regression (solver=lbfgs)'
    model = LogisticRegression(max_iter=200)
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    model_name = 'MLP100_30layers'
    model = MLPClassifier(hidden_layer_sizes=(100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100))
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

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

    end_time = datetime.now()
    print("End time: ", end_time.strftime("%m/%d/%Y, %H:%M:%S"))
    print("End time Duration: " + str(end_time - begin_time_train))

def testing(features, labels, pretrained_weights):

    testing_filepath = 'data/clean_testing1.csv'
    # Check whether the specified path exists or not
    if not os.path.exists(testing_filepath):
        print('Testing file not found in the app path.')
        return
    print('Reading from ' + testing_filepath)
    df1 = bert_model_training.load_dataset(testing_filepath)
    df1 = df1.dropna()
    a = np.array_split(df1, 8)
    values = []
    for i, aa in enumerate(a):
        print('Run: ' + str(i))
        testing_features = tokenize_text1(aa, 'clean_text', AutoModel, AutoTokenizer, pretrained_weights)
        trainClassifiers(testing_features, labels)


def main():
    """Main function of the program."""

    begin_time_main = datetime.now()
    print("Begin time: ", begin_time_main.strftime("%m/%d/%Y, %H:%M:%S"))

    df = bert_model_training.load_dataset(bert_model_training.PATH_TO_TRAINING_CSV)
    labels = df['human_tag']

    pretrained_weights = 'allenai/longformer-base-4096'
    features = tokenize_text1(
        df,
        'clean_text',
        AutoModel,
        AutoTokenizer,
        pretrained_weights,
    )


    end_time = datetime.now()
    print("End time: ", end_time.strftime("%m/%d/%Y, %H:%M:%S"))
    print("Duration: " + str(end_time - begin_time_main))

    trainClassifiers(features, labels)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_weights)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_weights,
        num_labels=2,
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    bert_model_training.train_model(
        df,
        tokenizer,
        model,
        pretrained_weights,
        bert_model_training.NUM_EPOCHS,
    )

if __name__ == "__main__":
    main()
