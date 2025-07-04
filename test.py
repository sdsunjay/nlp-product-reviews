import pickle
import string

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

def load_model(filename):
    # load the model from disk
    return pickle.load(open(filename, 'rb'))

def read_data(filepath):
    """Read the CSV from disk."""
    df = pd.read_csv(filepath, delimiter=',')

    print('Number of rows in dataframe: ' + str(len(df.index)))
    return df

def tokenizeText1(df, text_column_name, model_class, tokenizer_class,pretrained_weights):
    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Load pretrained model/tokenizer
    try:
        print('Starting to tokenize ' + text_column_name)
        # print(df.head(10))
        # (tokenizer_class, pretrained_weights) = (ppb.DistilBertTokenizer, 'distilbert-base-uncased')
        # want RoBERTa instead of distilBERT, Uncomment the following line:
        # model_class, tokenizer_class, pretrained_weights = (ppb.RobertaModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
        ## Want BERT instead of distilBERT? Uncomment the following line:
        # model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer,'bert-large-uncased')
        model = model_class.from_pretrained(pretrained_weights)
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=True)
        model.resize_token_embeddings(len(tokenizer))
        tokenized = df[text_column_name].apply((lambda x: tokenizer.encode(x,add_special_tokens=True)))

        ### Now let's save our model and tokenizer to a directory
        model.save_pretrained('./my_model/')
        tokenizer.save_pretrained('./my_model/')
        padded = padding(tokenized)
        return createTensor1(padded, model)
    except Exception:
        print("Exception in Tokenize code:")
        print("-"*60)
        traceback.print_exc(file=sys.stdout)
        print("-"*60)
        exit()
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # tokenized = tokenizer.tokenize(df[text_column_name])
        # model.resize_token_embeddings(len(tokenizer))

    print('Finished tokenizing text')
    return (tokenized,model,tokenizer)

def main():

    testing_filepath = 'data/clean_testing.csv'
    # Check whether the specified path exists or not
    isExist = os.path.exists(testing_filepath)
    if(isExist):
        print('Reading from ' + testing_filepath)
    else:
        print('Testing file not found in the app path.')
        exit()

    df = read_data(testing_filepath)

    model = BertForSequenceClassification.from_pretrained('./my_saved_model_directory/')
    tokenizer = BertTokenizer.from_pretrained('./my_saved_model_directory/')

    filepaths = [ 'MLP10','MLP3','MLP5','XGB']

    # Check whether the specified
    # path exists or not
    for filepath in filepaths:
        output_path = 'output_' + filepath + '.csv'
        isExist = os.path.exists(filepath)
        if(isExist):
            print('Reading model from ' + filepath)
        else:
            print('Model not found in the app path (' + filepath +')')
            exit()
        model = load_model(filepath)
        print('Model: ' + filepath)
        y_test_pred = model.predict(df["clean_text"])
        df["human_tag"] = y_test_pred
        header = ["ID", "human_tag"]
        print('Output: ' + output_path)
        df.to_csv(output_path, columns = header)

if __name__ == "__main__":
    main()
