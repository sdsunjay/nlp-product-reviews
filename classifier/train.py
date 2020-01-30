
import numpy as np
import pandas as pd
import torch
import transformers as ppb  # pytorch transformers
from sklearn.linear_model import LogisticRegression
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

from pytorch_pretrained_bert import BertTokenizer

import pickle
import string

from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

MAX_LEN = 1024
MAX_TOKENS = 512
MAX_WORDS = 475

def save_model(model, filename):
    print('Saving model to ' + filename)
    # filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))

def load_model(filename):
    # load the model from disk
    return pickle.load(open(filename, 'rb'))

def trainClassifiers(features, labels):
    print('Starting training')
    # create training and testing vars
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

    lr_clf = LogisticRegression()
    model = lr_clf.fit(X_train, y_train)

    print('Logistic Regiression')
    y_test_pred = lr_clf.predict(X_test)
	print(classification_report(y_test, np.around(y_test_pred)))
    print(roc_auc_score(y_test, y_test_pred))
    score = model.score(X_test, y_test)
    print('Score: ' + str(score))
    # save_model(lr_clf, 'LR_classifier')

    print('Linear SVC')
    # Create a simple Linear SVC classifier
    svm_classifier = svm.LinearSVC(random_state=42)
    svm_classifier.fit(X_train, y_train)
    y_test_pred = svm_classifier.predict(X_test)
	print(classification_report(y_test, np.around(y_test_pred)))
    print(roc_auc_score(y_test, y_test_pred))

    # Logistic Regression with Sklearn and random state
    print('Logistic regression model 2 (random state 42)')
    LG_classifier = SklearnClassifier(LogisticRegression(random_state=42))
    LG_classifier.fit(X_train, y_train)
    y_test_pred = LG_classifier.predict(X_test)
	print(classification_report(y_test, np.around(y_test_pred)))
    print(roc_auc_score(y_test, y_test_pred))
    score = model.score(X_test, y_test)
    print('Score: ' + str(score))

    # Naive Bayes Classifier
    print('Naive Bayes Classifier')
    NB_classifier = NaiveBayesClassifier()
    NB_classifier.fit(X_train, y_train)
    y_test_pred = NB_classifier.predict(X_test)
	print(classification_report(y_test, np.around(y_test_pred)))
    print(roc_auc_score(y_test, y_test_pred))
    score = model.score(X_test, y_test)
    print('Score: ' + str(score))

    print("Sparse Support Vector Classifier")
    SVC_classifier = SklearnClassifier(SVC(),sparse=False).train(train_features)
    SVC_classifier.fit(X_train, y_train)
    y_test_pred = SVC_classifier.predict(X_test)
	print(classification_report(y_test, np.around(y_test_pred)))
    print(roc_auc_score(y_test, y_test_pred))
    score = model.score(X_test, y_test)
    print('Score: ' + str(score))
    # save_model(NB_classifier, 'SVC_classifier')

    print("Linear Support Vector Classifier 1")
    LinearSVC_classifier1 = SklearnClassifier(SVC(kernel='linear', probability=True, tol=1e-3))
    LinearSVC_classifier1.fit(X_train, y_train)
    y_test_pred = LinearSVC_classifier1.predict(X_test)
	print(classification_report(y_test, np.around(y_test_pred)))
    print(roc_auc_score(y_test, y_test_pred))
    score = model.score(X_test, y_test)
    print('Score: ' + str(score))

    print("Linear Support Vector Classifier 2")
    LinearSVC_classifier2 = SklearnClassifier(LinearSVC("l1", dual=False, tol=1e-3))
    LinearSVC_classifier2 = SklearnClassifier(SVC(kernel='linear', probability=True, tol=1e-3))
    LinearSVC_classifier2.fit(X_train, y_train)
    y_test_pred = LinearSVC_classifier2.predict(X_test)
	print(classification_report(y_test, np.around(y_test_pred)))
    print(roc_auc_score(y_test, y_test_pred))
    score = model.score(X_test, y_test)
    print('Score: ' + str(score))

    print("Linear Support Vector Classifier 3")
    LinearSVC_classifier3 = SklearnClassifier(LinearSVC("l2", dual=False, tol=1e-3))
    LinearSVC_classifier3 = SklearnClassifier(SVC(kernel='linear', probability=True, tol=1e-3))
    LinearSVC_classifier3.fit(X_train, y_train)
    y_test_pred = LinearSVC_classifier3.predict(X_test)
	print(classification_report(y_test, np.around(y_test_pred)))
    print(roc_auc_score(y_test, y_test_pred))
    score = model.score(X_test, y_test)
    print('Score: ' + str(score))

def createTensor1(padded, model):
    print('create_tensor1')
    input_ids = torch.tensor(np.array(padded))
    with torch.no_grad():
        last_hidden_states = model(input_ids)
        # Slice the output for the first position for all the sequences, take all hidden unit outputs
        features = last_hidden_states[0][:, 0, :].numpy()
        print('Finished creating features')
        return features

def createTensor2(input_ids, model):
    print('create_tensor2')
    with torch.no_grad():
        last_hidden_states = model(input_ids)
        # Slice the output for the first position for all the sequences, take all hidden unit outputs
        features = last_hidden_states[0][:, 0, :].numpy()
        print('Finished creating features')
        return features

def padding(tokenized):
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
    # np.array(padded).shape
    print('Finished padding text')
    return padded

def something(df, text_column_name):
	### Let's load a model and tokenizer
	model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	### Do some stuff to our model and tokenizer
	# Ex: add new tokens to the vocabulary and embeddings of our model
	tokenizer.add_tokens(['[SPECIAL_TOKEN_1]', '[SPECIAL_TOKEN_2]'])
	model.resize_token_embeddings(len(tokenizer))
	# Train our model
	train(model)

	### Now let's save our model and tokenizer to a directory
	model.save_pretrained('./my_saved_model_directory/')
	tokenizer.save_pretrained('./my_saved_model_directory/')

def tokenizeText1(df, text_column_name, model_class):
    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Load pretrained model/tokenizer
    try:
        print('Starting to tokenize ' + text_column_name)
        tokenizer_class, pretrained_weights = (ppb.DistilBertTokenizer, 'distilbert-base-uncased')
        # want RoBERTa instead of distilBERT, Uncomment the following line:
        # model_class, tokenizer_class, pretrained_weights = (ppb.RobertaModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
        ## Want BERT instead of distilBERT? Uncomment the following line:
        # model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer,'bert-large-uncased')
        model = model_class.from_pretrained(pretrained_weights)
        tokenizer = tokenizer_class.from_pretrained('distilbert-base-uncased', do_lower_case=True)
        model.resize_token_embeddings(len(tokenizer))
        tokenized = df[text_column_name].apply((lambda x: tokenizer.encode(x,add_special_tokens=True)))
        # tokens = df[text_column_name].apply((lambda x: tokenizer.tokenize(x)[:511]))
        # input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))
        # tokenized = df[text_column_name].apply((lambda x: tokenizer.encode(x,add_special_tokens=True)))

        ### Now let's save our model and tokenizer to a directory
        # model.save_pretrained('./my_model/')
        # tokenizer.save_pretrained('./my_model/')
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

def tokenizeText2(df, text_column_name, model_class):
    # Load pretrained model/tokenizer
    try:
        print('Starting to tokenize 2' + text_column_name)
        tokenizer_class, pretrained_weights = (ppb.DistilBertTokenizer, 'distilbert-base-uncased')
        model = model_class.from_pretrained(pretrained_weights)
        tokenizer = tokenizer_class.from_pretrained('distilbert-base-uncased', do_lower_case=True)
        model.resize_token_embeddings(len(tokenizer))
        tokens = df[text_column_name].apply((lambda x: tokenizer.tokenize(x)[:511]))
        input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))
        # tokenized = df[text_column_name].apply((lambda x: tokenizer.encode(x,add_special_tokens=True)))

        ### Now let's save our model and tokenizer to a directory
        # model.save_pretrained('./my_model/')
        # tokenizer.save_pretrained('./my_model/')
        # padded = padding(tokenized)
        return createTensor2(input_ids, model):
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

def truncate(text):
    """Truncate the text."""
    # TODO fix this to use a variable instead of 511
    text = (text[:511]) if len(text) > MAX_TOKENS else text
    return text

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def contractions(text):
    contractions = {
        "ain't": "are not ",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i had",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it had",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as ",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you had",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"}

    words = text.split()
    final_string = ""
    try:
        for word in words:
            word = word.lower()
            if hasNumbers(word) == False:
                if word in contractions:
                    # print('Word: ' + word)
                    # print('Replacement: ' + contractions[word])
                    final_string += contractions[word]
                    final_string += ' '
                    flag = True
                else:
                    final_string += word
                    final_string += ' '
                    flag = False
        if(flag):
            final_string = final_string[:-1]
    except Exception as e:
        print("type error: " + str(e))
        exit()
    return final_string

def removePunctuationFromList(all_words):
    all_words = [''.join(c for c in s if c not in string.punctuation)
            for s in all_words]
    # Remove the empty strings:
    all_words = [s for s in all_words if s]
    return all_words

def cleanText(text):
    """Clean up the text."""
    try:
        text = str(text)

        # remove contactions and stop words
        text = contractions(text)
        # remove html entities
        cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        new_text = cleanr.sub('', text.strip())
        return re.sub(r'\s+', ' ', re.sub(r'\W+', " ", new_text))
        # TAG_RE = re.compile(r'<[^>]+>')
    except:
        # source = text
        # source = source.str.replace('[^A-Za-z]',' ')
        # data['description'] = data['description'].str.replace('\W+',' ')
        # source = source.str.lower()
        # source = source.str.replace("\s\s+" , " ")
        # source = source.str.replace('\s+[a-z]{1,2}(?!\S)',' ')
        # source = source.str.replace("\s\s+" , " ")
        print("An exception occurred with: " + text)
        return str(text)

def getAllWords(lines, stop_words):
    all_words = {}
    try:
        for line in lines:
            words = line.split()
            for word in words:
                if word not in stop_words:
                    all_words[word] = True
        temp = all_words.keys()
        # removePunctuationFromList(temp)


        top_words = nltk.FreqDist(temp)
        print("All Words list length : ", len(top_words))
        # print(str(list(all_words1.keys())[:100]))

        # use top 20000 words
        return list(top_words.keys())[:20000]
        # word_features = list(all_words.keys())[:6000]
        # featuresets = [(find_features(rev, word_features), category)
        #        for (rev, category) in documents]
        # print("Feature sets list length : ", len(featuresets))
    except Exception as e:
        print("type error: " + str(e))
        exit()



def removeWordsNotIn(text, stop_words):
    words = text.split()
    final_string = ""
    flag = False
    try:
        for word in words:
            word = word.lower()
            if word not in stop_words:
                final_string += word
                final_string += ' '
                flag = True
            else:
                flag = False
        if(flag):
            final_string = final_string[:-1]
    except Exception as e:
        # print("type error: " + str(e))
        print("type error")
        exit()
    return final_string

def addWordsIn(text, all_words):
    """ Also does truncation """
    # print('Adding only the top words')
    count = 0
    final_string = ""
    try:
        words = text.split()
        for word in words:
            word = word.lower()

            if word in all_words:
                count += 1
                if(count == MAX_WORDS-1):
                    # if we hit max number of token, stop parsing string
                    return final_string[:-1]
                else:
                    final_string += word
                    final_string += ' '
        final_string = final_string[:-1]
    except Exception as e:
        print("Error")
        # exit()
        print("type error: " + str(e))
    return final_string

def read_data(filepath):
    """Read the CSV from disk."""
    df = pd.read_csv(filepath, delimiter=',')

    stop_words = ["will", "done", "goes","let", "know", "just", "put" "also",
            "got", "can", "get" "said", "mr", "mrs", "one", "two", "three",
            "four", "five", "i", "me", "my", "myself", "we", "our",
            "ours","ourselves","you","youre","your","yours","yourself","yourselves","he","him","his","himself","she","her","hers","herself","it","its","itself","they","them","their","theirs","themselves","what","which","who","whom","this","that","these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did","doing","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about","against","between","into","through","during","before","after","above","below","to","from","up","down","in","out","on","off","over","under","again","further","then","once","here","there","when","where","why","how",
            "all","any","both","each","few","more","most","other","some","such",
            "can", "will",
            "just",
            "don",
            "don't",
            "should",
            "should've",
            "now",
            "d",
            "ll",
            "m",
            "o",
            "re",
            "ve",
            "y",
            "ain",
            "aren",
            "aren't",
            "couldn",
            "couldn't",
            "didn",
            "didn't",
            "doesn",
            "doesn't",
            "hadn",
            "hadn't",
            "hasn",
            "hasn't",
            "haven",
            "haven't",
            "isn",
            "isn't",
            "ma",
            "mightn",
            "mightn't",
            "mustn",
            "mustn't",
            "needn",
            "needn't",
            "shan",
            "shan't",
            "shouldn"
            "shouldn't",
            "wasn",
            "wasn't",
            "weren",
            "weren't",
    "won",
"won't",
"wouldn",
"wouldn't"]

    # pandas drop columns using list of column names
    df = df.drop(['ID', 'doc_id', 'date', 'title', 'star_rating'], axis=1)
    print('Cleaning text')
    df["clean_text"] = df['text'].apply(cleanText)
    print('Removing words in stop words')
    df['clean_text'] = [removeWordsNotIn(line, stop_words) for line in df['clean_text']]

    clean_text = df["clean_text"].tolist()
    # print(clean_text[:10])
    print('Getting all words')
    all_words = getAllWords(clean_text, stop_words)
    # print('adding words in all_words')
    df['clean_text'] = [addWordsIn(line, all_words) for line in df['clean_text']]

    # df.text = df.text.apply(lambda x: x.translate(None, string.punctuation))
    # df.clean_text = df.clean_text.apply(lambda x: x.translate(string.digits))
    # df["clean_text"] = df['text'].str.replace('[^\w\s]','')
    print('Finished reading and cleaning data')
    print('Number of rows in dataframe: ' + str(len(df.index)))
    # print(df.head(30))
    return df

def main(training_filepath):
    """Main function of the program."""
    df = read_data(training_filepath)
    # df.clean_text.to_csv('clean_text.csv')
    # split into training, validation, and test sets
    training, test = np.array_split(df.head(4000), 2)
    # tokenizer = getTokenizer(training, 'clean_text')

    features  = tokenizeText1(training, 'clean_text',ppb.DistilBertModel)
    labels = training['human_tag']
    trainClassifiers(features, labels)

    # features  = tokenizeText2(training, 'clean_text',ppb.DistilBertModel)
    # labels = training['human_tag']
    # trainClassifiers(features, labels)


if __name__ == "__main__":
    # Specify path
    training_filepath = 'data/training.csv'

    # Check whether the specified
    # path exists or not
    isExist = os.path.exists(training_filepath)
    if(isExist):
        print('Reading from ' + training_filepath)
    else:
        print('Training file not found in the app path.')
        exit()
    main(training_filepath)
