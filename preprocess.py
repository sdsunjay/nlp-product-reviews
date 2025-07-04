import datetime
import pandas as pd

import os.path
import sys, traceback
import re

import pickle
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import contractions
import emoji
import ftfy

# Download the NLTK stopwords and WordNet lemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.probability import FreqDist

MAX_TOKENS = 1000
MAX_WORDS = 1000

import emoji
import unicodedata

stop_words = {
    "will", "done", "goes", "let", "know", "just", "put", "also",
    "got", "can", "get", "said", "mr", "mrs", "one", "two", "three",
    "four", "five", "i", "me", "my", "myself", "we", "our",
    "ours", "ourselves", "you", "youre", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because",
    "as", "until", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "shouldn't", "don't", "should've", "now", "d", "ll", "m", "o", "re", "ve",
    "y", "ain", "aren", "aren't", "couldn", "couldn't", "didn", "didn't",
    "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't", "haven",
    "haven't", "isn", "isn't", "ma", "mightn", "mightn't", "mustn",
    "mustn't", "needn", "needn't", "shan", "shan't", "wasn", "wasn't",
    "weren", "weren't", "won", "won't", "wouldn't"}

def remove_stop_words(text):
    words = text.split()
    final_string = []
    try:
        for word in words:
            word = word.lower()
            if word not in stop_words:
                final_string.append(word)
    except Exception as e:
        print("type error")
        exit()
    return ' '.join(final_string)

def replace_emojis(text):
    return emoji.demojize(text, delimiters=("", ""))

def clean_text(text):
    # Check if text is not a string
    if pd.isna(text) or not isinstance(text, str):
        return ""  # Return an empty string or some other placeholder text

    # Expand contractions
    text = contractions.fix(text)
    # replace emoji with wor
    translated_text = replace_emojis(text)
    fixed_text = ftfy.fix_text(translated_text)
    cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    text = cleanr.sub('', fixed_text.strip())
    stop_words_removed = remove_stop_words(unicodedata.normalize('NFC', text))
    return stop_words_removed

def read_data(filepath):
    """Read the CSV from disk."""
    df = pd.read_csv(filepath, delimiter=',')
    # Drop optional metadata columns if they exist
    for col in ["doc_id", "date", "title"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    print('Cleaning text')
    df["clean_text"] = df['text'].apply(clean_text)
    print('Number of rows in dataframe: ' + str(len(df.index)))
    return df


def read_clean_data(filepath: str) -> pd.DataFrame:
    """Read a CSV containing already cleaned reviews.

    The file is expected to contain the columns ``clean_text``, ``star_rating``,
    and ``human_tag`` as produced by ``preprocess.py``.
    """

    df = pd.read_csv(filepath, delimiter=',')
    required_columns = {"clean_text", "star_rating", "human_tag"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns {sorted(missing)} in {filepath}"
        )
    print(f"Loaded {len(df.index)} rows from {filepath}")
    return df

def main():
    """Main function of the program."""
    # Specify path
    training_filepath = 'data/training.csv'
    testing_filepath = 'data/public_test_features.csv'

    # Check whether the specified path exists or not
    isExist = os.path.exists(training_filepath)
    if(isExist):
        print('Reading from ' + training_filepath)
    else:
        print('Training file not found in the app path.')
        exit()

    # Get the current date and time
    now = datetime.datetime.now()

    # Format the current time as a string in the "HH:MM" format
    current_time = now.strftime("%y_%m_%d_%H_%M")
    df = read_data(training_filepath)
    header = ["clean_text", "star_rating", "human_tag"]
    output_path = './data/25_07_04_11_18_clean_training.csv'
    print(f"Outputing to {output_path}")
    df.to_csv(output_path, columns = header, index=False)

if __name__ == "__main__":
    main()

