import nltk
nltk.download('stopwords')

import numpy as np
import pandas as pd
import re
import spacy
from spacy.tokenizer import Tokenizer
from spacy.symbols import ORTH
import contractions
import unicodedata
import wordninja

from collections import defaultdict, Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

class data_cleaning_jobpost:

    ## INITIALIZE INPUTS AND VARS
    def __init__(self,*,sample_text, nlp):
        self.sample_text = sample_text.lower()

        self.special_cases = {"tf-idf": [{"ORTH": "tf-idf"}], "t-sne": [{"ORTH": "t-sne"}], "d3.js": [{"ORTH": "d3.js"}],
                        "ggplot2": [{"ORTH": "ggplot2"}], "scikit-learn": [{"ORTH": "scikit-learn"}], 
                         "ci/cd": [{"ORTH": "ci/cd"}], "c#": [{"ORTH": "c#"}], "amazon s3": [{"ORTH": "amazon s3"}], 
                         "s3": [{"ORTH": "s3"}], "f-test": [{"ORTH": "f-test"}], "chi-square": [{"ORTH": "chi-square"}], 
                         "pgadmin4": [{"ORTH": "pgadmin4"}], "er/studio": [{"ORTH": "er/studio"}], 
                         "draw.io": [{"ORTH": "draw.io"}], "metatrader4": [{"ORTH": "metatrader4"}],
                         "asp.net": [{"ORTH": "asp.net"}], "html5": [{"ORTH": "html5"}], "j2ee": [{"ORTH": "j2ee"}],
                         "jax-rs 4": [{"ORTH": "jax-rs 4"}], "ui/ux": [{"ORTH": "ui/ux"}], "a/b tests": [{"ORTH": "a/b tests"}],
                         "a/b testing": [{"ORTH": "a/b testing"}], "caffe2": [{"ORTH": "caffe2"}], 
                         "h20": [{"ORTH": "h20"}]
                        }
        self.TAG_RE = re.compile('<[^>]+>')

        self.special_character_remover = re.compile('[/(){}\[\]\|@,;_]')
        self.extra_symbol_remover = re.compile('[^0-9a-z #+_]')
        self.STOPWORDS = set(stopwords.words('english'))
        self.nlp = nlp
        


    ## INITIALIZE SUB FUNCTIONS
    def expand_contractions(self,*,text):
        result = text
        try:
            result = contractions.fix(text)
        except:
            pass

        contractions = {
            "ain't": "am not / are not",
            "aren't": "are not / am not",
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
            "he'd": "he had / he would",
            "he'd've": "he would have",
            "he'll": "he shall / he will",
            "he'll've": "he shall have / he will have",
            "he's": "he has / he is",
            "how'd": "how did",
            "how'd'y": "how do you",
            "how'll": "how will",
            "how's": "how has / how is",
            "i'd": "I had / I would",
            "i'd've": "I would have",
            "i'll": "I shall / I will",
            "i'll've": "I shall have / I will have",
            "i'm": "I am",
            "i've": "I have",
            "isn't": "is not",
            "it'd": "it had / it would",
            "it'd've": "it would have",
            "it'll": "it shall / it will",
            "it'll've": "it shall have / it will have",
            "it's": "it has / it is",
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
            "she'd": "she had / she would",
            "she'd've": "she would have",
            "she'll": "she shall / she will",
            "she'll've": "she shall have / she will have",
            "she's": "she has / she is",
            "should've": "should have",
            "shouldn't": "should not",
            "shouldn't've": "should not have",
            "so've": "so have",
            "so's": "so as / so is",
            "that'd": "that would / that had",
            "that'd've": "that would have",
            "that's": "that has / that is",
            "there'd": "there had / there would",
            "there'd've": "there would have",
            "there's": "there has / there is",
            "they'd": "they had / they would",
            "they'd've": "they would have",
            "they'll": "they shall / they will",
            "they'll've": "they shall have / they will have",
            "they're": "they are",
            "they've": "they have",
            "to've": "to have",
            "wasn't": "was not",
            "we'd": "we had / we would",
            "we'd've": "we would have",
            "we'll": "we will",
            "we'll've": "we will have",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what shall / what will",
            "what'll've": "what shall have / what will have",
            "what're": "what are",
            "what's": "what has / what is",
            "what've": "what have",
            "when's": "when has / when is",
            "when've": "when have",
            "where'd": "where did",
            "where's": "where has / where is",
            "where've": "where have",
            "who'll": "who shall / who will",
            "who'll've": "who shall have / who will have",
            "who's": "who has / who is",
            "who've": "who have",
            "why's": "why has / why is",
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
            "you'd": "you had / you would",
            "you'd've": "you would have",
            "you'll": "you shall / you will",
            "you'll've": "you shall have / you will have",
            "you're": "you are",
            "you've": "you have",
            "sr.": "senior",
            "jr.": "junior",
            "sr": "senior",
            "jr": "junior",
            "ml": "machine learning"
        }

        for word in result.split():
            if word in contractions:
                result = result.replace(word, contractions[word])

        return result

    def remove_accented_chars(self,*,text):
        new_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return new_text

    def remove_extra_white_spaces(self,*,text):
        text = re.sub(' +', ' ', text)
        return text

    def remove_tags(self,*,text):
        text = str(text)
        if text != "":
            return self.TAG_RE.sub(' ', text)
        else:
            return text

    def token_descs(self,*,text):
        nlp=self.nlp
        nlp.tokenizer = Tokenizer(nlp.vocab, rules=self.special_cases)
        doc = nlp(text)
        return doc

    # get lemmatized words

    def get_words(self,*,doc):
        words = []
        for word in doc:
            if word.is_stop: # remove stop words
                continue
            elif word.like_url:
                continue
            elif word.is_digit:
                continue
            elif word.is_punct:
                continue
            elif word.like_email:
                continue
            else:
                words.append(word.text)

        return " ".join(words)

    def clean_text(self,*,text):
        text = self.special_character_remover.sub(' ', text)
        text = self.extra_symbol_remover.sub('', text)
        text = ' '.join(word for word in text.split() if word not in self.STOPWORDS)
        return text

    def remove_http(self,*,text):
        text = re.sub(r'http.?', '', text)
        text = re.sub(r'\bwww\w+com\s', '', text)
        return text

    def generate_clean_text(self):
        sample_text = self.sample_text
        sample_text = self.expand_contractions(text=str(sample_text))
        sample_text = self.remove_accented_chars(text=str(sample_text))
        sample_text = self.remove_extra_white_spaces(text=str(sample_text))
        sample_text = self.remove_tags(text=sample_text)
        sample_text = re.sub("\r\n", ' ', sample_text)
        tokenized = self.token_descs(text=sample_text)
        tokenized = self.get_words(doc=tokenized)       
        clean_text = self.clean_text(text=tokenized)
        clean_text = self.remove_http(text=clean_text)
        return clean_text