from nltk.tokenize import TweetTokenizer # doesn't split at apostrophes
import nltk
import pandas as pd
import numpy as np
import re
import math
from nltk import Text
from nltk.tokenize import regexp_tokenize
from nltk.tokenize import word_tokenize  
from nltk.tokenize import sent_tokenize 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#The data Indonesian Language
corpus = ['Kualitas dan harga barang-barangnya sangat baik.',
             'Gambar dan produk aslinya sesuai gambar.']

#Initialize Stemming
stemmer = PorterStemmer()
#print([[stemmer.stem(token) for token in word_tokenize(document)] for document in corpus])

# Initialize a CountVectorizer object: count_vectorizer
count_vec = CountVectorizer(stop_words=stopwords.words('indonesian'), analyzer='word',
                            ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None)

# Transforms the data into a bag of words
count_train = count_vec.fit(corpus)
bag_of_words = count_vec.transform(corpus)

print(vectorizer.get_feature_names())
vectorizer = TfidfVectorizer(stop_words=stopwords.words('indonesian'), analyzer='word')
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)