import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import handlingApi as api
import raw2apt as rev 
import requests
import os
from xml.etree import ElementTree
import re


books = pd.read_csv("books.csv", encoding = "ISO-8859-1")

ratings = pd.read_csv("ratings.csv", encoding = "ISO-8859-1")

book_tags = pd.read_csv('book_tags.csv', encoding = "ISO-8859-1")

tags = pd.read_csv('tagNames.csv')

tags_join_DF = pd.merge(book_tags, tags, left_on='tag_id', right_on='tag_id', how='inner')

books_with_tags = pd.merge(books, tags_join_DF, left_on='book_id', right_on='goodreads_book_id', how='inner')

temp_df = books_with_tags.groupby('book_id')['tag_name'].apply(' '.join).reset_index()

books = pd.merge(books, temp_df, left_on='book_id', right_on='book_id', how='inner')

books['corpus'] = (pd.Series(books[['authors', 'tag_name']].fillna('').values.tolist()).str.join(' '))
books.head()
tf_corpus = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix_corpus = tf_corpus.fit_transform(books['corpus'])
cosine_sim_corpus = linear_kernel(tfidf_matrix_corpus, tfidf_matrix_corpus)

titles = books['title']
corpus = books['corpus']
isbns = books['isbn']

indices = pd.Series(books.index, index=books['title'])

def corpus_recommendations(title1, title2, title3):
    idx1 = indices[title1]
    idx2 = indices[title2]
    idx3 = indices[title3]
    sim_scores1 = list(enumerate(cosine_sim_corpus[idx1]))
    sim_scores2 = list(enumerate(cosine_sim_corpus[idx2]))
    sim_scores3 = list(enumerate(cosine_sim_corpus[idx3]))
    tot_scores = list(enumerate(cosine_sim_corpus[idx1] + cosine_sim_corpus[idx2] + cosine_sim_corpus[idx3]))
   
    tot_scores = sorted(tot_scores, key=lambda x: x[1], reverse=True)
    tot_scores = tot_scores[1:21]
    book_indices = [i[0] for i in tot_scores]
   

    for x in book_indices:
    	if len(isbns.iloc[x]) < 10:
    		
    		isbnId = '0'+str(isbns.iloc[x])
    	else :
    		isbnId = str(isbns.iloc[x])
    		
    	print(isbnId, titles.iloc[x])
    	api.gatherData(isbnId)
    	rev.cleanhtml(isbnId)
    	rev.review(isbnId)


print(corpus_recommendations('Jane Eyre', 'Emma', 'Pride and Prejudice'))

