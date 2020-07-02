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
# from IPython.display import Javascript #addedLater2
# display(Javascript("google.colab.output.resizeIframeToContent()"))#addedLater2

books = pd.read_csv("books.csv", encoding = "ISO-8859-1")
# books.shape
ratings = pd.read_csv("ratings.csv", encoding = "ISO-8859-1")
# ratings.head()
book_tags = pd.read_csv('book_tags.csv', encoding = "ISO-8859-1")
# book_tags.head()
tags = pd.read_csv('tagNames.csv')
# tags.tail()

tags_join_DF = pd.merge(book_tags, tags, left_on='tag_id', right_on='tag_id', how='inner')
# tags_join_DF.head(30)
books_with_tags = pd.merge(books, tags_join_DF, left_on='book_id', right_on='goodreads_book_id', how='inner')
# books_with_tags.head(30)
temp_df = books_with_tags.groupby('book_id')['tag_name'].apply(' '.join).reset_index()
# temp_df.head(30)
books = pd.merge(books, temp_df, left_on='book_id', right_on='book_id', how='inner')
# books.head()
books['corpus'] = (pd.Series(books[['authors', 'tag_name']].fillna('').values.tolist()).str.join(' '))
books.head()
tf_corpus = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix_corpus = tf_corpus.fit_transform(books['corpus'])
cosine_sim_corpus = linear_kernel(tfidf_matrix_corpus, tfidf_matrix_corpus)
# cosine_sim_corpus

titles = books['title']
corpus = books['corpus']
isbns = books['isbn']
# titles[1: 20]
indices = pd.Series(books.index, index=books['title'])
# indices
def corpus_recommendations(title1, title2, title3):
    idx1 = indices[title1]
    idx2 = indices[title2]
    idx3 = indices[title3]
    sim_scores1 = list(enumerate(cosine_sim_corpus[idx1]))
    sim_scores2 = list(enumerate(cosine_sim_corpus[idx2]))
    sim_scores3 = list(enumerate(cosine_sim_corpus[idx3]))
    tot_scores = list(enumerate(cosine_sim_corpus[idx1] + cosine_sim_corpus[idx2] + cosine_sim_corpus[idx3]))
    # length = len(sim_scores1)
    # for i in range(length):
    #   tot_scores[i] = sim_scores1[i] + sim_scores2[i] + sim_scores3[i]
    
    # print(sim_scores1)
    # print(sim_scores2)
    # print(sim_scores3)
    # print(tot_scores)
    # return sim_scores1
    tot_scores = sorted(tot_scores, key=lambda x: x[1], reverse=True)
    tot_scores = tot_scores[1:21]
    book_indices = [i[0] for i in tot_scores]
    # return titles.iloc[book_indices]

    for x in book_indices:
    	if len(isbns.iloc[x]) < 10:
    		# print('0'+str(isbns.iloc[x]), titles.iloc[x])
    		isbnId = '0'+str(isbns.iloc[x])
    	else :
    		isbnId = str(isbns.iloc[x])
    		# print('0'+str(isbns.iloc[x]), titles.iloc[x])
    	print(isbnId, titles.iloc[x])
    	api.gatherData(isbnId)
    	rev.cleanhtml(isbnId)
    	rev.review(isbnId)

# corpus_recommendations("So You Want to Be a Wizard (Young Wizards, #1)", "Harry Potter and the Prisoner of Azkaban (Harry Potter, #3)", "Harry Potter and the Order of the Phoenix (Harry Potter, #5)")
print(corpus_recommendations('Jane Eyre', 'Emma', 'Pride and Prejudice'))
# print(corpus_recommendations("Outliers: The Story of Success", "Blink: The Power of Thinking Without Thinking", "Outliers: The Story of Success"))
# display.Javascript("google.colab.output.setIframeWidth('300px');") #added later 1