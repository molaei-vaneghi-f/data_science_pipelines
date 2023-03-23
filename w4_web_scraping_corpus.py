""" scraping the web and 
creating a corpus with basic text pre-processing functionalities"""

import os
import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin

import nltk
from nltk.corpus import CategorizedPlaintextCorpusReader
from nltk.corpus.reader import PlaintextCorpusReader
from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np
import wordcloud
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

base_url = 'https://www.lyrics.com'
artist_url = requests.get("https://www.lyrics.com/artist/Adele/861756")
corpus_db = './data'
  

def get_urls (artist_url):
    
    urls = []  
    
    soup = BeautifulSoup(artist_url.text, 'lxml')
    songs = soup.find_all(class_= "tal qx") # or 'tar qx'

    for song in songs:
        urls.append(song.find('a'))
    
    return urls


def save_lyrics (urls):
    
    for url in urls[:10]:
        
        song_url = urljoin(base_url, url.get('href'))
        song_title = url.get_text()
        song_page = requests.get(song_url)
        song_soup = BeautifulSoup(song_page.content, 'html.parser')
        lyrics = song_soup.find("pre", {"id": "lyric-body-text"})
        if lyrics:
            filename = f'{song_title}.txt'
            f = open(filename, "w", encoding = 'utf-8')
            f.write(lyrics.get_text())
            f.close()
            print(f"Lyrics for '{song_title}':\n{lyrics.get_text()}\n")
        else:
            print(f"No lyrics found for '{song_title}'.\n")
        urls.remove(url)
        if not urls:
            break



urls = get_urls (artist_url)
save_lyrics (urls)

# %% create te coorps
def create_corpus (corpus_db):
    
    text_pattern = r'.*\.txt'
    corpus = CategorizedPlaintextCorpusReader(corpus_db, text_pattern, cat_pattern=r'(.+)/.*')
    
    fileids = corpus.fileids()
    song_list = [file_name.split("/")[0].split("-")[0] for file_name in corpus.fileids()]
    corpus_text = [corpus.raw(file) for file in fileids]
    corpus_text = [text.replace("\n", " ").replace("\r", " ").replace(",", "").replace("\\", "") for text in corpus_text]
    
    return song_list, corpus_text


song_list =  create_corpus (corpus_db)[0]
corpus_text =  create_corpus (corpus_db)[1]

#%% word cloud
cloud = wordcloud.WordCloud().generate(str(corpus_text[:]))
plt.imshow(cloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#%% vectorization and normalization

vectorizer = CountVectorizer(stop_words='english', max_df=0.8)
X = vectorizer.fit_transform(corpus_text)
X_df = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names(), index=song_list)

X_dense=X.todense()
X_dense.shape

tf = TfidfTransformer() 
X_norm = tf.fit_transform(X)
X_norm_df = pd.DataFrame(X_norm.todense(), columns=vectorizer.get_feature_names(), index=song_list)