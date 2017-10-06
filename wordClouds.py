import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from PIL import ImageDraw, Image
import numpy as np
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from wordcloud import WordCloud
import string


ps = PorterStemmer()

def lyricstowords(lyrics_raw):
    letters_only = re.sub("[^a-zA-Z]", " ", lyrics_raw)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return ( " ".join(meaningful_words))

from os import path
from scipy.misc import imread
import matplotlib.pyplot as plt
import random

def generateWordCloud(text,genre):
    stops = list(stopwords.words("english"))
    stops = " ".join(stops)
    import unicodedata
    stops = str(stops)
    wordcloud = WordCloud(stopwords= stops,
                              background_color='white',
                              width=1200,
                              height=1000
                             ).generate(text)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig('wordcloud_' + genre + '.png')
    # plt.show()

# genres = ['Christian', 'Country Music', 'Hip Hop/Rap', 'R&B', 'Pop', 'Rock']
#
# df = pd.read_csv('lyrics_data.csv', quotechar="\"",encoding="utf-8-sig")
#
# df_ch = df.loc[df['genre'] == 'Christian']
# df_co = df.loc[df['genre'] == 'Country Music']
# df_hr = df.loc[df['genre'] == 'Hip Hop/Rap']
# df_rb = df.loc[df['genre'] == 'R&B']
# df_p = df.loc[df['genre'] == 'Pop']
# df_r = df.loc[df['genre'] == 'Rock']
#
# dataframes = [df_ch, df_co, df_hr, df_rb, df_p, df_r]
#
# for i in range(len(genres)):
#     clean_lyrics = []
#     num_lyrics = df['lyrics'].size
#     print num_lyrics
#     for j in range(0, num_lyrics):
#         if(df['genre'][j] == genres[i]):
#             lyrics_string = lyricstowords(df['lyrics'][j])
#             clean_lyrics.append(lyrics_string)
#
#     text = " ".join(clean_lyrics)
#
#     pattern = re.compile('[\W_]+')
#     genre = pattern.sub('', genres[i])
#     generateWordCloud(text,genre)

decades = ['50','60','70','80','90','00']

for decade in decades:
    df = pd.read_csv('lyrics' + decade + '.csv', quotechar="\"")
    clean_lyrics = []
    num_lyrics = df['lyrics'].size
    print num_lyrics
    for j in range(0, num_lyrics):
        lyrics_string = lyricstowords(df['lyrics'][j])
        clean_lyrics.append(lyrics_string)
    text = " ".join(clean_lyrics)

    pattern = re.compile('[\W_]+')
    genre = pattern.sub('', decade)
    generateWordCloud(text,decade)



