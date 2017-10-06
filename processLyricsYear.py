from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import re
from nltk.corpus import stopwords # Import the stop word list
import pandas as pd
import math
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
import itertools
import dataVisualization as dv

decades = ['50','60','70','80','90','00']

df = []
for decade in decades:
    df.append(pd.read_csv('lyrics' + decade + '.csv', quotechar="\""))


ps = PorterStemmer()

porter_stemmer = nltk.stem.porter.PorterStemmer()

with open('./stopwords_eng.txt', 'r') as infile:
    stop_words = infile.read().splitlines()

def porter_tokenizer(text, stemmer=porter_stemmer):
    lower_txt = text.lower()
    tokens = nltk.wordpunct_tokenize(lower_txt)
    stems = [porter_stemmer.stem(t) for t in tokens]
    no_punct = [s for s in stems if re.match('^[a-zA-Z]+$', s) is not None]
    return no_punct

def cosine_similarity(vector1, vector2):
    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if not magnitude:
        return 0
    return dot_product*1.0/magnitude


def lyricstowords(lyrics_raw):
    letters_only = re.sub("[^a-zA-Z]", " ", lyrics_raw)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [ps.stem(w) for w in words if not w in stops]
    return ( " ".join(meaningful_words))

def getCleanLyrics(df1, df2):
    clean_lyrics = []
    df = pd.concat([df1, df2], ignore_index=True)
    num_lyrics = df["lyrics"].size
    #
    for i in range(0, num_lyrics):
        lyrics_string = lyricstowords(df['lyrics'][i])
        clean_lyrics.append(lyrics_string)

    return clean_lyrics

def pairwiseClustering(df1, df2):
    clean_lyrics = getCleanLyrics(df1, df2)
    vec = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english', max_features=5000)
    tfidf_matrix = vec.fit_transform(clean_lyrics)
    feature_names = vec.get_feature_names()
    n1 = len(df1['lyrics'])
    tfidf_vectors = tfidf_matrix.toarray()

    n = len(clean_lyrics)
    distances = [[0 for x in range(n)] for y in range(n)]


    d_file = open('distances_bigram.txt','a+')

    for i in range(n):
        for j in range(n):
            distances[i][j] = 10*round(np.linalg.norm(tfidf_vectors[i] - tfidf_vectors[j]),5)
            d_file.write(str(distances[i][j]))
            if( j != n-1):
                d_file.write(',')
            else:
                d_file.write('\n')

    d_file.close()

    maxx = 0
    minx = 10000
    count = 0
    sum = 0
    for i in range(n):
        for j in range(n):
            if distances[i][j] != 0:
                sum += distances[i][j]
                count += 1
                if (distances[i][j] > maxx):
                    maxx = distances[i][j]
                if (distances[i][j] < minx):
                    minx = distances[i][j]

    import kmedoids

    A = np.matrix(distances)

    n = len(A)

    def cost(d_mat, M, C):
        k = len(M)
        costs = []
        for i in range(k):
            costs.append(0)
        for c_i in range(k):
            for i in C[c_i]:
                costs[c_i] += d_mat[M[c_i], i]

        return np.sum(costs)

    M, C = kmedoids.kMedoids(A, n, 2)

    for i in range(100):
        t_M, t_C = kmedoids.kMedoids(A, n, 2)
        if (cost(A, t_M, t_C) < cost(A, M, C)):
            M = t_M
            C = t_C


    print "Pair || " + str(df1['year'].iloc[0]) + ": " + str(len(df1['year'])) + ", " + str(df2['year'].iloc[0]) + ": " + str(len(df2['year']))
    print "===================================================="
    count1 = 0
    count2 = 0
    print "Cluster 1 : " + str(len(C[0]))
    for point in C[0]:
        if point < n1:
            count1 += 1
        else:
            count2 += 1
    if count1 > count2:
        c_1 = count1
    else:
        c_2 = count2

    print str(df1['year'].iloc[0]) + ": " + str(count1) + ", " + str(df2['year'].iloc[0]) + ": " + str(count2)
    count1 = 0
    count2 = 0
    print "Cluster 2 : " + str(len(C[1]))
    for point in C[1]:
        if point < n1:
            count1 += 1
        else:
            count2 += 1
    if count1 > count2:
        c_1 = count1
    else:
        c_2 = count2
    print str(df1['year'].iloc[0]) + ": " + str(count1) + ", " + str(df2['year'].iloc[0]) + ": " + str(count2)

    accuracy = (c_1 + c_2)*1.0/n

    print "\nAccuracy: " + str(accuracy) + "\n\n===================================================="
    print "\n"

    return accuracy

acc_matrix = [[0.5 for j in range(len(df))] for i in range(len(df))]

for i in range(len(df)-1):
    for j in range(i+1,len(df)):
        acc_matrix[i][j] = pairwiseClustering(df[i],df[j])
        acc_matrix[j][i] = acc_matrix[i][j]

print acc_matrix

dv.plot2d(acc_matrix, decades)




