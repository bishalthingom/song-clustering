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

df = pd.read_csv('lyrics_data.csv', quotechar="\"")

df_ch = df.loc[df['genre'] == 'Christian']
df_co = df.loc[df['genre'] == 'Country Music']
df_hr = df.loc[df['genre'] == 'Hip Hop/Rap']
df_rb = df.loc[df['genre'] == 'R&B']
df_p = df.loc[df['genre'] == 'Pop']
df_r = df.loc[df['genre'] == 'Rock']

genres = ['Christian', 'Country Music', 'Hip Hop/Rap', 'R&B', 'Pop', 'Rock']

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


    print "Pair || " + df1['genre'].iloc[0] + ": " + str(len(df1['genre'])) + ", " + df2['genre'].iloc[0] + ": " + str(len(df2['genre']))
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

    print df1['genre'].iloc[0] + ": " + str(count1) + ", " + df2['genre'].iloc[0] + ": " + str(count2)
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
    print df1['genre'].iloc[0] + ": " + str(count1) + ", " + df2['genre'].iloc[0] + ": " + str(count2)

    accuracy = (c_1 + c_2)*1.0/n

    print "\nAccuracy: " + str(accuracy) + "\n\n===================================================="
    # dt = [('len', float)]
    # A = np.array(distances)

    # A = A.view(dt)
    #
    # G = nx.from_numpy_matrix(A)
    # G = nx.nx_agraph.to_agraph(G)
    #
    # for i in range(0, n):
    #     node = G.get_node(i)
    #     if i < n1:
    #         node.attr['color'] = "#99CCFF"
    #     else:
    #         node.attr['color'] = "#FF9999"
    #
    # G.node_attr.update(style="filled", height="0.2", width = "0.2")
    # G.edge_attr.update(color="none", width="0.02", style="dashed")
    #
    # genre1 = df1['genre'].iloc[0]
    # genre2 = df2['genre'].iloc[0]
    # filename = 'distances_' + genre1[:2] + '_' + genre2[:2] + '.png'
    #
    # G.draw(filename, format='png', prog='neato');
    #
    # for i in C[0]:
    #     node = G.get_node(i)
    #     node.attr['color'] = "#3333FF"
    # for i in C[1]:
    #     node = G.get_node(i)
    #     node.attr['color'] = "#FF0000"
    #
    # G.node_attr.update(style="filled", height="0.2", width = "0.2")
    # G.edge_attr.update(color="none", width="0.02", style="dashed")
    #
    # filename = 'distances_clustered_' + genre1[:2] + '_' + genre2[:2] + '.png'
    # G.draw(filename, format='png', prog='neato')

    print "\n"

    return accuracy


def overallClustering(df1, df2, df3, df4, df5, df6):
    clean_lyrics = getCleanLyrics(df1, df2)
    clean_lyrics += getCleanLyrics(df3, df4)
    clean_lyrics += getCleanLyrics(df5, df6)

    n = [0 for i in range(len(6))]
    n[0] = len(df1['lyrics'])
    n[1] = n[0] + len(df2['lyrics'])
    n[2] = n[1] + len(df3['lyrics'])
    n[3] = n[2] + len(df4['lyrics'])
    n[4] = n[3] + len(df5['lyrics'])
    n[5] = n[4] + len(df6['lyrics'])


    vec = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english', max_features=5000)
    tfidf_matrix = vec.fit_transform(clean_lyrics)
    feature_names = vec.get_feature_names()

    tfidf_vectors = tfidf_matrix.toarray()

    n = len(clean_lyrics)
    distances = [[0 for x in range(n)] for y in range(n)]


    d_file = open('distances_bigram.txt','a+')

    for i in range(n):
        for j in range(n):
            # distances[i][j] = 1.0 - round(cosine_similarity(tfidf_vectors[i],tfidf_vectors[j]),5)
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

    M, C = kmedoids.kMedoids(A, n, 6)

    for i in range(100):
        t_M, t_C = kmedoids.kMedoids(A, n, 6)
        if (cost(A, t_M, t_C) < cost(A, M, C)):
            M = t_M
            C = t_C


    for i in range(len(C)):
        counts = [0 for j in range(6)]
        for point in C[i]:
            for j in range(len(C)):
                if point < n[j]:
                    counts[j] += 1

        print "Cluster " + str(i+1) + ":"
        print df1['genre'].iloc[0] + ": " + str(counts[0]) + ", " + df2['genre'].iloc[0] + ": " + str(counts[1]) + ", "  \
                  + df3['genre'].iloc[0] + ": " + str(counts[2]) + ", " + df4['genre'].iloc[0] + ": " + str(counts[3]) + ", "  \
              + df5['genre'].iloc[0] + ": " + str(counts[4]) + ", " + df6['genre'].iloc[0] + ": " + str(counts[5])

        print "========================================================================================================="

    # colors = ["#FFFF00", "#9ACD32", "#32CD32", "#2E8B57", "#20B2AA", "#00FFFF"]
    #
    # dt = [('len', float)]
    # A = np.array(distances)
    #
    # A = A.view(dt)
    #
    # G = nx.from_numpy_matrix(A)
    # G = nx.nx_agraph.to_agraph(G)
    #
    # for i in range(0, n):
    #     node = G.get_node(i)
    #     for j in range(len(C)):
    #         if i < n[j]:
    #             node.attr['color'] = colors[j]
    #
    # G.node_attr.update(style="filled", height="0.2", width = "0.2")
    # G.edge_attr.update(color="none", width="0.02", style="dashed")
    #
    # filename = 'distances_all.png'
    # G.draw('distances_all.png', format='png', prog='neato')
    #
    # for i in range(len(C)):
    #     for j in C[i]:
    #         node = G.get_node(j)
    #         node.attr['color'] = colors[i]
    #
    # G.node_attr.update(style="filled", height="0.2", width = "0.2")
    # G.edge_attr.update(color="none", width="0.02", style="dashed")
    #
    # filename = 'distances_clustered.png'
    # G.draw(filename, format='png', prog='neato')


# vec = CountVectorizer(
#             encoding='utf-8',
#             decode_error='replace',
#             strip_accents='unicode',
#             analyzer='word',
#             binary=False,
#             stop_words=stop_words,
#             tokenizer=porter_tokenizer,
#             ngram_range=(2,2),
#             max_features = 5000
#     )



#bi-gram
# vec = CountVectorizer(
#             encoding='utf-8',
#             decode_error='replace',
#             strip_accents='unicode',
#             analyzer='word',
#             binary=False,
#             stop_words=stop_words,
#             tokenizer=porter_tokenizer,
#             ngram_range=(2,2)
#     )


# dataframes = [df_ch, df_co, df_hr, df_rb, df_p, df_r]
#
# genrecombos = itertools.combinations(dataframes, 2)
#
# i = 0
#
# acc_matrix = [[0 for j in range(len(dataframes))] for i in range(len(dataframes))]
#
# for i in range(len(dataframes)-1):
#     for j in range(i+1,len(dataframes)):
#         acc_matrix[i][j] = pairwiseClustering(dataframes[i],dataframes[j])
#         acc_matrix[j][i] = acc_matrix[i][j]
#
# # print acc_matrix
# dv.drawGraph(acc_matrix)
# dv.plot2d(acc_matrix, genres)
# pairwiseClustering(genrecombos[0][0],genrecombos[0][1])
# for combo in genrecombos:
#     pairwiseClustering(combo[0], combo[1])
#     # if i == 0:
#     #     pairwiseClustering(combo[0], combo[1])
#     # i += 1
#     print "\n"

# overallClustering(df_ch, df_co, df_hr, df_rb, df_p, df_r)
