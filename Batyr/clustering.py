# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 13:15:00 2018

@author: bnuryyev
"""
import numpy as np
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt


# Instantiate Porter stemming algorithm
stemmer = PorterStemmer()


### Custom functions for vectorization

# Stem words
def stemWords(words_list, stemmer):
    """
        Stemming repetitive/similar words
        
        Params:
            list of words (text),
            stemmer algorithm
    """
    
    return [stemmer.stem(word) for word in words_list]


# Tokenize the text
def tokenize(text):
    """
        Tokenization of a text along with stemming
        
        Params:
            text
    """
    
    tokens = nltk.word_tokenize(text)
    stems  = stemWords(tokens, stemmer)
    return stems


### Import the data and process

# Load data
data = pd.read_csv('trump_data.csv', 
                   header=None, 
                   na_values=['.'], 
                   encoding='cp1252')

# Original data in np.array format
data_array = np.array(data)

# Flatten the 2D array to be a list of strings
data_array_flattened = data_array.flatten()

# Ordered array with extra column of tweet ids
# For a separate variable, also add the column (just numerates through)
numbers_column = np.expand_dims(np.array([x for x in range(1,11066)]), axis=1)
data_array_ordered = np.append(data_array, 
                               numbers_column, 
                               axis=1)


### -------- TF-IDF --------
### Term Frequency - Inverse Document Frequency

# Instantiate TF-IDF
vectorizer_tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')

# Tokenize and build vocabulary
# Row - represents a tweet
# Col - represents a token
tfs = vectorizer_tfidf.fit_transform(data_array_flattened)

# Summarize (REMOVE # TO PRINT)
#print("-- TF-IDF Vectorization Summary --")
#print("Vocab: ", vectorizer_tfidf.vocabulary_)
#print("Idf weights: ", vectorizer_tfidf.idf_)


### -------- Nearest Neigbors --------
### Trying to find similar tweets

# Function prints closest nearest neighbor
def printNearestNeighbors(query_tf_idf, tweets, knn_model, k):
    """
        Prints k-nearest neighbors
        
        Params:
            a query tf-idf vector,
            Trump tweets,
            the k-NN model, 
            number of neighbors as hyperparameter
    """
    
    distances, indices = knn_model.kneighbors(query_tf_idf, 
                                              n_neighbors=k+1)
    nearest_neighbors = [tweets[x] for x in indices.flatten()]
    
    for tweet_no in range(len(nearest_neighbors)):
        if tweet_no == 0:
            print("Query tweet: {0}\n".format(nearest_neighbors[tweet_no]))
        else:
            print("{0}: {1}\n".format(tweet_no, nearest_neighbors[tweet_no]))


# Fit Nearest Neighbors model; uses cosine metric and brute force search
model_tf_idf_nn = NearestNeighbors(metric='cosine', algorithm='brute')
model_tf_idf_nn.fit(tfs)

# Test: Pick a random tweet and find its nearest neighbor
tweet_id = np.random.choice(tfs.shape[0])
printNearestNeighbors(tfs[tweet_id], data_array_ordered, model_tf_idf_nn, k=5)


### -------- K-Means Clustering --------
### Trying to get similar tweets (with least variance) together.

# Get clusters along with related tweets
def getClusteredTweets(kmeans_labels, tweets):
    """
        Creates and returns dictionary of clusters and related tweets.
        
        Params:
            labels produced by KMeans,
            Trump tweets            
    """
    clusters_tweets_dict = {}
    
    for i in set(kmeans_labels):
        # Set [:,1] to [:,0] for the actual tweets
        current_cluster_tweets = [tweets[:,1][x] for x in np.where(kmeans_labels == i)[0]]
        clusters_tweets_dict[i] = current_cluster_tweets
    
    return clusters_tweets_dict


# Number of clusters. Can we be tweaked if necessary.
k = 50

# Fit the Kmeans model. CHANGE verbose=0 IF YOU DONT WANT THE DETAILS
kmeans_model = KMeans(n_clusters=k, max_iter=100, n_init=5, verbose=1)
kmeans_model.fit(tfs)

# Plot the distribution of cluster assignments (REMOVE # TO SEE PLOTS)
#plt.hist(kmeans_model.labels_, k)
#plt.show()

# Get the dictionary of clusters and related tweets
clusters_tweets_dict = getClusteredTweets(kmeans_model.labels_, 
                                              data_array_ordered)

# Take a look at random couple clusters (REMOVE # to get clusters & tweets printed)
#cluster_pick = np.random.choice(len(set(kmeans_model.labels_)))
#print("Cluster {0}".format(cluster_pick))
#print(clusters_tweets_dict[cluster_pick])


### -------- Trying to reverse-engineer the cluster themes using tf-idf --------

# Function to get clusters and their themes together in a dictionary
def getClusterThemes(clusters_tweets_dict, tokenizer_func, tweets_flattened):
    cluster_themes_dict = {}
    
    for key in clusters_tweets_dict.keys():
        current_tfidf = TfidfVectorizer(tokenizer=tokenizer_func, stop_words='english')
        current_tfs = current_tfidf.fit_transform(tweets_flattened)
        
        current_tf_idfs = dict(zip(current_tfidf.get_feature_names(), current_tfidf.idf_))
        tf_idfs_tuples = current_tf_idfs.items()
        cluster_themes_dict[key] = sorted(tf_idfs_tuples, key = lambda x : x[1])[:5]
        
    return cluster_themes_dict

# Show the keywords (REMOVE # TO RUN)
#cluster_themes_dict = getClusterThemes(clusters_tweets_dict, tokenize, data_array_flattened)
#print("Cluster 5 keywords: {0}".format([x[0] for x in cluster_themes_dict[5]]))
#print("Cluster 15 keywords: {0}".format([x[0] for x in cluster_themes_dict[15]]))
#print("Cluster 25 keywords: {0}".format([x[0] for x in cluster_themes_dict[25]]))
#print("Cluster 38 keywords: {0}".format([x[0] for x in cluster_themes_dict[38]]))
#print("Cluster 45 keywords: {0}".format([x[0] for x in cluster_themes_dict[45]]))
    

### -------- Visualize using t-SNE method --------
### t-SNE - t-stochastic neighbor embedding method. Dimensionality reduction 
### method which is a lot helpful when visualizing high dimensional data on 
### on a lower dimensional space

# Decompose our tf-idf matrix into lower-dimensional space
tfs_reduced = TruncatedSVD(n_components=k, random_state=0).fit_transform(tfs)

# Use t-SNE to get 2D representation of our 50-dimensional cluster data
tfs_embedded = TSNE(n_components=2, perplexity=60, verbose=2).fit_transform(tfs_reduced)

# Plot vector embeddings according to their colored clusters
fig = plt.figure(figsize = (10,10))
ax = plt.axes()
plt.scatter(tfs_embedded[:, 0], tfs_embedded[:, 1], marker = "x", c = kmeans_model.labels_)
plt.show()
