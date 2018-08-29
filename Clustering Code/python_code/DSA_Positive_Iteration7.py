
# coding: utf-8

# # IT7 Positive >10 words

# Coding was developed based on below sources:
# * Brandonrose.org. (2018). Document Clustering with Python. [online] Available at: http://brandonrose.org/clustering 
# * Scikit-learn.org. (2018). sklearn.metrics.silhouette_score — scikit-learn 0.19.2 documentation. [online] Available at: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
# * Twitterdev.github.io. (2018). Do More with Twitter Data — Do more with Twitter data 0.1 documentation. [online] Available at: https://twitterdev.github.io/do_more_with_twitter_data/clustering-users.html

# In[1]:


import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib as mpl
import mpld3
import sys
import csv
import string


# In[2]:


dsa_data = pd.read_csv('/Users/jessicakuo/Documents/Data/depression-stress-anxiety.csv')#e.g. /Users/rmhiwku/depression-stress-anxiety.csv
dsa_data = dsa_data.dropna() #filter out empty rows; otherwise it will fail later
dsa_data.columns = ['appname', 'rating', 'ratingcount', 'developer','apptype','reviewer','date','reviewer_rating','thumbsup','review']
dsa_data['words'] = dsa_data['review'].str.split()
dsa_data['word_len'] = dsa_data['words'].str.len()
print('Total reviews: ',len(dsa_data))

#filter in only positive reviews >5 words
dsa_positive = dsa_data[(dsa_data.reviewer_rating>3)&(dsa_data.word_len>8)]
print('Total positive reviews >5 words: ', len(dsa_positive))

#filter in only positive reviews >10 words
dsa_positive = dsa_data[(dsa_data.reviewer_rating>3)&(dsa_data.word_len>13)]
print('Total positive reviews >10 words: ', len(dsa_positive))
print()
#dsa_positive.head()


# In[3]:


#examine the data

dsa_positive.info()


# In[4]:


#create lists of filtered positive reviews
pos_apps = dsa_positive['appname'].tolist()
pos_reviews = dsa_positive['review'].tolist()

import random
num_to_select = 4338
list_of_random_items = random.sample(pos_reviews, num_to_select)
#only use a limited amount of positive reviews for analysis
pos_reviews = list_of_random_items


# # Data Cleaning

# In[6]:


# load NLTK's English stopwords as variable called 'stopwords'
stopwords = nltk.corpus.stopwords.words('english')
newStopWords = ['review','text','app',"also","would","stress","anxiety","depression","saying"]
stopwords.extend(newStopWords)
print (stopwords[:-1])


# In[7]:


# load NLTK's SnowballStemmer as variabled 'stemmer'
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")


# In[8]:


# here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]  
    # filter out stop words
    tokens = [w for w in tokens if not w in stopwords]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 2]
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    tokens = [w for w in tokens if not w in stopwords]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 2]
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


# In[9]:


# use extend so it's a big flat list of vocab
totalvocab_stemmed = []
totalvocab_tokenized = []

for i in pos_reviews:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'review_texts', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

print("Number of vocab tokenized and stemmed: ", len(totalvocab_stemmed))
print("Number of vocab tokenized only: ",len(totalvocab_tokenized))


# In[10]:


vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')


# In[11]:


print (vocab_frame.head())


# In[12]:


# Count frequency of words
from collections import Counter
list1=totalvocab_stemmed
counts = Counter(list1)
print(counts)


# # Tfidf Vectorizer

# In[14]:


from sklearn.feature_extraction.text import TfidfVectorizer

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                   min_df=0.01, stop_words=stopwords,
                                   use_idf=True,
                                   tokenizer=tokenize_and_stem, ngram_range=(1,1))

get_ipython().run_line_magic('time', 'tfidf_matrix = tfidf_vectorizer.fit_transform(pos_reviews) #fit the vectorizer to review_texts')

print(tfidf_matrix.shape)


# In[15]:


# Etermining what terms are useful enough to turn into features
terms = tfidf_vectorizer.get_feature_names()
print(len(terms))
terms


# In[16]:


# dist is defined as 1 - the cosine similarity of each document. 
# Cosine similarity is measured against the tf-idf matrix and can be used to generate a measure of similarity 
# between each document and the other documents in the corpus (each review among the reviews). 
# Subtracting it from 1 provides cosine distance which is used for plotting on a euclidean (2-dimensional) plane.
from sklearn.metrics.pairwise import cosine_similarity
get_ipython().run_line_magic('time', 'dist = 1 - cosine_similarity(tfidf_matrix)')


# # K-means clustering

# In[17]:


import pandas
import pylab as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ### Elbow Curve - Finding the best parameter 

# In[18]:


get_ipython().run_cell_magic('time', '', "\nimport logging\nfrom sklearn.metrics import silhouette_score\nseed = 42\n\n# compare a broad range of ks to start\nks = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]\n\n\n# track a couple of metrics\nsil_scores = []\ninertias = []\n\n# fit the models, save the evaluation metrics from each run\nfor k in ks:\n    logging.warning('fitting model for {} clusters'.format(k))\n    model = KMeans(n_clusters=k, n_jobs=-1, random_state=seed)\n    model.fit(tfidf_matrix)\n    labels = model.labels_\n    sil_scores.append(silhouette_score(tfidf_matrix, labels))\n    inertias.append(model.inertia_)\n\n# plot the quality metrics for inspection\nfig, ax = plt.subplots(2, 1, sharex=True)\n\nplt.subplot(211)\nplt.plot(ks, inertias, 'o--')\nplt.ylabel('inertia')\nplt.title('kmeans parameter search')\n\nplt.subplot(212)\nplt.plot(ks, sil_scores, 'o--')\nplt.ylabel('silhouette score')\nplt.xlabel('k');")


# In[20]:


# Generate average silouette score
X = tfidf_matrix

range_n_clusters = [2, 3, 4, 5, 6 , 7, 8, 9 ,10, 11, 12, 13, 14, 15]

for n_clusters in range_n_clusters:

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    km = KMeans(n_clusters=n_clusters, random_state=1).fit(tfidf_matrix)
    labels = km.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)


# ## (11 clusters) 

# In[35]:


# Do KMeans clustering to get the clusters
#from sklearn.cluster import KMeans

num_clusters = 11

km = KMeans(n_clusters=num_clusters, random_state=1)

get_ipython().run_line_magic('time', 'km.fit(tfidf_matrix)')

clusters = km.labels_.tolist()


# In[36]:


# Use joblib.dump to pickle the model, once it has converged and to reload the model/reassign the labels as the clusters
from sklearn.externals import joblib

#uncomment the below to save your model 
#since I've already run my model I am loading from the pickle
joblib.dump(km,  'doc_cluster7-1.pkl')

km = joblib.load('doc_cluster7-1.pkl')
clusters = km.labels_.tolist()


# In[37]:


# Create dictionary
appreview = { 'review': pos_reviews, 'cluster': clusters}
frame = pd.DataFrame(appreview, index = [clusters] , columns = ['review','cluster'])


# In[38]:


frame.to_csv("IT7-1Sample.csv")


# In[39]:


#number of reviews per cluster (clusters from 0 to 10)
frame['cluster'].value_counts()


# In[40]:


# indexing and sorting on each cluster to identify which are the top n (I used n=50) words that are nearest 
# to the cluster centroid. This gives a good sense of the main topic of the cluster.
from __future__ import print_function

print("Top terms per cluster:")
print()
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    
    for ind in order_centroids[i, :50]: #replace with n words per cluster
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0], end=',')

    print() #add whitespace
    print() #add whitespace

print()


# In[ ]:


#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, review=pos_reviews)) 

#group by cluster
groups = df.groupby('label')

#for i in range(len(df)):
    #original_review = text(df.ix[i]['x'], df.ix[i]['y'], 
            #df.ix[i]['review'], 
            #size=5)  
            

df.ix[305]['review']


# # Multidimensional scaling

# In[27]:


# convert the dist matrix into a 2-dimensional array using multidimensional scaling (MDS)
import os  # for os.path.basename
from sklearn.manifold import MDS

MDS()

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

get_ipython().run_line_magic('time', 'pos = mds.fit_transform(dist)  # shape (n_components, n_samples)')

xs, ys = pos[:, 0], pos[:, 1]
print()


# In[28]:


#save nparray
a = pos
np.savetxt("IT7-1POSnparray.csv", a, delimiter=",")


# # Visualizing document clusters

# In[32]:


#set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e',
                  5: '#5B1B9E', 6: '#9E1B7E', 7: '#F4EB10', 8: '#740437', 9: '#109C19',
                  10: '#F2602F', 11: '', 12: '#8ECE92', 13: '#E79FC0', 14: '#74045E',
                  15: '#EC86DD', 16: '#508DF4', 17: '', 18: '#C94E86', 19: '#5C45A5',
                  20: '#F0D0EB', 21: '', 22: '#E7E59F', 23: '#969E1B', 24: '#1AAEC3'}

#set up cluster names using a dict
cluster_names = {0: 'cluster1', 
                 1: 'cluster2', 
                 2: 'cluster3',
                 3: 'cluster4',
                 4: 'cluster5', 
                 5: 'cluster6', 
                 6: 'cluster7',
                 7: 'cluster8',
                 8: 'cluster9', 
                 9: 'cluster10',
                 10: 'cluster11'}


# ### First, define some dictionaries for going from cluster number to color and to cluster name. Then, based the cluster names off the words that were closest to each cluster centroid.

# In[44]:


#some ipython magic to show the matplotlib plots inline
get_ipython().run_line_magic('matplotlib', 'inline')

#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, review=pos_reviews)) 

#group by cluster
groups = df.groupby('label')


#set up plot
fig, ax = plt.subplots(figsize=(35, 18)) # set size
ax.margins(0.1) # add padding

#iterate through groups to layer the plot
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
    
ax.legend(numpoints=1)  #show legend with only 1 point
plt.show() #show the plot

#uncomment the below to save the plot if need be
#plt.savefig('clusters.png', dpi=200)


# In[ ]:


plt.close()
