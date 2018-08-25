from nltk.corpus import reuters 
import nltk
from scipy import sparse
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from nltk import download
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
nltk.download("stopwords")
cachedStopWords = stopwords.words("english")
nltk.download("reuters")
import os
import sys
#import numpy
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import numpy as np
from pandas import *
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

nltk.download("punkt")

def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text))
    words = [word for word in words
                  if word not in cachedStopWords]
    tokens =(list(map(lambda token: PorterStemmer().stem(token),
                  words)));
    p = re.compile('[a-zA-Z]+');
    filtered_tokens = list(filter(lambda token:
                  p.match(token) and len(token)>=min_length,
         tokens));
    tokensfile = open("tokenstest.txt", 'w');
    # for token in filtered_tokens :
    #     tokensfile.write(str(token) + "\n")
    # tokensfile.close()
    return filtered_tokens

# function to get unique values
def unique(list1):
 
    # intilize a null list
    unique_list = []
     
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x.lower())
    # print list
    return unique_list

doclist = []
category_docs = []
categories = reuters.categories()[:5] # 5 categories
print(categories)
category_length = len(categories)
print("Category length :",category_length)
# slice_index = 1200//category_length
# print(slice_index)
category_lengths = []
for category in categories :
    category_docs = category_docs + reuters.fileids(category)
test_docs = []
train_docs = []
for doc_id in category_docs:
    if doc_id.startswith("train"):
        train_docs.append(doc_id)
    else:
        test_docs.append(doc_id)
train_docs = train_docs[:300]
test_docs = test_docs[:700]
category_docs = train_docs + test_docs #30 percent training docs and 70 percent test
# print(category_docs)
print("Length of category documents:",len(category_docs))
allwords = []
docnames = []
for doc in category_docs:
    d = tokenize(reuters.raw(doc))
    doclist.append(d)
    allwords += d
    docnames.append(doc)
allwords = unique(allwords)
matrix = []
rows = []
allwords = allwords[:len(doclist)]
for doc in doclist:
    for word in allwords:
        if word in doc:
            rows.append(1)
        else:
            rows.append(0)
    matrix.append(rows)
    rows = []
print("original matrix size:",len(matrix),"by",len(matrix[0]))
f = open("project.txt",'w')
I = docnames
C = allwords[:len(doclist)]
printable_matrix = DataFrame(matrix,index = I, columns = C)

covariance_matrix = []
cov_row = []
length = len(doclist) 
for a in range(0,length) :
    for b in range(0, length) :
        if a == b:
            cov_row.append(1)
        else:
            x = [matrix[a], matrix[b]]
            y = np.cov(x)[1][0]
            cov_row.append(y)
    covariance_matrix.append(cov_row)
    cov_row = []
# test = I[: length]
# printable_cov_matrix = DataFrame(covariance_matrix,index = test, columns = test)
# f2 = open("covariance_matrix.txt", 'w')
# print(printable_cov_matrix, file = f2)


kmeans = KMeans(n_clusters=5) ## according to elbow graph
X = np.matrix(matrix) # data analysis on covariance matrix
# X = np.matrix(matrix) #data analyis on 1's and 0's matrix
X = StandardScaler().fit_transform(X)
# X = sparse.csr_matrix(X)
# pca = PCA(n_components = 2)
pca = PCA(2) #project to two dimensions

principalComponents = pca.fit_transform(X)


#ELBOW algorithm for determining clusters
#
#
# distortions = []
# elbow_X = X
# K = range(1,10)
# for k in K:
#     kmeanModel = KMeans(n_clusters=k).fit(elbow_X)
#     kmeanModel.fit(elbow_X)
#     distortions.append(sum(np.min(cdist(
#         elbow_X, kmeanModel.cluster_centers_, 'euclidean')
#             , axis=1)) / elbow_X.shape[0])
# # Plot the elbow
# plt.plot(K, distortions, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method showing the optimal k')
# plt.show()
# 
# 
# END OF ELBOW ALGORITHM


#kmeans algorithm for clustering data
#
#
#
#
kmeans = KMeans(n_clusters=5, init = 'k-means++', max_iter=300, n_init=1,verbose=0)
kmeans.fit(principalComponents)
y_kmeans = kmeans.predict(principalComponents)

LABEL_COLOR_MAP = {0 : 'r',
                   1 : 'k',
                   2 : 'b',
                   3 : 'g',
                   4 : 'm',
                   5 : 'y',
                   6 : 'c'
                   }
centers = kmeans.cluster_centers_
label_color = [LABEL_COLOR_MAP[l] for l in kmeans.labels_]
print(kmeans.labels_)
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.scatter(centers[:, 0], centers[:, 1], c = 'black', s= 300, marker = 'D');
plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=label_color);
plt.show()
# 
# 
# 
#END OF KMEANS ALGO
#
#
# elbow math set, before k means, how many clusters
# plot loss function after k means, which fit is better, covariance or original