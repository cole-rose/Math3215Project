from nltk.corpus import reuters 
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from nltk import download
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
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download("punkt")

def collection_stats():
    # List of documents
    documents = reuters.fileids()
    print(str(len(documents)) + " documents");
 
    train_docs = list(filter(lambda doc: doc.startswith("train"),
                        documents));
    print(str(len(train_docs)) + " total train documents");
 
    test_docs = list(filter(lambda doc: doc.startswith("test"),
                       documents));
    print(str(len(test_docs)) + " total test documents");
 
    # List of categories
    categories = reuters.categories();
    print(str(len(categories)) + " categories");
 
    # Documents in a category
    category_docs = reuters.fileids("acq");
    print("size of category_docs: " + str(len(category_docs)))
 
    # Words for a document
    document_id = category_docs[1]
    document_words = reuters.words(category_docs[1]);
    # for word in document_words:
    #     print(word)
    # print(document_words);  
 
    # # Raw document
    # print(reuters.raw(document_id));
    print(document_id)

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
def feature_values(doc, representer):
    doc_representation = representer.transform([doc])
    features = representer.get_feature_names()
    return [(features[index], doc_representation[0, index])
                 for index in doc_representation.nonzero()[1]]
# Return the representer, without transforming
def tf_idf(docs):
    tfidf = TfidfVectorizer(tokenizer=tokenize, min_df=3,
                        max_df=0.90, max_features=3000,
                        use_idf=True, sublinear_tf=True,
                        norm='l2');
    tfidf.fit(docs);
    return tfidf

def main():
    train_docs = []
    test_docs = []
 
    for doc_id in reuters.fileids():
        if doc_id.startswith("train"):
            train_docs.append(reuters.raw(doc_id))
        else:
            test_docs.append(reuters.raw(doc_id))
 
    representer = tf_idf(train_docs);
    f = open("main_test.txt", "w")

    for doc in test_docs:
        f.write(str(feature_values(doc, representer)))
    
        #print(feature_values(doc, representer))
    f.close()

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
category_docs = reuters.fileids("acq");
allwords = []
docnames = []
for doc in category_docs:
    d = tokenize(reuters.raw(doc))
    doclist.append(d)
    allwords += d
    docnames.append(doc)
allwords = unique(allwords)
f = open("rowandcol.txt", 'w')
print("rownames:", docnames, file = f)
print("\n" + "colnames:",sorted(allwords), file = f)
f.close()
matrix = []
cols = []
for doc in doclist:
    for word in allwords:
        if word in doc:
            cols.append(1)
        else:
            cols.append(0)
    matrix.append(cols)
    cols = []
f = open("project.txt",'w')
I = docnames
C = allwords
printable_matrix = DataFrame(matrix,index = I, columns = C)
#print(printable_matrix, file = f)
row0 = matrix[0]
row1 = matrix[1]
x = [row0,row1]
y = np.cov(x)[1][0]
print(y)

covariance_matrix = []
cov_row = []
for a in range(0,len(doclist) -2300) :
    for b in range(0, len(doclist) - 2300) :
        if a == b:
            cov_row.append(1)
        else:
            x = [matrix[a], matrix[b]]
            y = np.cov(x)[1][0]
            cov_row.append(y)
    covariance_matrix.append(cov_row)
    cov_row = []
test = I[: (len(I) - 2300)]
printable_cov_matrix = DataFrame(covariance_matrix,index = test, columns = test)
f2 = open("covariance_matrix.txt", 'w')
print(printable_cov_matrix, file = f2)