# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 23:44:52 2018

@author: Hufsa
"""

import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify import ClassifierI
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier
from statistics import mode


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers): # *args
        self._classifiers = classifiers
        
    def classify(self, features): 
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    def confidence(self, features): 
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = float(choice_votes/len(votes))
        return conf
    
         
        

# mnb - used for many categories
# GNB - continuous features follow a normal distribution.
# BNB - your features are binary.

 # append tuple of file id and category (neg/pos)
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents) # b/c we kow first 1000 are neg reviews, we shuffle

all_words = []

# put every word in corpus into a list to count freq dist
for w in movie_reviews.words():
    all_words.append(w.lower())
    
all_words = nltk.FreqDist(all_words) # creates key val pairings of words and freq
#print(all_words.most_common(15)) # top 15 most common wods
#print(all_words["stupid"]) # returns a list

#word_features = list(all_words.keys())[:3000] # grab up to the 3000th most common word.
#word_features ={k:all_words[k] for k in sorted(all_words.items(),reverse=True)[:3]}

word_features = [k[0] for k in sorted(all_words.items(),reverse=True, key=lambda x:x[1])[:3000]]

def find_features(doc):
    words = set(doc) # set puts all unique elements in a set (not obj)
    features = {}
    for w in word_features:
        features[w] = (w in words) # is one of 3000 in the movie reviews?
    return features
#print(find_features(movie_reviews.words('neg/cv000_29416.txt')))

# do for all reviews in documents
featureset = [(find_features(rev),category) for (rev, category) in documents]

# test only pos reviews
#trainset = featureset[:1900]
#testset = featureset[1900:]

# test only neg reviews
trainset = featureset[100:]
testset = featureset[:100]
# removing original NB training pickle to try MNB

#with open("pickle_NB_reviews.pickle","wb") as f:
#    pickle.dump(classifier, f)
# f closes after wuth clause
 
#with open("pickle_NB_reviews.pickle","rb") as f:
#    classifier = pickle.load(f)    

# using naive bayes to see if words fall in neg/pos reviews based on how likely they are to occur within most_common?
classifier = nltk.NaiveBayesClassifier.train(trainset)
print("Original Naive Bayes Accuracy %:", (nltk.classify.accuracy(classifier, testset))*100) 
#classifier.show_most_informative_features(15) # most weighted features
#
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(trainset)
#print("MNB Naive Bayes Accuracy %:", (nltk.classify.accuracy(MNB_classifier, testset))*100) 

#G_classifier = SklearnClassifier(GaussianNB())
#G_classifier.train(trainset)
#print("GaussianNB Naive Bayes Accuracy %:", (nltk.classify.accuracy(G_classifier, testset))*100) 

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(trainset)
print("BernoulliNB Naive Bayes Accuracy %:", (nltk.classify.accuracy(BNB_classifier, testset))*100) 

RFC_classifier = SklearnClassifier(RandomForestClassifier(n_estimators=10, max_depth=None, random_state=3))
RFC_classifier.train(trainset)
print("RFC classifier Accuracy %:", (nltk.classify.accuracy(RFC_classifier, testset))*100) 
#
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(trainset)
print("LogisticRegression_classifier Accuracy %:", (nltk.classify.accuracy(BNB_classifier, testset))*100) 

SVC_class = SklearnClassifier(SVC())
SVC_class.train(trainset)
print("SVC Accuracy %:", (nltk.classify.accuracy(SVC_class, testset))*100) 

# disabling so we have an odd num of classifiers for voteclassifier
#SGDClassifier_class = SklearnClassifier(SGDClassifier())
#SGDClassifier_class.train(trainset)
#print("SGDClassifier Accuracy %:", (nltk.classify.accuracy(SGDClassifier_class , testset))*100) 

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(trainset)
#print("LinearSVC Accuracy %:", (nltk.classify.accuracy(LinearSVC_classifier, testset))*100) 

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(trainset)
#print("NuSVC Accuracy %:", (nltk.classify.accuracy(NuSVC_classifier, testset))*100) 
    
voted_classifier = VoteClassifier(classifier,
                                  BNB_classifier,
                                  RFC_classifier,
                                  LogisticRegression_classifier,
                                  SVC_class, 
                                  LinearSVC_classifier,
                                  NuSVC_classifier)
print("Voted classifier acuracy: ", (nltk.classify.accuracy(voted_classifier, testset))*100)
print("Classification:", voted_classifier.classify(testset[0][0]), 
      "Confidence %:",voted_classifier.confidence(testset[0][0])*100)
print("Classification:", voted_classifier.classify(testset[1][0]), 
      "Confidence %:",voted_classifier.confidence(testset[1][0])*100)
print("Classification:", voted_classifier.classify(testset[2][0]), 
      "Confidence %:",voted_classifier.confidence(testset[20][0])*100)
print("Classification:", voted_classifier.classify(testset[3][0]), 
      "Confidence %:",voted_classifier.confidence(testset[3][0])*100)


