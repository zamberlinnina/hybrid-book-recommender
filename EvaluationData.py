# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 18:00:11 2026

@author: zambe
"""

from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from surprise import KNNBaseline
from collections import defaultdict
import random

class EvaluationData:
    
    def __init__(self, data, allBookIDs):
        
        self.allBookIDs = allBookIDs
        
        # full train set - Novelty, Coverage, Diversity
        self.fullTrainSet = data.build_full_trainset()
        
        # 75/25 train/test split - RMSE, MAE
        self.trainSet, self.testSet = train_test_split(data, test_size=0.25, random_state=1)
        
        # leave-one-out train/test split - HR, CHR, RHR
        loo = LeaveOneOut(n_splits=1, random_state=1)
        for train, test in loo.split(data):
            self.looTrainSet = train
            self.looTestSet = test
            
        # book popularity stats - Novelty
        self.bookPopularity = self.computeBookPopularity(self.fullTrainSet)
        
        # relevant books for each user - sampled candidate evaluation, Precision, Recall, NDCG
        self.relevantBooks = self.computeBookRelevance(self.testSet, minRating=4.0)
        
        # books used in training for each user - sampled candidate evaluation
        self.holdOutTrainBooks = self.computeTrainBooks(self.trainSet)
        self.looTrainBooks = self.computeTrainBooks(self.looTrainSet)
        self.fullTrainBooks = self.computeTrainBooks(self.fullTrainSet)
        
        # book similarity matrix - Diversity
        # will be computed when needed
        self.similarityModel = None
        
    def computeBookPopularity(self, trainSet):
        popularity = defaultdict(int)
        
        for innerUserID in trainSet.all_users():
            for innerBookID, rating in trainSet.ur[innerUserID]:
                rawBookID = trainSet.to_raw_iid(innerBookID)
                popularity[int(rawBookID)] += 1
                
        return popularity
    
    def computeBookRelevance(self, testSet, minRating=4.0):
        relevant = defaultdict(set)
        
        for userID, bookID, realRating in testSet:
            if (realRating >= minRating):
                relevant[int(userID)].add(int(bookID))
                
        return relevant
    
    def computeTrainBooks(self, trainSet):
        trainBooks = defaultdict(set)
        
        for innerUserID in trainSet.all_users():
            rawUserID = int(trainSet.to_raw_uid(innerUserID))
            
            for innerBookID, rating in trainSet.ur[innerUserID]:
                rawBookID = int(trainSet.to_raw_iid(innerBookID))
                trainBooks[rawUserID].add(rawBookID)
                
        return trainBooks
    
    def convertTrainsetToRegularData(self, trainSet):
        ratings = []
        
        for innerUserID in trainSet.all_users():
            rawUserID = int(trainSet.to_raw_uid(innerUserID))
            
            for innerBookID, rating in trainSet.ur[innerUserID]:
                rawBookID = int(trainSet.to_raw_iid(innerBookID))
                
                # Surprise Trainset --> regular data for TFRS
                ratings.append((str(rawUserID), str(rawBookID), float(rating)))
                
        return ratings
             
    def buildHoldOutCandidateSet(self, numOfNegatives=100):
        sampledCandidateTestSet = []
        rng = random.Random(1)
        
        for userID, relevants in self.relevantBooks.items():
            seenBooks = self.holdOutTrainBooks[userID]
            
            # negatives consist of all books that aren't in a user's train set
            # relevant books are removed because they will be added to the candidate pool later as positive candidates
            negativePool = self.allBookIDs - seenBooks - relevants
            
            numOfSamples = min(numOfNegatives, len(negativePool))
            sampledNegatives = rng.sample(list(negativePool), numOfSamples)
            
            candidates = list(relevants) + sampledNegatives
            rng.shuffle(candidates)
            
            # tuple (userID, bookID, realRating) for algo.test()
            for bookID in candidates:
                sampledCandidateTestSet.append((str(userID), str(bookID), self.trainSet.global_mean))
                
        return sampledCandidateTestSet
             
    def buildLOOCandidateSet(self, numOfNegatives=100):
        sampledCandidateTestSet = []
        rng = random.Random(1)
        
        for userID, leftOutBookID, realRating in self.looTestSet:
            userID = int(userID)
            leftOutBookID = int(leftOutBookID)
            
            seenBooks = self.looTrainBooks[userID]
            
            negativePool = self.allBookIDs - seenBooks - {leftOutBookID}
            
            numOfSamples = min(numOfNegatives, len(negativePool))
            sampledNegatives = rng.sample(list(negativePool), numOfSamples)
            
            candidates = [leftOutBookID] + sampledNegatives
            rng.shuffle(candidates)
            
            for bookID in candidates:
                sampledCandidateTestSet.append((str(userID), str(bookID), self.looTrainSet.global_mean))
                
        return sampledCandidateTestSet
    
    def buildUnseenCandidateSet(self, numOfNegatives=100):
        sampledCandidateTestSet = []
        rng = random.Random(1)
        
        for innerUserID in self.fullTrainSet.all_users():
            rawUserID = int(self.fullTrainSet.to_raw_uid(innerUserID))
            
            seenBooks = self.fullTrainBooks[rawUserID]
            
            negativePool = self.allBookIDs - seenBooks
            
            numOfSamples = min(numOfNegatives, len(negativePool))
            sampledNegatives = rng.sample(list(negativePool), numOfSamples)
            
            candidates = sampledNegatives
            rng.shuffle(candidates)
            
            for bookID in candidates:
                sampledCandidateTestSet.append((str(rawUserID), str(bookID), self.fullTrainSet.global_mean))
                
        return sampledCandidateTestSet
    
    def buildUnseenCandidateSetForUser(self, userID):
        candidateSet = []
        
        seenBooks = self.fullTrainBooks[userID]
        unseenBooks = self.allBookIDs - seenBooks
        
        for bookID in unseenBooks:
            candidateSet.append((str(userID), str(bookID), self.fullTrainSet.global_mean))
            
        return candidateSet
                  
    def getTrainSet(self):
        return self.trainSet
    
    def getTestSet(self):
        return self.testSet
    
    def getLOOTrainSet(self):
        return self.looTrainSet
    
    def getLOOTestSet(self):
        return self.looTestSet
    
    def getFullTrainSet(self):
        return self.fullTrainSet
    
    def getBookPopularity(self):
        return self.bookPopularity
    
    def getSimilarities(self):
        if (self.similarityModel is None):
            similarityOptions = {
                'name': 'cosine',
                'user_based': False
            }
            self.similarityModel = KNNBaseline(sim_options=similarityOptions)
            self.similarityModel.fit(self.fullTrainSet)
            
        return self.similarityModel
    
    def getRelevantBooks(self):
        return self.relevantBooks
    
    def getHoldOutTrainBooks(self):
        return self.holdOutTrainBooks
    
    def getLOOTrainBooks(self):
        return self.looTrainBooks
    
    def getFullTrainBooks(self):
        return self.fullTrainBooks
    
    def getNumberOfBooks(self):
        return len(self.allBookIDs)
    
    def getNumberOfRatings(self):
        return self.fullTrainSet.n_ratings
    

    