# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 17:03:35 2026

@author: zambe
"""

from BasicTFRS import buildBasicTFDataset, trainBasicTFRS
from HybridTFRS import buildHybridTFDataset, trainHybridTFRS
import tensorflow as tf
from surprise import Prediction

class TFRSAlgorithm:
    
    def __init__(self, uniqueUserIDs, uniqueBookIDs, hybrid=False, metadataByBookID=None):
        self.uniqueBookIDs = uniqueBookIDs
        self.uniqueUserIDs = uniqueUserIDs
        
        self.hybrid = hybrid
        self.metadataByBookID = metadataByBookID
        
        self.retrievalModel = None
        self.retrievalIndex = None
        self.rankingModel = None
        
    def fit(self, trainSet):  
        # conversion of Surprise trainset to regular ratings
        ratings = []
        
        for innerUserID in trainSet.all_users():
            rawUserID = int(trainSet.to_raw_uid(innerUserID))
            
            for innerBookID, rating in trainSet.ur[innerUserID]:
                rawBookID = int(trainSet.to_raw_iid(innerBookID))
                
                # Surprise Trainset --> regular data for TFRS
                ratings.append((str(rawUserID), str(rawBookID), float(rating)))
                
        # conversion of ratings to a TensorFlow data frame
        # training retrieval + ranking
        if (self.hybrid):
            tfTrainSet = buildHybridTFDataset(ratings, self.metadataByBookID)
            
            self.retrievalModel, self.rankingModel = trainHybridTFRS(
                self.uniqueUserIDs, self.uniqueBookIDs, self.metadataByBookID, tfTrainSet
            )     
        else:
            tfTrainSet = buildBasicTFDataset(ratings)
        
            self.retrievalModel, self.rankingModel = trainBasicTFRS(
                self.uniqueUserIDs, self.uniqueBookIDs, tfTrainSet
            )
        
        return self
    
    def test(self, testSet, batchSize=4096):
        predictions = []
        
        for start in range(0, len(testSet), batchSize):
            
            batch = testSet[start:start+batchSize]
            
            userIDs = []
            bookIDs = []
            for userID, bookID, realRating in batch:
                userIDs.append(str(userID))
                bookIDs.append(str(bookID))
            
            # expected input for the call function of the ranking model
            if (self.hybrid):
                inputs = self.getHybridBookFeatures(bookIDs)
                inputs['user_id'] = tf.constant(userIDs)
            else:
                inputs = {
                    'user_id': tf.constant(userIDs),
                    'book_id': tf.constant(bookIDs)
                }
            
            estRatings = self.rankingModel(inputs, training=False)
            estRatings = estRatings.numpy().flatten()
            
            for (userID, bookID, realRating), estRating in zip(batch, estRatings):
                # constraining the estimated rating
                estRating = float(max(1.0, min(5.0, estRating)))
                
                predictions.append(
                    Prediction(userID, bookID, realRating, estRating, details={})
                )
            
        return predictions
    
    def recommend(self, sampledUserID, candidateBookIDs, retrievalK=50, n=10):
        recommendations = []
        
        # conversion of input IDs to a tensor
        userID = tf.constant([str(sampledUserID)])
        bookIDsList = []
        for bookID in candidateBookIDs:
            bookIDsList.append(str(bookID))
        bookIDs = tf.constant(bookIDsList)
            
        if (len(candidateBookIDs) == 0):
            return []
        
        # RETRIEVAL
        userEmbedding = self.retrievalModel.userModel(userID)
        
        if (self.hybrid):
            candidateFeatures = self.getHybridBookFeatures(bookIDsList)
            bookEmbeddings = self.retrievalModel.bookModel(candidateFeatures)
        else:
            bookEmbeddings = self.retrievalModel.bookModel(bookIDs)
        
        # calculation of retrieval scores for all candidates
        # computing the dot product of a user (vector) with books (matrix)
        userEmbedding = tf.squeeze(userEmbedding, axis=0)
        retrievalScores = tf.linalg.matvec(bookEmbeddings, userEmbedding)
        
        # selecting top K candidates
        k = min(retrievalK, len(candidateBookIDs))
        scores, indices = tf.math.top_k(retrievalScores, k=k)
        retrievedBookIDs = tf.gather(bookIDs, indices)
        
        # RANKING
        if (self.hybrid):
            retrievedAuthors = tf.gather(candidateFeatures['authors'], indices)
            retrievedTags = tf.gather(candidateFeatures['tags'], indices)
            retrievedYears = tf.gather(candidateFeatures['publication_year'], indices)
            rankingInputs = {
                'user_id': tf.repeat(userID, repeats=k),
                'book_id': retrievedBookIDs,
                'authors': retrievedAuthors,
                'tags': retrievedTags,
                'publication_year': retrievedYears
            }
        else:
            rankingInputs = {
                'user_id': tf.repeat(userID, repeats=k),
                'book_id': retrievedBookIDs
            }
        
        estRatings = self.rankingModel(rankingInputs, training=False)
        estRatings = estRatings.numpy().flatten()
        
        for bookID, estRating in zip(retrievedBookIDs.numpy(), estRatings):
            bookID = int(bookID.decode('utf-8'))
            estRating = float(max(1.0, min(5.0, estRating)))
            
            recommendations.append((bookID, estRating))
            
        # sorting in descending order by estimated rating
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        topN = recommendations[:n]
        
        return topN
    
    def getHybridBookFeatures(self, bookIDs):
        bookIDsList = []
        authors = []
        bookTags = []
        publicationYears = []
        
        for bookID in bookIDs:
            bookID = str(bookID)
            author, year, tags = self.metadataByBookID[bookID]
            
            bookIDsList.append(bookID)
            authors.append(author)
            bookTags.append(tags)
            publicationYears.append(year)
            
        features = {
           'book_id': tf.constant(bookIDsList),
           'authors': tf.constant(authors),
           'tags': tf.constant(bookTags),
           'publication_year': tf.constant(publicationYears, dtype=tf.float32)
        }
        
        return features
        
        
        
        
        
        