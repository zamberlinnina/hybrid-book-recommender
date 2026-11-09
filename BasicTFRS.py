# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 15:46:36 2026

@author: zambe
"""

from TFRSBuilders import buildUserModel, buildRatingModel, trainRankingModel, trainRetrievalModel
import tensorflow as tf

def trainBasicTFRS(uniqueUserIDs, uniqueBookIDs, trainSet):
    
    # RETRIEVAL - weeding out all the candidates that the user is not interested in
    # the query tower
    retrievalUserModel = buildUserModel(uniqueUserIDs)
    # the candidate tower
    retrievalBookModel = buildBasicBookModel(uniqueBookIDs)
    
    retrievalModel = trainRetrievalModel(retrievalUserModel, retrievalBookModel, trainSet, hybrid=False)

    # RANKING - selecting the most likely candidates
    rankingUserModel = buildUserModel(uniqueUserIDs)
    rankingBookModel = buildBasicBookModel(uniqueBookIDs)
    ratingModel = buildRatingModel()

    rankingModel = trainRankingModel(rankingUserModel, rankingBookModel, ratingModel, trainSet, hybrid=False)
    
    return retrievalModel, rankingModel

def buildBasicBookModel(uniqueBookIDs, embeddingDimension=32):
    bookModel = tf.keras.Sequential([
        tf.keras.layers.StringLookup(vocabulary=uniqueBookIDs, mask_token=None),
        # extra index for out-of-vocabulary values
        tf.keras.layers.Embedding(len(uniqueBookIDs) + 1, embeddingDimension)
    ])
    
    return bookModel

def buildBasicTFDataset(trainRatings, batchSize=1024, shuffle=True):
    userIDs = []
    bookIDs = []
    ratings = []
    
    for userID, bookID, rating in trainRatings:
        userIDs.append(userID)
        bookIDs.append(bookID)
        ratings.append(rating)
        
    dataset = tf.data.Dataset.from_tensor_slices({
        'user_id': userIDs,
        'book_id': bookIDs,
        'rating': ratings
    })
    
    if (shuffle):
        dataset = dataset.shuffle(buffer_size=100000, seed=1)
        
    dataset = dataset.batch(batch_size=batchSize)
    
    return dataset







