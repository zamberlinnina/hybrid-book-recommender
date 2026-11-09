# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 13:38:23 2026

@author: zambe
"""

import tensorflow as tf
import tensorflow_recommenders as tfrs
from TFRSModels import RankingModel, RetrievalModel

def buildUserModel(uniqueUserIDs, embeddingDimension=32): 
    userModel = tf.keras.Sequential([
        tf.keras.layers.StringLookup(vocabulary=uniqueUserIDs, mask_token=None),
        # extra index for out-of-vocabulary values
        tf.keras.layers.Embedding(len(uniqueUserIDs) + 1, embeddingDimension)
    ])
    
    return userModel

def buildRatingModel():
    # neural network for rating prediction
    ratingModel = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    return ratingModel

def trainRetrievalModel(userModel, bookModel, trainSet, hybrid=False):

    # training the model so that real interactions get higher affinity scores - lowering the loss
    retrievalTask = tfrs.tasks.Retrieval()

    retrievalModel = RetrievalModel(userModel, bookModel, retrievalTask, hybrid=hybrid)

    # controls the adjustment of weights through backpropagation
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.1)
    retrievalModel.compile(optimizer=optimizer)
    # fitting the model
    retrievalModel.fit(trainSet, epochs=3)
    
    return retrievalModel

def trainRankingModel(userModel, bookModel, ratingModel, trainSet, hybrid=False):
    
    loss = tf.keras.losses.MeanSquaredError()
    metrics = [tf.keras.metrics.RootMeanSquaredError()]
    rankingTask = tfrs.tasks.Ranking(loss=loss, metrics=metrics)
    
    rankingModel = RankingModel(userModel, bookModel, ratingModel, rankingTask, hybrid=hybrid)
    
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.1)
    rankingModel.compile(optimizer=optimizer)
    rankingModel.fit(trainSet, epochs=3)
    
    return rankingModel