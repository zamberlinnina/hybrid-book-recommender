# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 18:00:21 2026

@author: zambe
"""

import tensorflow as tf
import tensorflow_recommenders as tfrs

class RetrievalModel(tfrs.Model):
    
    def __init__(self, userModel, bookModel, retrievalTask, hybrid=False):
        super().__init__()
        
        self.userModel = userModel
        self.bookModel = bookModel
        self.retrievalTask = retrievalTask
        self.hybrid = hybrid
        
    def compute_loss(self, features, training=False):
        # passing features into the models and getting embeddings back
        userEmbeddings = self.userModel(features['user_id'])
        
        if (self.hybrid):
            positiveBookEmbeddings = self.bookModel(features)
        else:
            positiveBookEmbeddings = self.bookModel(features['book_id'])
        
        # the task computes the loss and the metrics
        return self.retrievalTask(userEmbeddings, positiveBookEmbeddings)
    
class RankingModel(tfrs.Model):
    
    def __init__(self, userModel, bookModel, ratingModel, rankingTask, hybrid=False):
        super().__init__()
        
        self.userModel = userModel
        self.bookModel = bookModel
        self.ratingModel = ratingModel
        self.rankingTask = rankingTask
        self.hybrid = hybrid
        
    def call(self, features):
        userEmbeddings = self.userModel(features['user_id'])
        
        if (self.hybrid):
            bookEmbeddings = self.bookModel(features)
        else:
            bookEmbeddings = self.bookModel(features['book_id'])
        
        # concatenating embeddings for passing them into the neural network
        # rating prediction for a user id and a book id
        combinedEmbeddings = tf.concat([userEmbeddings, bookEmbeddings], axis=1)
        
        return self.ratingModel(combinedEmbeddings)
    
    def compute_loss(self, features, training=False):
        predictions = self(features)
        labels = features['rating']
        
        # comparison of the real and the predicted rating
        return self.rankingTask(labels, predictions)
    
class HybridBookModel(tf.keras.Model):
    
    def __init__(self, uniqueBookIDs, uniqueAuthors, uniqueTags, publicationYears, embeddingDimesnion=32):
        super().__init__()
        
        # book ID feature
        self.bookIDModel = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=uniqueBookIDs, mask_token=None),
            # extra index for out-of-vocabulary values
            tf.keras.layers.Embedding(len(uniqueBookIDs) + 1, embeddingDimesnion)
        ])
    
        # author feature
        self.authorModel = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=uniqueAuthors, mask_token=None),
            # extra index for out-of-vocabulary values
            tf.keras.layers.Embedding(len(uniqueAuthors) + 1, embeddingDimesnion)
        ])
        
        # tags feature
        self.tagModel = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=uniqueTags, mask_token=None),
            # extra index for out-of-vocabulary values
            tf.keras.layers.Embedding(len(uniqueTags) + 1, embeddingDimesnion)
        ])
        
        # publication year feature - numerical
        self.yearNormalization = tf.keras.layers.Normalization()
        # conversion to tensor
        publicationYears = tf.constant(publicationYears, dtype=tf.float32)
        publicationYears = tf.expand_dims(publicationYears, axis=1)
        self.yearNormalization.adapt(publicationYears)
        
        # combining all features into a final embedding
        self.denseLayers = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(embeddingDimesnion)
        ])
        
    def call(self, features):
        bookIDEmbedding = self.bookIDModel(features['book_id'])
        authorEmbedding = self.authorModel(features['authors'])
        # one embedding for each tag - 5
        tagEmbeddings = self.tagModel(features['tags'])
    
        publicationYear = tf.expand_dims(features['publication_year'], axis=1)
        normalizedYear = self.yearNormalization(publicationYear)
        
        # creating one tag embedding out of 5 by averageing them
        tagEmbedding = tf.reduce_mean(tagEmbeddings, axis=1)
        
        combinedFeatures = tf.concat(
            [bookIDEmbedding, authorEmbedding, tagEmbedding, normalizedYear], axis=1
        )
        
        return self.denseLayers(combinedFeatures)
        
        