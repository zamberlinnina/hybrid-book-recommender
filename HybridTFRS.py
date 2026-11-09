# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:01:34 2026

@author: zambe
"""

from TFRSModels import HybridBookModel
from TFRSBuilders import buildUserModel, buildRatingModel, trainRetrievalModel, trainRankingModel
import tensorflow as tf

def trainHybridTFRS(uniqueUserIDs, uniqueBookIDs, metadataByookID, trainSet):
    uniqueAuthorsSet = set()
    uniqueTagsSet = set()
    publicationYears  = []
    
    for author, year, tags in metadataByookID.values():
        uniqueAuthorsSet.add(author)
        publicationYears.append(year)
        
        for tag in tags:
            uniqueTagsSet.add(tag)
            
    uniqueAuthors = list(uniqueAuthorsSet)
    uniqueTags = list(uniqueTagsSet)
    
    # RETRIEVAL - weeding out all the candidates that the user is not interested in
    # the query tower
    retrievalUserModel = buildUserModel(uniqueUserIDs)
    # the candidate tower
    retrievalBookModel = buildHybridBookModel(uniqueBookIDs, uniqueAuthors, uniqueTags, publicationYears)
    
    retrievalModel = trainRetrievalModel(retrievalUserModel, retrievalBookModel, trainSet, hybrid=True)
    
    # RANKING - selecting the most likely candidates
    rankingUserModel = buildUserModel(uniqueUserIDs)
    rankingBookModel = buildHybridBookModel(uniqueBookIDs, uniqueAuthors, uniqueTags, publicationYears)
    ratingModel = buildRatingModel()
    
    rankingModel = trainRankingModel(rankingUserModel, rankingBookModel, ratingModel, trainSet, hybrid=True)
    
    return retrievalModel, rankingModel

def buildHybridBookModel(uniqueBookIDs, uniqueAuthors, uniqueTags, publicationYears, embeddingDimension=32):
    bookModel = HybridBookModel(
        uniqueBookIDs, uniqueAuthors, uniqueTags, publicationYears, embeddingDimension
    )
    
    return bookModel

def buildHybridTFDataset(trainRatings, metadataByBookID, batchSize=1024, shuffle=True):
    userIDs = []
    bookIDs = []
    authors = []
    bookTags = []
    publicationYears = []
    ratings = []
        
    for userID, bookID, rating in trainRatings:
        author, year, tags = metadataByBookID[bookID]
        
        userIDs.append(userID)
        bookIDs.append(bookID)
        authors.append(author)
        bookTags.append(tags)
        publicationYears.append(year)
        ratings.append(rating)
        
    dataset = tf.data.Dataset.from_tensor_slices({
        'user_id': userIDs,
        'book_id': bookIDs,
        'authors': authors,
        'tags': bookTags,
        'publication_year': publicationYears,
        'rating': ratings
    }) 
        
    if (shuffle):
        dataset = dataset.shuffle(buffer_size=100000, seed=1)
        
    dataset = dataset.batch(batch_size=batchSize)
    
    return dataset
        
        
        

