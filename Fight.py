# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 23:27:05 2026

@author: zambe
"""

from DataLoader import DataLoader
from Evaluator import Evaluator
from TFRSAlgorithm import TFRSAlgorithm
from surprise import SVD, SVDpp
from surprise import NormalPredictor

dl = DataLoader()
data = dl.loadBookRatings()
allBookIDs = dl.getAllBookIDs()
ratingsDF = dl.loadRatingsForTFRS()
uniqueUserIDs = dl.getUniqueUserIDs(ratingsDF)
uniqueBookIDs = dl.getUniqueBookIDs(ratingsDF)

bookTags = dl.loadBookTags()
metadata = dl.loadBookMetadata(bookTags)
metadataByBookID = dl.getMetadataByBookID(metadata)

evaluator = Evaluator(data, allBookIDs, dl)

# SVD
#svd = SVD()
#evaluator.addAlgorithm(svd, 'SVD')

# BASIC TFRS 
#basicTFRS = TFRSAlgorithm(uniqueUserIDs, uniqueBookIDs, hybrid=False)
#evaluator.addAlgorithm(basicTFRS, 'Basic TFRS')

# HYBRID TFRS
hybridTFRS = TFRSAlgorithm(uniqueUserIDs, uniqueBookIDs, hybrid=True, metadataByBookID=metadataByBookID)
evaluator.addAlgorithm(hybridTFRS, 'Hybrid TFRS')

# SVD++
#svdpp = SVDpp()
#evaluator.addAlgorithm(svdpp, 'SVD++')

# random recs - naive baseline
#random = NormalPredictor()
#evaluator.addAlgorithm(random, 'Random')

# fight!
evaluator.printResults(topNMetrics=True)


