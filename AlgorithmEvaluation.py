# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 23:14:34 2026

@author: zambe
"""

from EvaluationMetrics import EvaluationMetrics
from collections import defaultdict

class AlgorithmEvaluation:
    
    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name
        
    def Evaluate(self, evaluationData, topNMetrics, n=10):
        metrics = {}
        
        print('Evaluating accuracy of rating predictions (RMSE, MAE)...')
        # training model on 75% of ratings 
        self.algorithm.fit(evaluationData.getTrainSet())
        predictions = self.algorithm.test(evaluationData.getTestSet())
        metrics['MAE'] = EvaluationMetrics.MAE(predictions)
        metrics['RMSE'] = EvaluationMetrics.RMSE(predictions)
        
        if (topNMetrics):
            # hold out top N evaluation
            print('Evaluating top N on a sampled candidate set...')
            holdOutCandidates = evaluationData.buildHoldOutCandidateSet()
            
            if (hasattr(self.algorithm, 'recommend')):
                holdOutTopN = self.getTFRSTopNRecs(holdOutCandidates, n)
            else:
                holdOutPredictions = self.algorithm.test(holdOutCandidates)
                holdOutTopN = self.getTopNRecs(holdOutPredictions, n)
                
            print('Analyzing Precision, Recall and NDCG...')
            relevantBooks = evaluationData.getRelevantBooks()
            metrics['Precision'] = EvaluationMetrics.Precision(holdOutTopN, relevantBooks)
            metrics['Recall'] = EvaluationMetrics.Recall(holdOutTopN, relevantBooks)
            metrics['NDCG'] = EvaluationMetrics.NDCG(holdOutTopN, relevantBooks)
            
            # leave-one-out top N evaluation
            print('Evaluating top N with leave-one-out...')
            self.algorithm.fit(evaluationData.getLOOTrainSet())
            looCandidates = evaluationData.buildLOOCandidateSet()
            
            if (hasattr(self.algorithm, 'recommend')):
                looTopN = self.getTFRSTopNRecs(looCandidates, n)
            else:
                looPredictions = self.algorithm.test(looCandidates)
                looTopN = self.getTopNRecs(looPredictions, n)
                
            leftOut = evaluationData.getLOOTestSet()
            print('Analyzing Hit Rate, Cumulative Hit Rate and Rating Hit Rate...')
            metrics['HR'] = EvaluationMetrics.HitRate(looTopN, leftOut)
            metrics['CHR'] = EvaluationMetrics.CumulativeHitRate(looTopN, leftOut)
            metrics['RHR'] = EvaluationMetrics.RatingHitRate(looTopN, leftOut)
            
            # full training set top N evaluation
            print('Evaluating top N on a full training set...')
            self.algorithm.fit(evaluationData.getFullTrainSet())
            candidates = evaluationData.buildUnseenCandidateSet()
            
            if (hasattr(self.algorithm, 'recommend')):
                topN = self.getTFRSTopNRecs(candidates, n)
            else:
                predictions = self.algorithm.test(candidates)
                topN = self.getTopNRecs(predictions, n)
            print('Analyzing Catalog Coverage, Diversity and Novelty...')
            metrics['CCoverage'] = EvaluationMetrics.CatalogCoverage(topN, evaluationData.getNumberOfBooks())
            metrics['Diversity'] = EvaluationMetrics.Diversity(topN, evaluationData.getSimilarities())
            metrics['Novelty'] = EvaluationMetrics.Novelty(topN, evaluationData.getBookPopularity(), evaluationData.getNumberOfRatings())
            
            print('Analysis complete')
            
        return metrics
    
    def getTopNRecs(self, predictions, n=10):
        topN = defaultdict(list) # tuple (bookID, estimated rating) for each user
        
        for userID, bookID, realRating, estRating, details in predictions:
            topN[int(userID)].append((int(bookID), estRating))
            
        for userID, estimations in topN.items():
            estimations.sort(key=lambda x: x[1], reverse=True) # sort by estimated rating
            topN[int(userID)] = estimations[:n]
            
        return topN
    
    def getTFRSTopNRecs(self, candidateSet, n=10, retrievalK=50):
        candidates = {}
        topN = {}
        
        for userID, bookID, dummyRating in candidateSet:
            userID = int(userID)
            bookID = int(bookID)
            
            if (userID not in candidates):
                candidates[userID] = []
                
            candidates[userID].append(bookID)
            
        for userID, candidateBookIDs in candidates.items():
            recommendations = self.algorithm.recommend(
                userID, candidateBookIDs, retrievalK, n
            )
            
            topN[userID] = recommendations
            
        return topN
            
    def getSampledUserTopN(self, evaluationData, goodbooks, sampledUserID=34, k=10):
        recs = []
        
        print('\nComputing recommendations...')
        # taking unseen books for the user
        candidates = evaluationData.buildUnseenCandidateSetForUser(sampledUserID)
        
        if (hasattr(self.algorithm, 'recommend')):
            candidateBookIDs = []
            
            for userID, bookID, dummyRating in candidates:
                candidateBookIDs.append(int(bookID))
                
            recs = self.algorithm.recommend(
                sampledUserID, candidateBookIDs, retrievalK=100, n=k
            )   
        else:    
            # estimating ratings
            predictions = self.algorithm.test(candidates)
        
            for userID, bookID, realRating, estRating, details in predictions:
                recs.append((int(bookID), estRating))
            
            recs.sort(key=lambda x: x[1], reverse=True)
        
        print('\nRecommendations for user', sampledUserID, 'using', self.name, ':')
        for bookID, estRating in recs[:k]:
            print(goodbooks.getBookTitle(bookID), estRating)
    
    def getAlgorithm(self):
        return self.algorithm
    
    def getAlgorithmName(self):
        return self.name
            
            
            