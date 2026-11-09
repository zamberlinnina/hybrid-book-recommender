# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 12:07:31 2026

@author: zambe
"""

from surprise import accuracy
from collections import defaultdict
import math
import itertools

class EvaluationMetrics:
    
    # Mean Absolute Error
    def MAE(predictions):
        return accuracy.mae(predictions, verbose=False)
    
    # Root Mean Squared Error
    def RMSE(predictions):
        return accuracy.rmse(predictions, verbose=False)
    
    def HitRate(topN, leftOut):
        hits = 0
        
        for lo in leftOut:
            userID = lo[0]
            leftOutBookID = lo[1]
            
            hit = False
            for bookID, estRating in topN[int(userID)]:
                if (int(leftOutBookID) == int(bookID)):
                    hit = True
                    break  
                
            if (hit):
                hits += 1
                    
        hitRate = hits / len(leftOut)
        return hitRate
    
    def CumulativeHitRate(topN, leftOut, ratingCutoff=4.0):
        hits = 0
        total = 0
        
        for userID, leftOutBookID, realRating in leftOut:    
            if (realRating >= ratingCutoff):
                total += 1
                hit = False
                for bookID, estRating in topN[int(userID)]:
                    if (int(leftOutBookID) == int(bookID)):
                        hit = True
                        break
                    
                if (hit):
                    hits += 1
                        
        cumulativeHitRate = hits / total
        return cumulativeHitRate   

    def RatingHitRate(topN, leftOut):
        hits = defaultdict(float)
        total = defaultdict(float)
        ratingHitRate = {}
        
        for userID, leftOutMovieID, realRating in leftOut:
            hit = False
            for bookID, estRating in topN[int(userID)]:
                if (int(leftOutMovieID) == int(bookID)):
                    hit = True
                    break
            if (hit):
                hits[realRating] += 1
                
            total[realRating] += 1
            
        for rating in sorted(total.keys()):
            ratingHitRate[rating] = hits[rating] / total[rating]
            
        return ratingHitRate
    
    # How much of what I recommended is relevant to the user?
    def Precision(topN, relevantBooks):
        totalPrecision = 0
        users = 0
        
        for userID, recommendations in topN.items():
            hits = 0
            
            for bookID, estRating in recommendations: 
                if (bookID in relevantBooks[int(userID)]):
                    hits += 1
                    
            userPrecision = hits / len(recommendations)
            totalPrecision += userPrecision
            
            users += 1
         
        # average precision over every user
        precision = totalPrecision / users                 
        return precision
    
    # How many relevant (to the user) books did I recommend?
    def Recall(topN, relevantBooks):
        totalRecall = 0
        users = 0
        
        for userID, relevants in relevantBooks.items():
            hits = 0
            
            for relevantBookID in relevants:
                for bookID, estRating in topN[int(userID)]:
                    if (int(relevantBookID) == int(bookID)):
                        hits += 1
                        break
                    
            userRecall = hits / len(relevants)
            totalRecall += userRecall
            
            users += 1
            
        # average recall over every user
        recall = totalRecall / users
        return recall
    
    # Normalized Discounted Cumulative Gain 
    # contribution of a hit is dependant on its position in the topN list
    def NDCG(topN, relevantBooks):
        totalNDCG = 0
        users = 0
        
        for userID, recommendations in topN.items():
            dcg = 0
            
            for rank, (bookID, estRating) in enumerate(recommendations, start=1):
                if (bookID in relevantBooks[int(userID)]):
                    relevance = 1
                else:
                    relevance = 0
                    
                dcg += relevance / math.log2(rank + 1)
                
            # number of 1s in an ideal list
            numOfRelevantBooks = min(
                len(relevantBooks[int(userID)]),
                len(recommendations)
            )
            
            # ideal discounted cumulative gain - all relevant books are sorted on the top 
            idcg = 0     
            for i in range(numOfRelevantBooks):
                idcg += 1 / math.log2(i + 2)
                
            if (idcg > 0):
                userNDCG = dcg / idcg
                totalNDCG += userNDCG
                
            users += 1
            
        # average NDCG over every user
        ndcg = totalNDCG / users
        return ndcg
    
    def CatalogCoverage(topN, numOfBooks):
        recommendedBooks = set()
        
        for userID, recommendations in topN.items():
            for bookID, estRating in recommendations:
                recommendedBooks.add(int(bookID))
                
        coverage = len(recommendedBooks) / numOfBooks
        return coverage
    
    def Novelty(topN, bookPopularity, numOfRatings):
        totalNovelty = 0
        numOfRecommendations = 0
        
        for userID, recommendations in topN.items():
            for bookID, estRating in recommendations:
                popularity = bookPopularity[int(bookID)]
                p = popularity / numOfRatings
                totalNovelty += -math.log2(p)
                
                numOfRecommendations += 1
                
        novelty = totalNovelty / numOfRecommendations
        return novelty
    
    def Diversity(topN, similarityModel):
        numOfPairs = 0
        totalSimilarity = 0
        similarityMatrix = similarityModel.sim
        
        for userID in topN.keys():
            pairs = itertools.combinations(topN[int(userID)], 2)
            
            for (bookID1, estRating1), (bookID2, estRating2) in pairs:
                innerBookID1 = similarityModel.trainset.to_inner_iid(str(bookID1))
                innerBookID2 = similarityModel.trainset.to_inner_iid(str(bookID2))
                
                pairSimilarity = similarityMatrix[innerBookID1][innerBookID2]
                totalSimilarity += pairSimilarity
                numOfPairs += 1
                
        if (numOfPairs > 0):
            similarity = totalSimilarity / numOfPairs
            diversity = 1 - similarity
            return diversity
        else:
            return 0 
                
    
    
            
                
                
                 
                    
                 
    
    
                    
            
        
            
            
            
        

        