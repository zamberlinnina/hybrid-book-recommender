# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 16:18:49 2026

@author: zambe
"""
from EvaluationData import EvaluationData
from AlgorithmEvaluation import AlgorithmEvaluation

class Evaluator:
    
    def __init__(self, data, allBookIDs, dataLoader):
        self.algorithms = []
        self.data = EvaluationData(data, allBookIDs)
        self.dataLoader = dataLoader
        
    def addAlgorithm(self, algorithm, name):
        alg = AlgorithmEvaluation(algorithm, name)
        self.algorithms.append(alg)
    
    def printResults(self, topNMetrics):
        
        for algorithm in self.algorithms:
            algorithmName = algorithm.getAlgorithmName()
            print('\nEvaluating ', algorithmName, '...')
            metrics = algorithm.Evaluate(self.data, topNMetrics)
            
            print('\n')
          
            print('Results for ', algorithmName, ':')
            print('------------------------')
            print('{:<10} {:<8.4f} {:<10} {:<8.4f}'.format(
                'RMSE', metrics['RMSE'], 'MAE', metrics['MAE']))
            
            if (topNMetrics):
                print('{:<10} {:<8.4f} {:<10} {:<8.4f} {:<10} {:<8.4f}'.format(
                    'Precision', metrics['Precision'], 'Recall', metrics['Recall'], 'NDCG', metrics['NDCG']))
                
                rhrString = ', '.join(f'{int(rating)}: {hitRate:8.4f}'
                      for rating, hitRate in metrics['RHR'].items())
                print('{:<10} {:<8.4f} {:<10} {:<8.4f} {:<10} {}'.format(
                    'HR', metrics['HR'], 'CHR', metrics['CHR'], 'RHR', rhrString))
                
                print('{:<10} {:<8.4f} {:<10} {:<8.4f} {:<10} {:<8.4f}'.format(
                    'Coverage', metrics['CCoverage'], 'Diversity', metrics['Diversity'], 'Novelty', metrics['Novelty']))
                
                algorithm.getSampledUserTopN(self.data, self.dataLoader)
                
            
                