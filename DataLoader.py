# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 15:45:26 2026

@author: zambe
"""

from surprise import Dataset
from surprise import Reader
import csv
import pandas as pd

class DataLoader:
    
    ratings = 'goodbooks-10k-master/ratings.csv'
    books = 'goodbooks-10k-master/books.csv'
    bookTags = 'goodbooks-10k-master/book_tags.csv'
    tags = 'goodbooks-10k-master/tags.csv'
    
    def __init__(self):
        self.bookID_to_title = {}
        self.title_to_bookID = {}
    
    def loadBookRatings(self):
       
        ratingsReader = Reader(line_format='user item rating', sep=',', skip_lines=1)
        
        ratingsDataset = Dataset.load_from_file(self.ratings, ratingsReader)
        
        # bookID <-> title mapping
        with open(self.books, newline='', encoding='utf-8') as csvfile:
            bookReader = csv.DictReader(csvfile)
            
            for row in bookReader:
                bookID = int(row['book_id'])
                title = row['title']
                
                self.bookID_to_title[bookID] = title
                self.title_to_bookID[title] = bookID
        
        return ratingsDataset
    
    def loadRatingsForTFRS(self):
        ratingsDataset = pd.read_csv(self.ratings)
        
        ratingsDataset['user_id'] = ratingsDataset['user_id'].astype(str)
        ratingsDataset['book_id'] = ratingsDataset['book_id'].astype(str)
        ratingsDataset['rating'] = ratingsDataset['rating'].astype('float32')
        
        return ratingsDataset
    
    def loadBookMetadata(self, bookTags):
        booksDataset = pd.read_csv(self.books)
        
        metadata = booksDataset[[
            'book_id', 'authors', 'original_publication_year'
        ]].copy()
        
        metadata['book_id'] = metadata['book_id'].astype(str)
        
        metadata = metadata.merge(bookTags, on='book_id', how='left')
        
        medianYear = metadata['original_publication_year'].median()
        
        metadata['authors'] = metadata['authors'].fillna('')
        metadata['original_publication_year'] = (
            metadata['original_publication_year'].fillna(medianYear).astype('float32')
        )
        # if a book doesn't have tags --> empty list of 5 instead of NaN
        metadata['tag_name'] = metadata['tag_name'].apply(
           lambda tags: tags if isinstance(tags, list) else ['','','','',''] 
        )
        
        return metadata
    
    def loadBookTags(self, topK=5):
        bookTagsDataset = pd.read_csv(self.bookTags)
        tagsDataset = pd.read_csv(self.tags)
        booksDataset = pd.read_csv(self.books)
        
        # tag ID <-> tag name 
        bookTagsDataset = bookTagsDataset.merge(tagsDataset, on='tag_id')
        
        # goodreads book ID <-> internal book ID
        bookTagsDataset = bookTagsDataset.merge(
            booksDataset[['book_id', 'goodreads_book_id']], on='goodreads_book_id'
        )
        
        # most common tags on the top
        bookTagsDataset = bookTagsDataset.sort_values(
            ['book_id', 'count'], ascending=[True, False]
        )
        
        # keeping top K tags per book
        topTags = bookTagsDataset.groupby('book_id').head(topK).copy()
        
        # a list of tags for each book
        topTags = topTags.groupby('book_id')['tag_name'].apply(list).reset_index()
        
        topTags['book_id'] = topTags['book_id'].astype(str)
        
        # if there are less than topK (5) tags for a book fill the spots with ''
        for index, row in topTags.iterrows():
            tags = row['tag_name']
            
            while (len(tags) < topK):
                tags.append('')
                
            topTags.at[index, 'tag_name'] = tags
            
        return topTags
    
    def getMetadataByBookID(self, bookMetadata):
        metadataByBookID = {}
        
        for index, row in bookMetadata.iterrows():
            bookID = row['book_id']
            metadataByBookID[bookID] = (
                row['authors'], row['original_publication_year'], row['tag_name']
            )
            
        return metadataByBookID
                
    def getBookTitle(self, bookID):
        if bookID in self.bookID_to_title: 
            return self.bookID_to_title[bookID]
        return ""
    
    def getBookID(self, title):
        if title in self.title_to_bookID:
            return self.title_to_bookID[title]
        return ""
    
    def getAllBookIDs(self):
        return set(self.bookID_to_title.keys())
    
    def getUniqueUserIDs(self, ratingsDF):
        return ratingsDF['user_id'].unique()
    
    def getUniqueBookIDs(self, ratingsDF):
        return ratingsDF['book_id'].unique()
