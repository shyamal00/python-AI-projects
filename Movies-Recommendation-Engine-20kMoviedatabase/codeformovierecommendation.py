# Building a movie recommendation engine using python

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('movies.csv',low_memory=False)
ms = df.head(3)
#print(ms)

#important features which are important title  original_language genres vote_average vote_count 

#get a count of the number of rows/movies in that dataset

size = df.shape
#print(size)

#Create a list of important columns for recommendation

column = ['Title','genre','actors','director','votes']
columnsize = df[column].head(3)
print(columnsize)

#check for any missing values in the important columns
missvalue=df[column].isnull().values.any
print(missvalue)

def get_important_features(data):
    important_features = []
    for i in range(0, data.shape[0]):
        important_features.append(str(data['genre'][i])+' '+str(data['Title'][i])+' '+str(data['actors'][i])+' '+str(data['director'][i]))
    return important_features
    

#create a column to hold the combined strings
df['important_features'] = get_important_features(df)
a = df.head(3)
print(a)

#Convert the text into matrix of counts

cm = CountVectorizer().fit_transform(df['important_features'])

#get the cosine similarity matrix from the count text

cs = cosine_similarity(cm)
print(cs)


#get the shape of the cosine similarity matrix 
ab = cs.shape
print(ab)

T = 'Inception'

#find the movie id

movie_id = df[df.Title == T]['Movie_id'].values[0]

# create a list of enumearate for the similarity score
scores = list(enumerate(cs[movie_id]))
#print(scores)

#sort the list
sorted_scores = sorted(scores, key = lambda X:X[1], reverse=True)
sorted_scores = sorted_scores[1:]
#print(sorted_scores)


#create a loop to print the first 7 similar movies

j = 0 
print('The 7 most recommended movies to',T,'are:\n')
for item in sorted_scores:
    movie_title = df[df.Movie_id == item[0]]['Title'].values[0]
    print(j+1,movie_title)
    j = j+1
    if j>6:
        break

