import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dataset1=pd.read_csv('D:\Dataset/tmdb_5000_credits.csv')
dataset2=pd.read_csv('D:\Dataset/tmdb_5000_movies.csv')
dataset1
dataset2
dataset1.columns=['id','tittle','cast','crew']
dataset2=dataset2.merge(dataset1,on='id')
C=dataset2['vote_average'].mean()
m=dataset2['vote_count'].quantile(0.9)
q_movies=dataset2.copy().loc[dataset2['vote_count']>=m]
def weighted_rating(x,m=m,C=C):
    V=x['vote_count']
    R=x['vote_average']
     #Calculation based on IMDB formula
    return(V/(V+m) * R) +(m/(m+V) * C)
q_movies['score']=q_movies.apply(weighted_rating,axis=1)
q_movies=q_movies.sort_values('score',ascending=False)
#Print the top 15 movies
q_movies[['tittle','vote_count','vote_average','score']].head(10)
pop=dataset2.sort_values('popularity',ascending=False)
plt.figure(figsize=(12,4))
plt.barh(pop['title'].head(6),pop['popularity'].head(6),align='center',color='green')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")
#Content Based Filtering
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object,Remove all english stop words such as 'the','a'
tfidf=TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
dataset2['overview']=dataset2['overview'].fillna('')

#Construct the TF-IDF matrix by fitting and transformimg the data 
tfidf_matrix=tfidf.fit_transform(dataset2['overview'])
tfidf_matrix   
from sklearn.metrics.pairwise import cosine_similarity
#Compute the cosine similarity matrix
cosine_sim=cosine_similarity(tfidf_matrix,tfidf_matrix)
#Construct a reverse map of index and movie titles
indices=pd.Series(dataset2.index,index=dataset2['title']).drop_duplicates()
#Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title,cosine_sim=cosine_sim):
    #Get the index of movie which matches the title
    idx=indices[title]
    
    #Get the pairwise similarity scores of all movies with that movie
    sim_scores=list(enumerate(cosine_sim[idx]))
    
    #Sort the movies based on the similarity scores
    sim_scores=sorted(sim_scores,key=lambda x:x[1],reverse=True)
    
    #Get the scores of 10 most similar movies
    sim_scores=sim_scores[1:11]
     
    #Get the movie indices
    movie_indices=[i[0] for i in sim_scores]
    
    #Return the top 10 most similar movies
    return dataset2['title'].iloc[movie_indices]
    get_recommendations('The Dark Knight Rises')
    get_recommendations('Minions')
    get_recommendations('Interstellar')
