#!/usr/bin/env python
# coding: utf-8

# # Load the Libraries

# In[1]:


import numpy as np
import pandas as pd


# # Load the DataSet

# In[4]:


df1=pd.read_csv('tmdb_5000_credits.csv.zip')
df1.head()


# # Get the basic information of Data

# In[6]:


df1.shape


# In[7]:


df1.info()


# # Load the second dataset : Movies Dataset

# In[8]:


df2=pd.read_csv("tmdb_5000_movies.csv.zip")
df2.head()


# # Basic information about Dataset

# In[9]:


df2.shape


# In[10]:


df2.info()


# # Merge the two Dataframes

# In[11]:


df1.columns=['id','title','cast','crew']
df2=df2.merge(df1,on='id')


# In[12]:


df2.head()


# In[13]:


df2.shape


# In[15]:


df2.columns


# In[18]:


C=df2['vote_average'].mean()
C


# # Minimum votes to be listed

# In[20]:


m=df2['vote_count'].quantile(0.9)
m


# # Getting list of movies to be listed

# In[21]:


list_movies=df2.copy().loc[df2['vote_count']>=m]
list_movies.shape


# # Defining a function

# In[22]:


def weighted_rating(x,m=m,C=C):
    v=x['vote_count'] 
    R=x['vote_average']
    #Calculation based on IMDB formula (m=1838, c=6.09)
    return (v/(v+m)*R)+(v/(v+m)*C)
    


# In[23]:


#Define a new feature 'score'and calculate its value with 'weighted rating()'
list_movies['score']=list_movies.apply(weighted_rating,axis=1)


# In[24]:


list_movies.head(3)


# In[25]:


list_movies.shape


# # Sort the movies

# In[29]:


#Sort movies based on score calculated above
list_movies=list_movies.sort_values('score',ascending=False)

#Print the top 15 movies
list_movies[['title_x','vote_count','vote_average','score']].head(10)
           


# # Most popular movies

# In[34]:


pop=df2.sort_values('popularity',ascending=False)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))

plt.barh(pop['title_x'].head(6),pop['popularity'].head(6),align='center',color='m')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular movies")


# In[35]:


df2.columns


# # High Budget Movies

# In[39]:


pop=df2.sort_values('budget',ascending=False)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))

plt.barh(pop['title_x'].head(6),pop['popularity'].head(6),align='center',color='lightblue')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("High budget movies")


# #  Revenue on Movies

# In[38]:


pop=df2.sort_values('revenue',ascending=False)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))

plt.barh(pop['title_x'].head(6),pop['popularity'].head(6),align='center',color='pink')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Revenue on movies")


# # Drop the title_y from the data frame

# In[40]:


list_movies.drop(['title_y'],axis=1,inplace=True)


# In[43]:


list_movies.shape


# In[44]:


list_movies.head(2)


# # Overview Column

# In[45]:


df2['overview'].head(10)


# based on description we shall find the similarity among the movies

# In[48]:


from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf=TfidfVectorizer(stop_words='english')

#Replace NAN with an empty string
df2['overview']=df2['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix=tfidf.fit_transform(df2['overview'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape


# In[50]:


#Import linear_kernal
from sklearn.metrics.pairwise import linear_kernel

#Compute the cosine similarity matrix
cosine_sim=linear_kernel(tfidf_matrix,tfidf_matrix)


# In[51]:


#Construct a rverse map of indices and movie titles
indices=pd.Series(df2.index,index=df2['title_x']).drop_duplicates()


# In[61]:


#Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title,cosine_sim=cosine_sim):
    #Get the index of the movie that matches the file
    idx=indices[title]
    
    #Get the pairwise similarities of all the movies wiith that movie
    sim_scores=list(enumerate(cosine_sim[idx]))
    
    #Sort the movies based on similarity scores
    sim_scores=sorted(sim_scores,key=lambda x:x[1], reverse=True)
    
    #Get the scores of the 10 most similar movies
    sim_scores=sim_scores[1:11]
    
    #Get the movies indices
    movie_indices=[i[0] for i in sim_scores]
    
    #Return the top 10 most similar movies
    return df2['title_x'].iloc[movie_indices]


# In[63]:


get_recommendations('The Dark Knight Rises')


# In[64]:


get_recommendations('JFK')


# In[66]:


get_recommendations('Avatar')


# In[71]:


get_recommendations('Speed')


# In[ ]:




