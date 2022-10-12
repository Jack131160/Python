#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


movie_df = pd.read_csv('https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/title.basics.tsv', sep='\t') 
rating_df = pd.read_csv('https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/title.ratings.tsv', sep='\t') 


# In[3]:


print(movie_df.head())


# In[4]:


print(movie_df.info())


# In[5]:


print(movie_df.isnull().sum())


# In[ ]:


#Discarding Data with NULL Value


# In[6]:


print(movie_df.loc[(movie_df['primaryTitle'].isnull()) | (movie_df['originalTitle'].isnull())])


# In[ ]:





# In[7]:


#update movie_df by removing NULL values
movie_df = movie_df.loc[(movie_df['primaryTitle'].notnull()) & (movie_df['originalTitle'].notnull())]
#display the number of data after data with NULL values ​​are discarded
print(len(movie_df))


# In[8]:


#Discarding Data with NULL Value
movie_df.loc[movie_df['genres'].isnull()] 


# In[9]:


#update movie_df by removing NULL values
movie_df = movie_df.loc[movie_df['genres'].notnull()]

#display the number of data after data with NULL values ​​are discarded
print(len(movie_df))


# In[10]:


#change the value of '\\N' on startYear to np.nan and cast the column to float64
movie_df['startYear'] = movie_df['startYear'].replace('\\N',np.nan)
movie_df['startYear'] = movie_df['startYear'].astype('float64')
print(movie_df['startYear'].unique()[:5])

#change the value of '\\N' in endYear to np.nan and cast the column to float64
movie_df['endYear'] = movie_df['endYear'].replace('\\N',np.nan)
movie_df['endYear'] = movie_df['endYear'].astype('float64')
print(movie_df['endYear'].unique()[:5])

#changed the value of '\\N' at runtimeMinutes to np.nan and cast the column to float64
movie_df['runtimeMinutes'] = movie_df['runtimeMinutes'].replace('\\N',np.nan)
movie_df['runtimeMinutes'] = movie_df['runtimeMinutes'].astype('float64')
print(movie_df['runtimeMinutes'].unique()[:5])


# In[11]:


#Change the genres value to a list
def transform_to_list(x):
    if ',' in x: 
    # change it to a list if there is data in the genre column
        return x.split(',')
    else: 
    #if there is no data, change it to an empty list
        return [] 

movie_df['genres'] = movie_df['genres'].apply(lambda x: transform_to_list(x))


# In[12]:


rating_df.head()


# In[13]:


#Join both tables
movie_rating_df = pd.merge(movie_df, rating_df, on='tconst', how='inner')

#Show top 5 data
print(movie_rating_df.head())

#Show data type of each column
print(movie_rating_df.info())


# In[14]:


#reduce table size by removing all NULL values ​​from the startYear and runtimeMinutes fields
#because it doesn't make sense if the film is not known when the release year and duration is
movie_rating_df = movie_rating_df.dropna(subset=['startYear','runtimeMinutes'])

#To ensure that there are no more NULL values
print(movie_rating_df.info())


# In[15]:


#find the value of C which is the average of averageRating
C = movie_rating_df['averageRating'].mean()
print(C)


# In[16]:


#Take samples of films with numVotes above 80% of the population,
# so the population that will be taken is only 20%
m = movie_rating_df['numVotes'].quantile(0.8)
print(m)


# In[18]:


#Create a function using a dataframe as a variable
def imdb_weighted_rating(df, var=0.8):
    v = df['numVotes']
    R = df['averageRating']
    C = df['averageRating'].mean()
    m = df['numVotes'].quantile(var)
    df['score'] = (v/(m+v))*R+(m/(m+v))*C #Rumus IMDb 
    return df['score']
    
imdb_weighted_rating(movie_rating_df)


# In[19]:


#check dataframes
print(movie_rating_df.head())


# In[ ]:


#From the function above, a 'score' field has been added to help create a simple recommender system. The first thing to do is to filter numVotes that are more than m then sort the scores from highest to lowest, to take the values ​​of the top few values.


# In[20]:


def simple_recommender(df, top=100):
    df = df.loc[df['numVotes'] >= m]
    df = df.sort_values(by='score', ascending=False) #urutkan dari nilai tertinggi ke terendah
    
   #Take top 100 data
    df = df[:top]
    return df


# In[21]:


#Take top 25 data

print(simple_recommender(movie_rating_df, top=25))


# In[ ]:


#Simple recommender system with user preferences


# In[22]:


df = movie_rating_df.copy()

def user_prefer_recommender(df, ask_adult, ask_start_year, ask_genre, top=100):
    #ask_adult = yes/no
    if ask_adult.lower() == 'yes':
        df = df.loc[df['isAdult'] == 1]
    elif ask_adult.lower() == 'no':
        df = df.loc[df['isAdult'] == 0]

    #ask_start_year = numeric
    df = df.loc[df['startYear'] >= int(ask_start_year)]

    #ask_genre = 'all' or else
    if ask_genre.lower() == 'all':
        df = df
    else:
        def filter_genre(x):
            if ask_genre.lower() in str(x).lower():
                return True
            else:
                return False
        df = df.loc[df['genres'].apply(lambda x: filter_genre(x))]

    df = df.loc[df['numVotes'] >= m] #Take a movie with a bigger m than numVotes
    df = df.sort_values(by='score', ascending=False)
    
    #if you just want to take the top 100
    df = df[:top]
    return df

print(user_prefer_recommender(df,
                       ask_adult = 'no',
                        ask_start_year = 2000,
                       ask_genre = 'drama'
                       ))


# In[ ]:




