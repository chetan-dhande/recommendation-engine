# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 18:16:41 2020

@author : Chetan
"""
import pandas as pd
from sklearn.metrics.pairwise import  cosine_similarity
import matplotlib.pyplot as plt
df = pd.read_csv("D:\\chetan\\assignment\\8.recommendation system\\book.csv",encoding='latin-1')

df.columns
df.head(5)

df.head()
df["Book.Title"].head(5)
df["Book.Rating"].isnull().sum() 
df["User.ID"].unique().shape
df.describe()
df.head()

#unique values

unique_user = len(df["User.ID"].unique())
print(unique_user)

unique_books = len(df["Book.Title"].unique())
print(unique_books)
len(df["Book.Rating"].unique())

ratings = pd.DataFrame(df.groupby("Book.Title")["Book.Rating"].mean())
print(ratings)
ratings["no_of_ratings"] = df.groupby("Book.Title")["Book.Rating"].count()
ratings = ratings.reset_index()

plt.scatter(ratings["Book.Title"].head(),ratings["no_of_ratings"].head())
ratings.columns

df.columns
book_matrix = df.pivot_table(index='Book.Title', columns='User.ID', values='Book.Rating')

len(book_matrix.isnull())

book_matrix = book_matrix.fillna(0)

def std(x):
    y = (x - x.mean())/(x.max() - x.min())
    return y

book_matrix = book_matrix.apply(std)
book_matrix.head(5)
books = df['Book.Title'].unique()
ratings_simil = pd.DataFrame((cosine_similarity(book_matrix)),index = books,columns=books)

def recommendation(book,no):
    score = ratings_simil[str(book)]
    score=score.sort_values(ascending=False).head(int(no))
    return score
   
recommendation("Classical Mythology",2)   
  

def recommendation2():
    book = str(input("enter fav books = "))
    no = int(input("number of sugesstion want = "))
    score = ratings_simil[str(book)]
    score=score.sort_values(ascending=False).head(int(no))
    return score

recommendation2()


