# BOOKVISTA-MACHINE-LEARNING--INFUSED-BOOK-RECOMMENDATION-SYSTEM
The Book Recommendation System using K-Nearest Neighbors (KNN) algorithm in Machine Learning is a project aimed at developing an intelligent system that can recommend books to users based on their preferences and similarities with other books. The project leverages the power of the KNN algorithm to make accurate and personalized recommendations.
import pandas as pd
import numpy as np

# for data visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# for interactive plots
import ipywidgets
from ipywidgets import interact
from ipywidgets import interact_manual
df = pd.read_csv("/books (1).csv", error_bad_lines = False)
df.columns = df.columns.str.strip()
df.columns
df.describe(include = 'object')
df.isnull().sum()
df.duplicated().any
df.info()
df.isbn.nunique()
df.isbn13.nunique()
df.drop(['bookID', 'isbn', 'isbn13'], axis = 1, inplace = True)
df.publication_date
df['year'] = df['publication_date'].str.split('/')
df['year'] = df['year'].apply(lambda x:x[0])
df['year'] = df['year'].astype('int')
df.dtypes
#Exploratory Data Analysis
df[df['year'] == 2020][['title', 'authors','average_rating','language_code','publisher' ]]
df.groupby(['year'])['title'].agg('count').sort_values(ascending = False).head(20)
plt.figure(figsize = (20, 10))
sns.countplot(x = 'authors', data = df,
             order = df['authors'].value_counts().iloc[:10].index)
plt.title("Top 10 Authors with maximum book publish")
plt.xticks(fontsize = 12)
plt.show()
df.language_code.value_counts()
df.groupby(['language_code'])[['average_rating',
                               'ratings_count',
                               'text_reviews_count']].agg('mean').style.background_gradient(cmap = 'Wistia')
book = df['title'].value_counts()[:20]
book
# to find most occuring book in our data
plt.figure(figsize = (20, 6))
book = df['title'].value_counts()[:20]
sns.barplot(x = book.index, y = book,
           palette = 'winter_r')
plt.title("Most occuring Books")
plt.xlabel("Number of Occurance")
plt.ylabel("Books")
plt.xticks(rotation = 75, fontsize = 13)
plt.show()
sns.histplot(df['average_rating'])
plt.show()
df[df.average_rating == df.average_rating.max()][['title','authors','language_code','publisher']]
publisher = df['publisher'].value_counts()[:20]
publisher
publisher = df['publisher'].value_counts()[:20]
sns.barplot(x = publisher.index, y = publisher, palette = 'winter_r')
plt.title("Publishers")
plt.xlabel("Number of Occurance")
plt.ylabel("Publishers")
plt.xticks(rotation = 75, fontsize = 13)
plt.show()
#Recommending Books based on Publishers
#Recommending Books based on Authors
#Recommending Books based on Language
df.publisher.value_counts()
def recomd_books_publisheres(x):
    a = df[df['publisher'] == x][['title', 'average_rating']]
    a = a.sort_values(by = 'average_rating', ascending = False)
    return a.head(10)
recomd_books_publisheres('Vintage')
recomd_books_publisheres('Penguin Books')
@interact
def recomd_books_publishers(publisher_name = list(df['publisher'].value_counts().index)):
    a = df[df['publisher'] == publisher_name][['title', 'average_rating']]
    a = a.sort_values(by = 'average_rating', ascending = False)
    return a.head(10)
#Based upon Authors

@interact
def recomd_books_authors(authors_name = list(df['authors'].value_counts().index)):
    a = df[df['authors'] == authors_name][['title', 'average_rating']]
    a = a.sort_values(by = 'average_rating', ascending = False)
    return a.head(10)
@interact
def recomd_books_lang(language = list(df['language_code'].value_counts().index)):
    a = df[df['language_code'] == language][['title', 'average_rating']]
    a = a.sort_values(by = 'average_rating', ascending = False)
    return a.head(10)
#Data Preprocessing

df.head(2)
df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce')
def num_to_obj(x):
    if x > 0 and x <= 1:
        return "between 0 and 1"
    if x > 1 and x <= 2:
        return "between 1 and 2"
    if x > 2 and x <= 3:
        return "between 2 and 3"
    if x > 3 and x <= 4:
        return "between 3 and 4"
    if x > 4 and x <= 5:
        return "between 4 and 5"


df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce')

df['rating_obj'] = df['average_rating'].apply(num_to_obj)
df['rating_obj'].value_counts()
language_df = pd.get_dummies(df['language_code'])
language_df.head()
features = pd.concat([rating_df,language_df, df['average_rating'],
                    df['ratings_count'], df['title']], axis = 1)
features.set_index('title', inplace= True)
features.head()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
#Model Building

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
nan_rows = np.isnan(features_scaled).any(axis=1)
features_scaled = features_scaled[~nan_rows]
model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', metric='euclidean')
model.fit(features_scaled)
dist, idlist = model.kneighbors(features_scaled)
df['title'].value_counts()
df['title'].value_counts()
@interact
def BookRecommender(book_name=list(df['title'].value_counts().index)):
    book_list_name = []
    book_id = df[df['title'] == book_name].index
    book_id = book_id[0]
    for newid in idlist[book_id]:
        book_list_name.append(df.iloc[newid]['title'])
    return book_list_name
from ipywidgets import interact, Dropdown

def BookRecommender(book_name):
    book_list_name = []
    book_id = df[df['title'] == book_name].index
    book_id = book_id[0]
    for newid in idlist[book_id]:
        book_list_name.append(df.iloc[newid]['title'])
    return book_list_name

book_names = list(df['title'].value_counts().index)
interact(BookRecommender, book_name=Dropdown(options=book_names))


