
# coding: utf-8

# In[1]:

import pandas as pd
dataFile='/resources/data/BX-Book-Ratings.csv'
#use row as 0 first, then it is incremented every time
data=pd.read_csv(dataFile,sep=";",header=0,names=["user","isbn","rating"])


# In[2]:

data.head()


# In[3]:

bookFile='/resources/data/BX-Books.csv'
#take first 3 cols only. take isbn as the row
books=pd.read_csv(bookFile,sep=";",header=0,error_bad_lines=False, usecols=[0,1,2],index_col=0,names=['isbn',"title","author"])


# In[4]:

books.head()


# In[5]:

#Break the title and author.
def bookMeta(isbn):
    title = books.at[isbn,"title"]
    author = books.at[isbn,"author"]
    return title, author
bookMeta("0671027360")


# In[7]:




# In[6]:

def faveBooks(user,N):
    #Filter data relevant to user from the 'data' dataframe
    userRatings = data[data["user"]==user]
    #Sort by rating col in descending order and give the top N favourite books 
    sortedRatings = pd.DataFrame.sort_values(userRatings,['rating'],ascending=[0])[:N] 
    #take the table of N sorted rating and add col 'title' applying the bookMeta function
    sortedRatings["title"] = sortedRatings["isbn"].apply(bookMeta)
    return sortedRatings


# In[7]:

#Proceed with onlythe isbn present in the table
data = data[data["isbn"].isin(books.index)]
#Check if it works
faveBooks(204622,5)


# In[9]:

data.shape


# In[8]:

#Now we have to make a matrix of ratings with users as rows and isbn as columns
#Calculate the distinch no. of isbns present in the dataframe and no. of users who rated that isbn
usersPerISBN = data.isbn.value_counts()
usersPerISBN.head(10)


# In[11]:

usersPerISBN.shape


# In[9]:

#distinct set of users and the no. of books they have rated
ISBNsPerUser = data.user.value_counts()


# In[10]:

#NO. of users in the matrix
ISBNsPerUser.shape


# In[11]:

#Subset data to isbn read by more than 10 users. get the isbn by using .index. we keep rows which match the selected list of rows.
data = data[data["isbn"].isin(usersPerISBN[usersPerISBN>10].index)]


# In[12]:

#Keep users who have read more than 10 books.
data = data[data["user"].isin(ISBNsPerUser[ISBNsPerUser>10].index)]
#So now we have books which have been read frequently and users who have read a lot of books. No more a sparse matrix.


# In[13]:

#Creating rating matrix. col used to fill the cells in values parameter is rating. col that will be used for the row names in the index parameter(user col). col used to fill col names is isbn
userItemRatingMatrix=pd.pivot_table(data, values='rating',
                                    index=['user'], columns=['isbn'])


# In[14]:

userItemRatingMatrix.head()


# In[19]:

userItemRatingMatrix.shape


# In[15]:

#lets assign 2 users with user id from the dataset.
user1 = 204622
user2 = 255489


# In[16]:

#IN the table user is in row so we transpose it for the purpose of comparision with other user in order to calculate distance.
user1Ratings = userItemRatingMatrix.transpose()[user1]
user1Ratings.head()


# In[17]:

user2Ratings = userItemRatingMatrix.transpose()[user2]


# In[18]:

from scipy.spatial.distance import hamming 
#Gives the percentage disagreement
hamming(user1Ratings,user2Ratings)


# In[19]:

import numpy as np
def distance(user1,user2):
        try:
            user1Ratings = userItemRatingMatrix.transpose()[user1]
            user2Ratings = userItemRatingMatrix.transpose()[user2]
            distance = hamming(user1Ratings,user2Ratings)
        except: 
            distance = np.NaN
        return distance 


# In[20]:

distance(204622,10118)


# In[21]:

user = 204622
#Get the id for all users
allUsers = pd.DataFrame(userItemRatingMatrix.index)
allUsers = allUsers[allUsers.user!=user]
allUsers.head()


# In[22]:

allUsers["distance"] = allUsers["user"].apply(lambda x: distance(user,x))


# In[23]:

allUsers.head()


# In[24]:

K = 10
KnearestUsers = allUsers.sort_values(["distance"],ascending=True)["user"][:K]


# In[25]:

KnearestUsers


# In[29]:

def nearestNeighbors(user,K=10):
    #Fetch all the users
    allUsers = pd.DataFrame(userItemRatingMatrix.index)
    #Except the active user
    allUsers = allUsers[allUsers.user!=user]
    #Find the distance between each user and active user
    allUsers["distance"] = allUsers["user"].apply(lambda x: distance(user,x))
    #Sort and find the k nearest neighbours
    KnearestUsers = allUsers.sort_values(["distance"],ascending=True)["user"][:K]
    return KnearestUsers


# In[30]:

KnearestUsers = nearestNeighbors(user)


# In[31]:

KnearestUsers


# In[33]:

#Get the rating of nearest neighbours for all books.
NNRatings = userItemRatingMatrix[userItemRatingMatrix.index.isin(KnearestUsers)]
NNRatings


# In[34]:

#Take the NNrating df and compute the nanmean. each col will return a value. drop books which dont have a rating
avgRating = NNRatings.apply(np.nanmean).dropna()
avgRating.head()


# In[35]:

#Get the ratings of active user and drop books without a rating and get the isbn by .index
booksAlreadyRead = userItemRatingMatrix.transpose()[user].dropna().index
booksAlreadyRead


# In[36]:

#remove the average ratings for books already read by the user 
avgRating = avgRating[~avgRating.index.isin(booksAlreadyRead)]


# In[37]:

N=3
topNISBNs = avgRating.sort_values(ascending=False).index[:N]


# In[39]:

#Apply the bookMeta function to get the top N isbn
pd.Series(topNISBNs).apply(bookMeta)


# In[40]:

#We want 3 recommendations for the user
def topN(user,N=3):
    #Get the ratings the nearest neighbours for all books
    KnearestUsers = nearestNeighbors(user)
    NNRatings = userItemRatingMatrix[userItemRatingMatrix.index.isin(KnearestUsers)]
    #Find the average rating of the nearest neighbours for all books
    avgRating = NNRatings.apply(np.nanmean).dropna()
    booksAlreadyRead = userItemRatingMatrix.transpose()[user].dropna().index
    #Remove the average ratings for books already rated
    avgRating = avgRating[~avgRating.index.isin(booksAlreadyRead)]
    topNISBNs = avgRating.sort_values(ascending=False).index[:N]
    return pd.Series(topNISBNs).apply(bookMeta)


# In[41]:

#Print out list of books for a user
faveBooks(204813,10)


# In[44]:

#Print top 10 recommendations for the user. 2 of the books are by Nora Roberts!
topN(204813,10)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



