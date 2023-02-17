
#https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pearsonr.html

#https://medium.com/swlh/how-to-build-simple-recommender-systems-in-python-647e5bcd78bd

#https://medium.com/analytics-vidhya/recommendation-system-using-collaborative-filtering-cc310e641fde

#https://github.com/pancr9/Netflix-Recommender-System

#https://medium.com/@suvasan19/collaborative-filtering-using-python-and-simple-similarity-algorithms-4591fd7851d7

#https://towardsdatascience.com/collaborative-filtering-based-recommendation-systems-exemplified-ecbffe1c20b1

#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html


#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html

from operator import index
import os
import math
from random import Random, random
from time import process_time_ns
import numpy as np
import pandas as pd
import torch
import tensorly as tl
from numpy.core.fromnumeric import shape
from typing import Tuple, List
from tensorly.random import random_tucker


from scipy.stats import pearsonr
from scipy.spatial import distance

from sklearn import preprocessing

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import mean_squared_error

def CreateMatrix():
    df = pd.read_csv("{0}\data\data.csv".format(os.getcwd()))
    df = df.reset_index()
    data_len = len(df)

    #print(df.to_numpy()[0])
    #print(data_len)

    main_matrix = np.empty(shape=(data_len * 10, 3))
    y_pred = np.empty(shape=(data_len * 10), dtype=int)
    m_for_cluster = np.empty(shape=(data_len, 6))
    
    index = 0
    for i, row in df.iterrows():
        #print(i)
        #print(row)        

        # m_user_item[i] = [row['M1'], row['M2'], row['M3'], row['M4'], row['M5'], 
        #     row['M6'], row['M7'], row['M8'], row['M9'], row['M10']]
        
        m_for_cluster[i] = [row['EI'], row['N'], row['E'], row['O'], row['A'], row['C']]
        
        for j in range(10):
            main_matrix[index] = [
                int(row['UserID']), 
                int(j+1), 
                int(row['M'+str(j+1)]) 
            ]
            #print(main_matrix[index])
            index = index + 1

    #print("main_matrix:" , main_matrix)
    #print("shape(main_matrix):", shape(main_matrix))
    #print("m_for_cluster:", m_for_cluster)
    
    return m_for_cluster, main_matrix, y_pred


def ListHelperUnique(List):
    unique_list = []
    for x in List:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

def MiniBatchKMeans_Clustering(Matrix, Clusters):
    print("shape Matrix =" + str(shape(Matrix)))
    kmeans = MiniBatchKMeans(n_clusters=Clusters, random_state=1, batch_size=10000, verbose=0)
    kmeans = kmeans.partial_fit(Matrix)
    kmeans.fit_predict(Matrix)
    print("MiniBatchKMeans_Clustering.labels:")
    print(kmeans.labels_)
    Unique_Labels = ListHelperUnique(kmeans.labels_)
    print("Unique_Labels:")
    print(Unique_Labels)
    if (len(Unique_Labels) == 1):
        if(kmeans.labels_[0] == 0):
            kmeans.labels_[0] = 1
        else:
            kmeans.labels_[0] = 0

    _silhouette_score = silhouette_score(Matrix, kmeans.labels_, metric="euclidean")
    _bouldin_score = davies_bouldin_score(Matrix, kmeans.labels_)
    _calinski_harabasz_score = calinski_harabasz_score(Matrix, kmeans.labels_)

    print("For n_clusters =", Clusters,
          "The silhouette_score is :", _silhouette_score)
    
    print("For n_clusters =", Clusters, "The bouldin_score is :",
          _bouldin_score, "lower Is better")
    
    print("For n_clusters =", Clusters, "The calinski_harabasz_score is :",
          _calinski_harabasz_score, "higher Is better")

    print("End Of Kmeans Clustering")
    return kmeans.labels_

def PredictByPearsonR(item_main_matrix, main_matrix, m_for_cluster, label_cluster):

    UserID = int(item_main_matrix[0])
    UserIndex = int(item_main_matrix[0]) - 1
    ClusterNumber = label_cluster[UserIndex]
    m_for_cluster_item = m_for_cluster[UserIndex]

    min_distance = -1
    min_distance_index = -1    
    i=0
    for c in label_cluster:
        if(c == ClusterNumber):
            #print(i, c)
            m_for_cluster_user = m_for_cluster[i]
            if(UserIndex != i):
                r,p = pearsonr(m_for_cluster_item, m_for_cluster_item)
                #print(r, p)
                if(r > min_distance):
                    min_distance = r
                    min_distance_index = i
        i = i + 1 

    if(min_distance_index > -1):
        for m in main_matrix:
            if(m[0] == min_distance_index + 1 and m[1] == item_main_matrix[1]):
                return m[2]

    print("XXXXXXXXXXXXX")
    return 0

def PredictAll(main_matrix, m_for_cluster, label_cluster, y_pred):
    y_true = np.empty(shape=(len(y_pred)), dtype=int)
    count_false=0
    i=0
    for m in main_matrix:
        #print(m)
        #print(i)
        y_true[i] = m[2]
        y_pred[i] = PredictByPearsonR(m, main_matrix, m_for_cluster, label_cluster)
        if(y_true[i] != y_pred[i]):
            count_false = count_false + 1
            #print("y_true[i]:", y_true[i], ", y_pred[i]:", y_pred[i])
        i = i + 1
    
    error = mean_squared_error(y_true, y_pred)
    print("count_false:" , count_false)
    print("mean_squared_error:" , error)



def GetDataFrame():
    df = pd.read_csv("{0}\data\data.csv".format(os.getcwd()))
    df = df.reset_index()
    return df


def create_cosine_similarity_matrix_persons_by_character():
    
    df = pd.read_csv("{0}\data\data.csv".format(os.getcwd()))
    df = df.reset_index()

    data_len = len(df)
    main_matrix = np.empty(shape=(data_len, 8))
    
    #Gender	Age	EI	N	E	O	A	C

    index = 0
    for i, row in df.iterrows():
        main_matrix[index] = [            
            int(row['Gender']),
            int(row['Age']),
            int(row['EI']),
            int(row['N']),
            int(row['E']),
            int(row['O']),
            int(row['A']),
            int(row['C'])
        ]
        #print(main_matrix[index])
        index = index + 1
    #print(main_matrix)
    #print(shape(main_matrix))
    
    main_matrix = preprocessing.normalize(main_matrix)

    matrix_User_Based_Demographic = np.empty(shape=(data_len, data_len))
    i=0
    for m1 in main_matrix:
        j=0
        for m2 in main_matrix:
            matrix_User_Based_Demographic[i,j] = cosine_similarity([m1], [m2])
            j = j + 1
        i = i + 1
    
    print(matrix_User_Based_Demographic)
    print(shape(matrix_User_Based_Demographic))

    print("-----------create_cosine_similarity_matrix_persons_by_character-----------")

    return matrix_User_Based_Demographic

def create_cosine_similarity_matrix_persons_by_movies_ranks():
    df = pd.read_csv("{0}\data\data.csv".format(os.getcwd()))
    df = df.reset_index()

    data_len = len(df)
    main_matrix = np.empty(shape=(data_len, 50))
    
    index = 0
    for i, row in df.iterrows():

        for j in range(50):
            s = 'M' + str(j+1)           
            if math.isnan(row[s]):
                row[s] = -1

        main_matrix[index] = [
            int(row['M1']),
            int(row['M2']),
            int(row['M3']),
            int(row['M4']),
            int(row['M5']),
            int(row['M6']),
            int(row['M7']),
            int(row['M8']),
            int(row['M9']),
            int(row['M10']),

            int(row['M11']),
            int(row['M12']),
            int(row['M13']),
            int(row['M14']),
            int(row['M15']),
            int(row['M16']),
            int(row['M17']),
            int(row['M18']),
            int(row['M19']),
            int(row['M20']),

            int(row['M21']),
            int(row['M22']),
            int(row['M23']),
            int(row['M24']),
            int(row['M25']),
            int(row['M26']),
            int(row['M27']),
            int(row['M28']),
            int(row['M29']),
            int(row['M30']),

            int(row['M31']),
            int(row['M32']),
            int(row['M33']),
            int(row['M34']),
            int(row['M35']),
            int(row['M36']),            
            int(row['M37']),
            int(row['M38']),
            int(row['M39']),            
            int(row['M40']),

            int(row['M41']),
            int(row['M42']),
            int(row['M43']),
            int(row['M44']),
            int(row['M45']),
            int(row['M46']),
            int(row['M47']),
            int(row['M48']),
            int(row['M49']),
            int(row['M50'])            
        ]
        #print(main_matrix[index])
        index = index + 1
    #print(main_matrix)
    #print(shape(main_matrix))
    
    main_matrix = preprocessing.normalize(main_matrix)


    matrix_Persons_Based_Movies_Ranks = np.empty(shape=(data_len, data_len))
    i=0
    for m1 in main_matrix:
        j=0
        for m2 in main_matrix:
            matrix_Persons_Based_Movies_Ranks[i,j] = cosine_similarity([m1], [m2]) * distance.jaccard(m1, m2)
            j = j + 1
        i = i + 1
    
    print(matrix_Persons_Based_Movies_Ranks)
    print(shape(matrix_Persons_Based_Movies_Ranks))

    print("-----------create_cosine_similarity_matrix_persons_by_movies_ranks-----------")

    return matrix_Persons_Based_Movies_Ranks

def create_cosine_similarity_matrix_movies_by_genre():
    
    df = pd.read_csv("{0}\data\movies.csv".format(os.getcwd()))
    df = df.reset_index()

    data_len = len(df)
    main_matrix = np.empty(shape=(data_len, 10))

    #MovieID	Genre	Name	GenreID
    index = 0
    for i, row in df.iterrows():
        main_matrix[index] = [            
            int(row['G1']),
            int(row['G2']),
            int(row['G3']),
            int(row['G4']),
            int(row['G5']),
            int(row['G6']),
            int(row['G7']),
            int(row['G8']),
            int(row['G9']),
            int(row['G10'])
        ]
        #print(main_matrix[index])
        index = index + 1
    #print(main_matrix)
    #print(shape(main_matrix))

    main_matrix = preprocessing.normalize(main_matrix)

    matrix_Movie_Based_Genre = np.empty(shape=(data_len, data_len))
    i=0
    for m1 in main_matrix:
        j=0
        for m2 in main_matrix:
            matrix_Movie_Based_Genre[i,j] = movie_similarity(m1, m2)
            j = j + 1
        i = i + 1
    
    
    print(matrix_Movie_Based_Genre)
    print(shape(matrix_Movie_Based_Genre))

    
    print("-----------create_cosine_similarity_matrix_movies_by_genre-----------")

    return matrix_Movie_Based_Genre

def movie_similarity(m1, m2):

    q11 = 0
    q01 = 0
    q10 = 0

    for j in range(10):
        if m1[j] == m2[j] == 1:
            q11 = q11 + 1
        if m1[j] == 0 and m2[j] == 1:
            q01 = q01 + 1
        if m1[j] == 1 and m2[j] == 0:
            q10 = q10 + 1


    sim = q11 / (q11 + q01 + q10)
    
    return sim


def create_cosine_similarity_matrix_movies_by_movies_ranks():

    df = pd.read_csv("{0}\data\movies.csv".format(os.getcwd()))
    df = df.reset_index()

    data_len = len(df)
    main_matrix = np.empty(shape=(data_len, 51))

    index = 0
    for i, row in df.iterrows():

        for j in range(51):
            s = 'R' + str(j+1)           
            if math.isnan(row[s]):
                row[s] = -1

        main_matrix[index] = [
            int(row['R1']),
            int(row['R2']),
            int(row['R3']),
            int(row['R4']),
            int(row['R5']),
            int(row['R6']),
            int(row['R7']),
            int(row['R8']),
            int(row['R9']),
            int(row['R10']),

            int(row['R11']),
            int(row['R12']),
            int(row['R13']),
            int(row['R14']),
            int(row['R15']),
            int(row['R16']),
            int(row['R17']),
            int(row['R18']),
            int(row['R19']),
            int(row['R20']),

            int(row['R21']),
            int(row['R22']),
            int(row['R23']),
            int(row['R24']),
            int(row['R25']),
            int(row['R26']),
            int(row['R27']),
            int(row['R28']),
            int(row['R29']),
            int(row['R30']),

            int(row['R31']),
            int(row['R32']),
            int(row['R33']),
            int(row['R34']),
            int(row['R35']),
            int(row['R36']),
            int(row['R37']),
            int(row['R38']),
            int(row['R39']),
            int(row['R40']),

            int(row['R41']),
            int(row['R42']),
            int(row['R43']),
            int(row['R44']),
            int(row['R45']),
            int(row['R46']),
            int(row['R47']),
            int(row['R48']),
            int(row['R49']),
            int(row['R50']),
            
            int(row['R51']),
        ]
        #print(main_matrix[index])
        index = index + 1
    #print(main_matrix)
    #print(shape(main_matrix))

    main_matrix = preprocessing.normalize(main_matrix)

    matrix_Movies_Based_Movies_Ranks = np.empty(shape=(data_len, data_len))
    i=0
    for m1 in main_matrix:
        j=0
        for m2 in main_matrix:
            matrix_Movies_Based_Movies_Ranks[i,j] = cosine_similarity([m1], [m2]) * distance.jaccard(m1, m2)
            j = j + 1
        i = i + 1
    
    print(matrix_Movies_Based_Movies_Ranks)
    print(shape(matrix_Movies_Based_Movies_Ranks))

    print("-----------create_cosine_similarity_matrix_movies_by_movies_ranks-----------")

    return matrix_Movies_Based_Movies_Ranks
    


# m_for_cluster, main_matrix, y_pred = CreateMatrix()
# label_cluster_3 = MiniBatchKMeans_Clustering(m_for_cluster, 3)
# label_cluster_5 = MiniBatchKMeans_Clustering(m_for_cluster, 5)
# label_cluster_7 = MiniBatchKMeans_Clustering(m_for_cluster, 7)


# # print("main_matrix:" , main_matrix)
# # print("shape(main_matrix):", shape(main_matrix))

# PredictAll(main_matrix, m_for_cluster, label_cluster_3, y_pred)
# PredictAll(main_matrix, m_for_cluster, label_cluster_5, y_pred)
# PredictAll(main_matrix, m_for_cluster, label_cluster_7, y_pred)


matrix_persons_by_character = create_cosine_similarity_matrix_persons_by_character()
matrix_persons_by_movies_ranks = create_cosine_similarity_matrix_persons_by_movies_ranks()

matrix_movies_by_genre = create_cosine_similarity_matrix_movies_by_genre()
matrix_movies_by_movies_ranks = create_cosine_similarity_matrix_movies_by_movies_ranks()


print('-----------------phase 2-----------------')


mr1 = np.empty(shape=shape(matrix_persons_by_character))
mr2 = np.empty(shape=shape(matrix_movies_by_genre))


Rmatrix = np.empty(shape=(len(matrix_persons_by_character), 4))


for i in range(len(matrix_persons_by_character)):
    neigh = NearestNeighbors(n_neighbors=5)
    neigh.fit(matrix_persons_by_character)
    kneighbors = neigh.kneighbors([matrix_persons_by_character[i]])
    print(kneighbors)







i=0
for m1 in mr1:
    j=0
    for m2 in m1:
        r1 = matrix_persons_by_character[i, j]        
        r2 = matrix_persons_by_movies_ranks[i, j]

        if(r1 == 0 and r2 == 0):
            mr1[i, j] = 0
        elif(r1 != 0 and r2 == 0):
            mr1[i, j] = r1
        elif(r1 == 0 and r2 != 0):
            mr1[i, j] = r2
        else:
            mr1[i, j] = 2 * r1 * r2 / (r1 + r2)


        j = j + 1
    i = i + 1

# print(mr1)
# print(shape(mr1))

print('-----------------------------------------')

i=0
for m1 in mr2:
    j=0
    for m2 in m1:
        r1 = matrix_movies_by_genre[i, j]        
        r2 = matrix_movies_by_movies_ranks[i, j]

        if(r1 == 0 and r2 == 0):
            mr2[i, j] = 0
        elif(r1 != 0 and r2 == 0):
            mr2[i, j] = r1
        elif(r1 == 0 and r2 != 0):
            mr2[i, j] = r2
        else:
            mr2[i, j] = 2 * r1 * r2 / (r1 + r2)


        j = j + 1
    i = i + 1

# print(mr2)
# print(shape(mr2))

print('-----------------------------------------')


# x = cosine_similarity([[1, 1, 1, 1]], [[1, 0, 0, 1]])
# print(x)

# normalized = preprocessing.normalize([[1,3,565,8767,8678,7686,686]])
# print("Normalized Data = ", normalized)

# print("jaccard distance = ", distance.jaccard([1, 0, 0], [1, 1, 1]))

# samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
# neigh = NearestNeighbors(n_neighbors=1)
# neigh.fit(samples)
# print(neigh.kneighbors([[1., 1., 1.]]))



# x = cosine_similarity([[1, 1, 1, 1]], [[1, 0, 0, 1]])
# print(x)

# normalized = preprocessing.normalize([[1,3,565,8767,8678,7686,686]])
# print("Normalized Data = ", normalized)

# print("jaccard distance = ", distance.jaccard([1, 0, 0], [1, 1, 1]))

# samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
# neigh = NearestNeighbors(n_neighbors=1)
# neigh.fit(samples)
# print(neigh.kneighbors([[1., 1., 1.]]))