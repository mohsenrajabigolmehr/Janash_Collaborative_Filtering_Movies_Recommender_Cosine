
#https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pearsonr.html

#https://medium.com/swlh/how-to-build-simple-recommender-systems-in-python-647e5bcd78bd

#https://medium.com/analytics-vidhya/recommendation-system-using-collaborative-filtering-cc310e641fde

#https://github.com/pancr9/Netflix-Recommender-System

#https://medium.com/@suvasan19/collaborative-filtering-using-python-and-simple-similarity-algorithms-4591fd7851d7

#https://towardsdatascience.com/collaborative-filtering-based-recommendation-systems-exemplified-ecbffe1c20b1

#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html


#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html

import os
import math
import numpy as np
import pandas as pd

from numpy.core.fromnumeric import shape
# from typing import Tuple, List
# from tensorly.random import random_tucker


# from scipy.stats import pearsonr
from scipy.spatial import distance

from sklearn import preprocessing

from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors

# from sklearn.cluster import MiniBatchKMeans
# from sklearn.metrics import silhouette_score
# from sklearn.metrics import davies_bouldin_score
# from sklearn.metrics import calinski_harabasz_score
# from sklearn.metrics import mean_squared_error



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
    

matrix_persons_by_character = create_cosine_similarity_matrix_persons_by_character()
matrix_persons_by_movies_ranks = create_cosine_similarity_matrix_persons_by_movies_ranks()

matrix_movies_by_genre = create_cosine_similarity_matrix_movies_by_genre()
matrix_movies_by_movies_ranks = create_cosine_similarity_matrix_movies_by_movies_ranks()

print('-----------------phase 2-----------------')

def calculate_all_weights():
    resp = []

    df_users = pd.read_csv("{0}\data\data.csv".format(os.getcwd()))
    df_users = df_users.reset_index()
    
    df_items = pd.read_csv("{0}\data\movies.csv".format(os.getcwd()))
    df_items = df_items.reset_index()
    
    iteration = 10000
    landa = 0.0000001
    teta = 0.0001

    iteration = 5
    landa = 0.0000001
    teta = 0.0001

    for u, row in df_users.iterrows():
        resp.append([0.5, 0.5])

    for i in range(iteration):
        print("iteration: ", i)
        for index_user, row in df_users.iterrows():
            w_u = resp[index_user][0]
            w_i = resp[index_user][1]
            rated_items = []

            for j in range(50):
                s = 'M' + str(j+1)
                if math.isnan(row[s]) == False:
                    rated_items.append([j, row[s]])

            for rate_item in rated_items:
                r_u_i = rate_item[1] #real rate
                #Compute prediction with current weights
                userId = i
                itemId = rate_item[0]
                r_u_i_prim = predict_rank(userId, itemId, df_users, df_items, w_i, w_u)

                #compare with real rating r_u_i and determine the error e_U_i
                e_u_i = r_u_i - r_u_i_prim
                #adjust w_u in gradient step
                w_u = w_u + (teta * (e_u_i - (landa * w_u)))
                #adjust w_i in gradient step
                w_i = w_i + (teta * (e_u_i - (landa * w_i)))
                resp[index_user][0] = w_u
                resp[index_user][1] = w_i


    return resp

def calculate_user_r(df, type, userId, itemId):
    
    matrix = matrix_persons_by_character
    if type == "r2":
        matrix = matrix_persons_by_movies_ranks

    neigh = NearestNeighbors(n_neighbors=5)
    neigh.fit(matrix)
    neigh_dist, neigh_ind = neigh.kneighbors([matrix[userId]])
    
    #میانگین رنک های داده شده کاربر جاری
    ra_AVG = 0
    for i, row in df.iterrows():
        if i == userId:
            ra_AVG = row["ra_AVG"]

    #print(ra_AVG)
    #مجموع شباهت های همسایه ها
    xigma_sim_A = 0
    xigma_sim_B = 0
    pred = 0

    for neigh_id in neigh_ind[0]:
        if userId != neigh_id:
            rb_AVG = 0
            rb_p = 0
            for i, row in df.iterrows():
                movieId = 'M' + str(itemId+1)
                if i == neigh_id and math.isnan(row[movieId]) == False:
                    rb_AVG = row["ra_AVG"]
                    rb_p = row[movieId]
                    A = matrix[userId, neigh_id] * (rb_p - rb_AVG)
                    B = matrix[userId, neigh_id]
                    xigma_sim_A = xigma_sim_A + A
                    xigma_sim_B = xigma_sim_B + B

            #print("neigh_id:", neigh_id)
            #todo
    
    #todo فخیخ از استاد سوال شود مخرج صفر شد چکار کنیم
    if xigma_sim_B != 0:
        pred = ra_AVG + (xigma_sim_A / xigma_sim_B)
    else:
        pred = ra_AVG
    #print("xigma_sim_B:", xigma_sim_B)
    #print("xigma_sim_B:", xigma_sim_B)
    #print("pred:", pred)

    return pred

def calculate_item_r(df, type, userId, itemId):
    matrix = matrix_movies_by_genre
    if type == "r2":
        matrix = matrix_movies_by_movies_ranks

    neigh = NearestNeighbors(n_neighbors=5)
    neigh.fit(matrix)
    neigh_dist, neigh_ind = neigh.kneighbors([matrix[itemId]])

    #میانگین رنک های داده شده فیلم جاری
    ra_AVG = 0
    for i, row in df.iterrows():
        if i == itemId:
            ra_AVG = row["ra_AVG"]

    #print(ra_AVG)
    #مجموع شباهت های همسایه ها
    xigma_sim_A = 0
    xigma_sim_B = 0
    pred = 0

    for neigh_id in neigh_ind[0]:
        if itemId != neigh_id:
            rb_AVG = 0
            rb_p = 0
            for i, row in df.iterrows():
                rankId = 'R' + str(userId+1)
                if i == neigh_id and math.isnan(row[rankId]) == False:
                    rb_AVG = row["ra_AVG"]
                    rb_p = row[rankId]
                    A = matrix[itemId, neigh_id] * (rb_p - rb_AVG)
                    B = matrix[itemId, neigh_id]
                    xigma_sim_A = xigma_sim_A + A
                    xigma_sim_B = xigma_sim_B + B
    
    #todo فخیخ از استاد سوال شود مخرج صفر شد چکار کنیم
    if xigma_sim_B != 0:
        pred = ra_AVG + (xigma_sim_A / xigma_sim_B)
    else:
        pred = ra_AVG

    #print("xigma_sim_B:", xigma_sim_B)
    #print("xigma_sim_B:", xigma_sim_B)
    #print("pred:", pred)

    return pred

def predict_rank(userId, itemId, df_users, df_items, w_item, w_user):
    user_r1 = calculate_user_r(df_users, "r1", userId, itemId)
    user_r2 = calculate_user_r(df_users, "r2", userId, itemId)
    user_r = 0

    if(user_r1 == 0 and user_r2 == 0):
        user_r = 0
    elif(user_r1 != 0 and user_r2 == 0):
        user_r = user_r1
    elif(user_r1 == 0 and user_r2 != 0):
        user_r = user_r2
    else:
        user_r = 2 * user_r1 * user_r2 / (user_r1 + user_r2)
    
    #print("user_r:", user_r, "user_r1:", user_r1, "user_r2:", user_r2)

    item_r1 = calculate_item_r(df_items, "r1", userId, itemId)
    item_r2 = calculate_item_r(df_items, "r2", userId, itemId)
    item_r = 0

    if(item_r1 == 0 and item_r2 == 0):
        item_r = 0
    elif(item_r1 != 0 and item_r2 == 0):
        item_r = item_r1
    elif(item_r1 == 0 and item_r2 != 0):
        item_r = item_r2
    else:
        item_r = 2 * item_r1 * item_r2 / (item_r1 + item_r2)
    
    #print("item_r:", item_r, "item_r1:", item_r1, "item_r2:", item_r2)
  
    final_rank = round(((w_item * item_r) + (w_user * user_r)))
    return final_rank

def calculate_all_ranks(weights):

    resultFileName = "FinalResult"
    path = "{0}\data\{1}.txt".format(os.getcwd(), resultFileName)
    file = open(path, "w")
    allScoresResultText = ""

    #users part
    df_users = pd.read_csv("{0}\data\data.csv".format(os.getcwd()))
    df_users = df_users.reset_index()
    
    df_items = pd.read_csv("{0}\data\movies.csv".format(os.getcwd()))
    df_items = df_items.reset_index()

    countIsNan = 0
    for i, row in df_users.iterrows():
        for j in range(50):
            s = 'M' + str(j+1)
            if math.isnan(row[s]):
                userId = i
                itemId = j
                w_user = weights[userId][0]
                w_item = weights[userId][1]
                
                final_rank = predict_rank(userId, itemId, df_users, df_items, w_item, w_user)
                
                str_template = """{0}
user:{1}, item:{2}, rank:{3}"""
                allScoresResultText = str_template.format(allScoresResultText, (userId + 1), (itemId + 1), final_rank)

                print("user:", userId, "item:", itemId, "rank:", final_rank)

                countIsNan = countIsNan + 1
                #print("IsNan:", userId, itemId)
        #todo remove break later
        #break
    
    #print("countIsNan:", countIsNan)
    
    file.write(allScoresResultText)
    file.close()

def predict_rated_items_rank(weights):
    resultFileName = "FinalResult2"
    path = "{0}\data\{1}.txt".format(os.getcwd(), resultFileName)
    file = open(path, "w")
    allScoresResultText = ""

    #users part
    df_users = pd.read_csv("{0}\data\data.csv".format(os.getcwd()))
    df_users = df_users.reset_index()
    
    df_items = pd.read_csv("{0}\data\movies.csv".format(os.getcwd()))
    df_items = df_items.reset_index()

    countIsNan = 0
    for i, row in df_users.iterrows():
        for j in range(50):
            s = 'M' + str(j+1)
            if math.isnan(row[s]) == False:
                
                real_rank = row[s]

                userId = i
                itemId = j
                w_user = weights[userId][0]
                w_item = weights[userId][1]
                final_rank = predict_rank(userId, itemId, df_users, df_items, w_item, w_user)
                
                str_template = """{0}
user:{1}, item:{2}, real_rank:{3}, predicted_rank:{4}"""
                allScoresResultText = str_template.format(
                    allScoresResultText, 
                    (userId + 1), 
                    (itemId + 1), 
                    round(real_rank),
                    final_rank)

                print("user:", userId, "item:", itemId,"real_rank:", round(real_rank), "predict_rank:", final_rank)

                countIsNan = countIsNan + 1
                #print("IsNan:", userId, itemId)
        #todo remove break later
        #break
    
    #print("countIsNan:", countIsNan)
    
    file.write(allScoresResultText)
    file.close()


weights = calculate_all_weights()
print("weights:", weights)

calculate_all_ranks(weights)
predict_rated_items_rank(weights)


