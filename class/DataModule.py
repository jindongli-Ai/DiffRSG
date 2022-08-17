#-*- coding: utf-8 -*-

from __future__ import division
from cmath import tau
from collections import defaultdict
import numpy as np
from time import time
import random
import tensorflow as tf

tau_uiu = 0.2
tau_ii = 0.3        #### yelp: 0.3   flickr: 0.7


class DataModule():
    def __init__(self, conf, filename):
        self.conf = conf                                                          
        self.data_dict = {}
        self.terminal_flag = 1
        self.filename = filename                                                                    
        self.index = 0

    #######  Initalize Procedures #######
    def prepareModelSupplement(self, model):
        data_dict = {}
        if 'CONSUMED_ITEMS_SPARSE_MATRIX' in model.supply_set:
            self.generateConsumedItemsSparseMatrix()
            #self.arrangePositiveData()
            data_dict['CONSUMED_ITEMS_INDICES_INPUT'] = self.consumed_items_indices_list
            data_dict['CONSUMED_ITEMS_VALUES_INPUT'] = self.consumed_items_values_list
            data_dict['CONSUMED_ITEMS_VALUES_WEIGHT_AVG_INPUT']  = self.consumed_items_values_weight_avg_list
            data_dict['CONSUMED_ITEMS_NUM_INPUT'] = self.consumed_item_num_list
            data_dict['CONSUMED_ITEMS_NUM_DICT_INPUT'] = self.user_item_num_dict
            # data_dict['USER_ITEM_SPARSITY_DICT'] = self.user_item_sparsity_dict           ##### 用不到！！！！！！

        if 'SOCIAL_NEIGHBORS_SPARSE_MATRIX' in model.supply_set:
            self.readSocialNeighbors()
            self.generateSocialNeighborsSparseMatrix()
            data_dict['SOCIAL_NEIGHBORS_INDICES_INPUT'] = self.social_neighbors_indices_list
            data_dict['SOCIAL_NEIGHBORS_VALUES_INPUT'] = self.social_neighbors_values_list
            data_dict['SOCIAL_NEIGHBORS_VALUES_WEIGHT_AVG_INPUT'] = self.social_neighbors_values_weight_avg_list
            data_dict['SOCIAL_NEIGHBORS_NUM_INPUT'] = self.social_neighbor_num_list
            data_dict['SOCIAL_NEIGHBORS_NUM_DICT_INPUT'] = self.social_neighbors_num_dict   
            # data_dict['USER_USER_SPARSITY_DICT']= self.user_user_sparsity_dict            #### 用不到！！！！！！
            ##################################################################
            ##################################################################
            ##################################################################
            data_dict['user_user_uiu_indices_input'] = self.user_user_uiu_indices_list
            data_dict['user_user_uiu_values_input'] = self.user_user_uiu_values_list
            data_dict['user_user_uiu_values_weight_avg_input'] = self.user_user_uiu_values_weight_avg_list
            data_dict['user_user_u0u_indices_input'] = self.user_user_u0u_indices_list
            data_dict['user_user_u0u_values_input'] = self.user_user_u0u_values_list
            data_dict['user_user_u0u_values_weight_avg_input'] = self.user_user_u0u_values_weight_avg_list

        if 'ITEM_CUSTOMER_SPARSE_MATRIX' in model.supply_set:
            self.generateConsumedItemsSparseMatrixForItemUser()      
            data_dict['ITEM_CUSTOMER_INDICES_INPUT'] = self.item_customer_indices_list
            data_dict['ITEM_CUSTOMER_VALUES_INPUT'] = self.item_customer_values_list 
            data_dict['ITEM_CUSTOMER_VALUES_WEIGHT_AVG_INPUT'] = self.item_customer_values_weight_avg_list 
            data_dict['ITEM_CUSTOMER_NUM_INPUT'] = self.item_customer_num_list
            data_dict['ITEM_USER_NUM_DICT_INPUT'] = self.item_user_num_dict
            ##################################################################
            ##################################################################
            ##################################################################
            data_dict['item_item_iui_indices_input'] = self.item_item_iui_indices_list
            data_dict['item_item_iui_values_input'] = self.item_item_iui_values_list
            data_dict['item_item_iui_values_weight_avg_input'] = self.item_item_iui_values_weight_avg_list
            data_dict['item_item_iui_num_input'] = self.item_item_iui_num_list
            data_dict['item_item_iui_num_dict_input'] = self.item_item_iui_num_dict

        return data_dict

    def initializeRankingTrain(self):
        self.readData()
        self.arrangePositiveData()
        self.arrangePositiveDataForItemUser() 
        self.generateTrainNegative()

    def initializeRankingVT(self):
        self.readData()
        self.arrangePositiveData()
        self.arrangePositiveDataForItemUser() 
        self.generateTrainNegative()

    def initalizeRankingEva(self):
        self.readData()
        self.getEvaPositiveBatch()
        self.generateEvaNegative()

    def linkedMap(self):
        self.data_dict['USER_LIST'] = self.user_list
        self.data_dict['ITEM_LIST'] = self.item_list
        self.data_dict['LABEL_LIST'] = self.labels_list
    
    def linkedRankingEvaMap(self):
        self.data_dict['EVA_USER_LIST'] = self.eva_user_list
        self.data_dict['EVA_ITEM_LIST'] = self.eva_item_list

    #######  Data Loading #######
    def readData(self):
        f = open(self.filename) 
        total_user_list = set()
        hash_data = defaultdict(int) 
        for _, line in enumerate(f):
            arr = line.split("\t")
            hash_data[(int(arr[0]), int(arr[1]))] = 1
            total_user_list.add(int(arr[0]))

        self.total_user_list = list(total_user_list)                                                        #### 
        self.hash_data = hash_data                                                                          #### hash_data[(userID, itemID)] = 1

    # def arrangePositiveData(self):                                                                        #### 
    #     positive_data = defaultdict(set)
    #     user_item_num_dict = defaultdict(set)
    #     total_data = set()
    #     hash_data = self.hash_data
    #     for (u, i) in hash_data:
    #         total_data.add((u, i))
    #         positive_data[u].add(i)

    #     user_list = sorted(list(positive_data.keys()))

    #     for u in range(self.conf.num_users):
    #         user_item_num_dict[u] = len(positive_data[u]) + 1
    #     self.positive_data = positive_data                                                                  #### positive_data 
    #     self.user_item_num_dict = user_item_num_dict                                                        #### 一个user
    #     self.user_item_num_for_sparsity_dict = user_item_num_for_sparsity_dict
    #     self.total_data = len(total_data)                                                                   ####（user-item）pair

    # def Sparsity_analysis_for_user_item_network(self):            #### 用不到！！！！！！！！！！！！！！！
    #     hash_data_for_user_item = self.hash_data
    #     sparisty_user_item_dict = {}

    def arrangePositiveDataForItemUser(self):
        positive_data_for_item_user = defaultdict(set)
        item_user_num_dict = defaultdict(int)

        total_data_for_item_user = set()
        hash_data_for_item_user = self.hash_data
        for (u, i) in hash_data_for_item_user:
            total_data_for_item_user.add((i, u))
            positive_data_for_item_user[i].add(u)

        # item_list = sorted(list(positive_data_for_item_user.keys()))

        for i in range(self.conf.num_items):
            item_user_num_dict[i] = len(positive_data_for_item_user[i]) + 1                                 #### 一个user

        self.item_user_num_dict = item_user_num_dict
        self.positive_data_for_item_user = positive_data_for_item_user                                      #### positive_data_for _item_user 
        self.total_data_for_item_user = len(total_data_for_item_user)                                       #### (item-user)pair

    
    # ----------------------
    # This function designes for generating train/val/test negative  
    def generateTrainNegative(self):
        num_items = self.conf.num_items
        num_negatives = self.conf.num_negatives
        negative_data = defaultdict(set)
        total_data = set()
        hash_data = self.hash_data                                  #### hash_data[(userID, itemID)] = 1
        for (u, i) in hash_data:
            total_data.add((u, i))
            for _ in range(num_negatives):                          #### 
                j = np.random.randint(num_items)                    #### 
                while (u, j) in hash_data:
                    j = np.random.randint(num_items)
                negative_data[u].add(j)
                total_data.add((u, j))                              #### ----
        self.negative_data = negative_data
        self.terminal_flag = 1                                      #### 
      

    # ----------------------
    # This function designes for val/test set, compute loss         #### 为validation/test
    def getVTRankingOneBatch(self):
        positive_data = self.positive_data
        negative_data = self.negative_data
        total_user_list = self.total_user_list
        user_list = []
        item_list = []
        labels_list = []
        for u in total_user_list:
            user_list.extend([u] * len(positive_data[u]))
            item_list.extend(positive_data[u])
            labels_list.extend([1] * len(positive_data[u]))         #### 
            user_list.extend([u] * len(negative_data[u]))
            item_list.extend(negative_data[u])
            labels_list.extend([0] * len(negative_data[u]))         #### 
        
        self.user_list = np.reshape(user_list, [-1, 1])             #### 
        self.item_list = np.reshape(item_list, [-1, 1])
        self.labels_list = np.reshape(labels_list, [-1, 1])
    
    # ----------------------
    # This function designes for the training process
    def getTrainRankingBatch(self):                                 #### 为train
        positive_data = self.positive_data
        negative_data = self.negative_data
        total_user_list = self.total_user_list
        index = self.index
        batch_size = self.conf.training_batch_size

        user_list, item_list, labels_list = [], [], []
        
        if index + batch_size < len(total_user_list):
            target_user_list = total_user_list[index:index+batch_size]
            self.index = index + batch_size
        else:
            target_user_list = total_user_list[index:len(total_user_list)]
            self.index = 0                                                                                      #### 
            self.terminal_flag = 0

        for u in target_user_list:
            user_list.extend([u] * len(positive_data[u]))
            item_list.extend(list(positive_data[u]))
            labels_list.extend([1] * len(positive_data[u]))         #### 
            user_list.extend([u] * len(negative_data[u]))
            item_list.extend(list(negative_data[u]))
            labels_list.extend([0] * len(negative_data[u]))         #### 
        
        self.user_list = np.reshape(user_list, [-1, 1])             #### 
        self.item_list = np.reshape(item_list, [-1, 1])
        self.labels_list = np.reshape(labels_list, [-1, 1])
    
    # ----------------------
    # This function is designed for the positive data
    def getEvaPositiveBatch(self):
        hash_data = self.hash_data
        user_list = []
        item_list = []
        index_dict = defaultdict(list)
        index = 0                                                   #### 
        for (u, i) in hash_data:
            user_list.append(u)
            item_list.append(i)
            index_dict[u].append(index)
            index = index + 1
        self.eva_user_list = np.reshape(user_list, [-1, 1])
        self.eva_item_list = np.reshape(item_list, [-1, 1])
        self.eva_index_dict = index_dict

    
    # ----------------------
    #This function is designed for generating negative data 
    def generateEvaNegative(self):
        hash_data = self.hash_data
        total_user_list = self.total_user_list
        num_evaluate = self.conf.num_evaluate
        num_items = self.conf.num_items
        eva_negative_data = defaultdict(list)                       #### 
        for u in total_user_list:
            for _ in range(num_evaluate):                           #### 
                j = np.random.randint(num_items)
                while (u, j) in hash_data:
                    j = np.random.randint(num_items)
                eva_negative_data[u].append(j)
        self.eva_negative_data = eva_negative_data

    # ----------------------
    #This function designs for generating negative batch in rating evaluation, 
    def getEvaRankingBatch(self):
        batch_size = self.conf.evaluate_batch_size
        num_evaluate = self.conf.num_evaluate
        eva_negative_data = self.eva_negative_data
        total_user_list = self.total_user_list
        index = self.index
        terminal_flag = 1
        total_users = len(total_user_list)
        user_list = []
        item_list = []
        if index + batch_size < total_users:                                #### 
            batch_user_list = total_user_list[index:index+batch_size]
            self.index = index + batch_size
        else:
            terminal_flag = 0
            batch_user_list = total_user_list[index:total_users]
            self.index = 0
        for u in batch_user_list:
            user_list.extend([u]*num_evaluate)                              #### 
            item_list.extend(eva_negative_data[u])
        self.eva_user_list = np.reshape(user_list, [-1, 1])                 #### 
        self.eva_item_list = np.reshape(item_list, [-1, 1])
        return batch_user_list, terminal_flag


    # ----------------------
    # Read social network information
    def readSocialNeighbors(self, friends_flag = 1):                        ####
        social_neighbors = defaultdict(set)
        social_neighbors_num_dict = defaultdict(int)

        links_file = open(self.conf.links_filename)                         #### 
        for _, line in enumerate(links_file):
            tmp = line.split('\t')
            u1, u2 = int(tmp[0]), int(tmp[1])
            social_neighbors[u1].add(u2)
            if friends_flag == 1:                                           ####
                social_neighbors[u2].add(u1)                                                                                            #### 无向
        # user_list = sorted(list(social_neighbors.keys()))                   #### ----
        for u in range(self.conf.num_users):
            social_neighbors_num_dict[u] = len(social_neighbors[u]) + 1     ##### 每个user

        self.social_neighbors_num_dict = social_neighbors_num_dict
        self.social_neighbors = social_neighbors

    def arrangePositiveData(self):                                          #### 
        positive_data = defaultdict(set)
        user_item_num_dict = defaultdict(int)
        total_data = set()
        hash_data = self.hash_data

        ############################################################################
        ############################################################################
        ############################################################################
        item_users = defaultdict(set)

        for (u, i) in hash_data:
            total_data.add((u, i))
            positive_data[u].add(i)
            item_users[i].add(u)

        user_list = sorted(list(positive_data.keys()))
        for u in range(self.conf.num_users):
            user_item_num_dict[u] = len(positive_data[u]) + 1

        self.positive_data = positive_data
        self.user_item_num_dict = user_item_num_dict                        #### 89
        self.total_data = len(total_data)

        #############################################################################
        #############################################################################
        #############################################################################
        self.user_items = self.positive_data
        self.item_users = item_users

    # ----------------------
    #Generate Social Neighbors Sparse Matrix Indices and Values
    def generateSocialNeighborsSparseMatrix(self):
        social_neighbors = self.social_neighbors                            ####
        social_neighbors_num_dict = self.social_neighbors_num_dict  #weight avg     #### 

        social_neighbors_indices_list = []
        social_neighbors_values_list = []
        social_neighbors_values_weight_avg_list = []
        social_neighbor_num_list = []
        social_neighbors_dict = defaultdict(list)

        # user_user_num_for_sparsity_dict = defaultdict(set)                    #### 
        # user_user_sparsity_dict = {}

        # user_user_sparsity_dict['0-4'] = []                                 #### 0 <= x < 4 （左闭右开）
        # user_user_sparsity_dict['4-8'] = []
        # user_user_sparsity_dict['8-16'] = []
        # user_user_sparsity_dict['16-32'] = []
        # user_user_sparsity_dict['32-64'] = []
        # user_user_sparsity_dict['64-'] = []
  
        # for u in range(self.conf.num_users):                                  
        #     user_user_num_for_sparsity_dict[u] = len(social_neighbors[u])   #### 

        for u in social_neighbors:
            social_neighbors_dict[u] = sorted(social_neighbors[u])
            
        user_list = sorted(list(social_neighbors.keys()))

        #node att                                                                                                           #### 
        for user in range(self.conf.num_users):
            if user in social_neighbors_dict:
                social_neighbor_num_list.append(len(social_neighbors_dict[user]))           #### 
            else:
                social_neighbor_num_list.append(1)                                          #### 
                print("{} 没有trust的朋友！！！！！！！！！！！".format(user))                                    #### 
        
        for user in user_list:
            for friend in social_neighbors_dict[user]:
                social_neighbors_indices_list.append([user, friend])
                social_neighbors_values_list.append(1.0/len(social_neighbors_dict[user]))
                social_neighbors_values_weight_avg_list.append(1.0 / ( np.sqrt(social_neighbors_num_dict[user]) * np.sqrt(social_neighbors_num_dict[friend]) ) )  #weight avg
   
        # for u in range(self.conf.num_users):
        #     cur_user_neighbors_num = user_user_num_for_sparsity_dict[u]                     #### 
        #     if( (cur_user_neighbors_num >=0) & (cur_user_neighbors_num<4) ):
        #         user_user_sparsity_dict['0-4'].append(u)
        #     elif( (cur_user_neighbors_num >=4) & (cur_user_neighbors_num<8) ):
        #         user_user_sparsity_dict['4-8'].append(u)
        #     elif( (cur_user_neighbors_num >=8) & (cur_user_neighbors_num<16) ):
        #         user_user_sparsity_dict['8-16'].append(u)
        #     elif( (cur_user_neighbors_num >=16) & (cur_user_neighbors_num<32) ):
        #         user_user_sparsity_dict['16-32'].append(u)
        #     elif( (cur_user_neighbors_num >=32) & (cur_user_neighbors_num<64) ):
        #         user_user_sparsity_dict['32-64'].append(u)                
        #     elif( cur_user_neighbors_num >=64):
        #         user_user_sparsity_dict['64-'].append(u)


        # self.user_user_sparsity_dict = user_user_sparsity_dict                              #### 
        self.social_neighbors_indices_list = np.array(social_neighbors_indices_list).astype(np.int64)
        self.social_neighbors_values_list = np.array(social_neighbors_values_list).astype(np.float32)
        self.social_neighbors_values_weight_avg_list = np.array(social_neighbors_values_weight_avg_list).astype(np.float32)   # weight avg
        self.social_neighbor_num_list = np.array(social_neighbor_num_list).astype(np.int64)
        #self.social_neighbors_values_list = tf.Variable(tf.random_normal([len(self.social_neighbors_indices_list)], stddev=0.01))      #### 


        ################################################################################################
        ################################################################################################
        ################################################################################################
        #### u1与u2之间有共同的item
        user_user_uiu_indices_list = []
        user_user_uiu_indeces_visited = set()                           ####
        user_user_uiu_values_list = []
        user_user_uiu_values_weight_avg_list = []

        #### u1与u2之间没有共同的item
        user_user_u0u_indices_list = []
        user_user_u0u_indices_visited = set()
        user_user_u0u_values_list = []
        user_user_u0u_values_weight_avg_list = []


        print('此时的user-user阈值是{}'.format(tau_uiu))

        ID = 0

        for u1 in user_list:
            u1_items_set = set(self.user_items[u1])
            # print(u1_items_set)
            for u2 in social_neighbors_dict[u1]:
                u1_u2_common_item = set()                    ####
                u2_items_set = set(self.user_items[u2])
                for i2 in u2_items_set:
                    if i2 in u1_items_set:
                        u1_u2_common_item.add(i2)
                # print(have_common_item)
                if len(u1_u2_common_item) > 0:
                    if (u1, u2) in user_user_uiu_indeces_visited:
                        continue                                        #### 
                    user_user_uiu_indices_list.append([u1, u2])
                    user_user_uiu_indeces_visited.add((u1, u2))         #### 
                    user_user_uiu_values_list.append(1.0/len(social_neighbors_dict[u1]))
                    user_user_uiu_values_weight_avg_list.append(1.0 / ( np.sqrt(social_neighbors_num_dict[u1]) * np.sqrt(social_neighbors_num_dict[u2]) ) )  #weight avg
                    ###################################################################################################
                    ###################################################################################################
                    ###################################################################################################
                    if len(u1_u2_common_item) / len(u1_items_set) >= tau_uiu and len(u1_u2_common_item) / len(u2_items_set) >= tau_uiu:
                        for u3 in social_neighbors_dict[u2]:
                            if u1 == u3 or u1 == u2 or u2 == u3:
                                continue
                            for i3 in self.user_items[u3]:
                                if i3 in u1_u2_common_item:
                                    if (u1, u3) in user_user_uiu_indeces_visited:
                                        continue                                        #### 
                                    user_user_uiu_indices_list.append([u1, u3])
                                    user_user_uiu_indeces_visited.add((u1, u3))         ####
                                    user_user_uiu_values_list.append(1.0/len(social_neighbors_dict[u1]))
                                    user_user_uiu_values_weight_avg_list.append(1.0 / ( np.sqrt(social_neighbors_num_dict[u1]) * np.sqrt(social_neighbors_num_dict[u3]) ) )  #weight avg
                                    # print(ID)
                                    ID += 1
                                    # for u4 in social_neighbors_dict[u3]:
                                    #     if len(set([u1, u2, u3, u4])) != 4:
                                    #         continue
                                    #     for i4 in self.user_items[u4]:
                                    #         if i4 in u1_items_set:
                                    #             user_user_uiu_indices_list.append([u1, u4])
                                    #             user_user_uiu_values_list.append(1.0/len(social_neighbors_dict[u1]))
                                    #             user_user_uiu_values_weight_avg_list.append(1.0 / ( np.sqrt(social_neighbors_num_dict[u1]) * np.sqrt(social_neighbors_num_dict[u4]) ) )  #weight avg
                                    #             break
                                    break
                else:
                    if (u1, u2) in user_user_u0u_indices_visited:
                        continue                                                        #### 
                    user_user_u0u_indices_list.append([u1, u2])
                    user_user_u0u_indices_visited.add((u1, u2))                         #### 
                    user_user_u0u_values_list.append(1.0/len(social_neighbors_dict[u1]))
                    user_user_u0u_values_weight_avg_list.append(1.0 / ( np.sqrt(social_neighbors_num_dict[u1]) * np.sqrt(social_neighbors_num_dict[u2]) ) )  #weight avg
                    
        print(ID)

        self.user_user_uiu_indices_list = np.array(user_user_uiu_indices_list).astype(np.int64)
        self.user_user_uiu_values_list = np.array(user_user_uiu_values_list).astype(np.float32)
        self.user_user_uiu_values_weight_avg_list = np.array(user_user_uiu_values_weight_avg_list).astype(np.float32)
        self.user_user_u0u_indices_list = np.array(user_user_u0u_indices_list).astype(np.int64)
        self.user_user_u0u_values_list = np.array(user_user_u0u_values_list).astype(np.float32)
        self.user_user_u0u_values_weight_avg_list = np.array(user_user_u0u_values_weight_avg_list).astype(np.float32)

    # ----------------------
    #Generate Consumed Items Sparse Matrix Indices and Values
    def generateConsumedItemsSparseMatrix(self):
        positive_data = self.positive_data  
        consumed_items_indices_list = []
        consumed_items_values_list = []
        consumed_items_values_weight_avg_list = []
        consumed_item_num_list = []
        consumed_items_dict = defaultdict(list)
        # user_item_num_for_sparsity_dict = defaultdict(set)                                    ##### 
        # user_item_sparsity_dict = {}

        # user_item_sparsity_dict['0-4'] = []                                                   #### 
        # user_item_sparsity_dict['4-8'] = []
        # user_item_sparsity_dict['8-16'] = []
        # user_item_sparsity_dict['16-32'] = []
        # user_item_sparsity_dict['32-64'] = []
        # user_item_sparsity_dict['64-'] = []
        
        consumed_items_num_dict = self.user_item_num_dict   #weight avg
        #social_neighbors_num_dict = self.social_neighbors_num_dict  #weight avg
        item_user_num_dict = self.item_user_num_dict  #weight avg

        for u in positive_data:
            consumed_items_dict[u] = sorted(positive_data[u])                                   #### 

        user_list = sorted(list(positive_data.keys()))                                          #### userID

        # for u in range(self.conf.num_users):
        #     user_item_num_for_sparsity_dict[u] = len(positive_data[u])                          #### 
        
        for user in range(self.conf.num_users):
            if user in consumed_items_dict:
                consumed_item_num_list.append(len(consumed_items_dict[user]))                   #### 
            else:
                consumed_item_num_list.append(1)                                                #### 

        for u in user_list:
            for i in consumed_items_dict[u]:
                consumed_items_indices_list.append([u, i])
                consumed_items_values_list.append(1.0/len(consumed_items_dict[u]))
                consumed_items_values_weight_avg_list.append(1.0/(  np.sqrt(consumed_items_num_dict[u]) *  np.sqrt(item_user_num_dict[i])  ))  #weight avg

        # for u in range(self.conf.num_users):                                                  #### 
        #     cur_user_consumed_item_num = user_item_num_for_sparsity_dict[u]
        #     if( (cur_user_consumed_item_num >=0) & (cur_user_consumed_item_num<4) ):
        #         user_item_sparsity_dict['0-4'].append(u)
        #     elif( (cur_user_consumed_item_num >=4) & (cur_user_consumed_item_num<8) ):
        #         user_item_sparsity_dict['4-8'].append(u)
        #     elif( (cur_user_consumed_item_num >=8) & (cur_user_consumed_item_num<16) ):
        #         user_item_sparsity_dict['8-16'].append(u)
        #     elif( (cur_user_consumed_item_num >=16) & (cur_user_consumed_item_num<32) ):
        #         user_item_sparsity_dict['16-32'].append(u)
        #     elif( (cur_user_consumed_item_num >=32) & (cur_user_consumed_item_num<64) ):
        #         user_item_sparsity_dict['32-64'].append(u)
        #     elif( cur_user_consumed_item_num >=64):
        #         user_item_sparsity_dict['64-'].append(u)

        # self.user_item_sparsity_dict = user_item_sparsity_dict                                  #### 
        self.consumed_items_indices_list = np.array(consumed_items_indices_list).astype(np.int64)
        self.consumed_items_values_list = np.array(consumed_items_values_list).astype(np.float32)
        self.consumed_items_values_weight_avg_list = np.array(consumed_items_values_weight_avg_list).astype(np.float32)   #weight avg
        self.consumed_item_num_list = np.array(consumed_item_num_list).astype(np.int64)



    def generateConsumedItemsSparseMatrixForItemUser(self):
        positive_data_for_item_user = self.positive_data_for_item_user  
        item_customer_indices_list = []
        item_customer_values_list = []
        item_customer_values_weight_avg_list = []
        item_customer_num_list = []
        item_customer_dict = defaultdict(list)

        consumed_items_num_dict = self.user_item_num_dict   #weight avg                             #### 
        #social_neighbors_num_dict = self.social_neighbors_num_dict  #weight avg
        item_user_num_dict = self.item_user_num_dict  #weight avg                                   #### 

        for i in positive_data_for_item_user:                                                       #### 
            item_customer_dict[i] = sorted(positive_data_for_item_user[i])
        item_list = sorted(list(positive_data_for_item_user.keys()))

        for item in range(self.conf.num_items):
            if item in item_customer_dict:
                item_customer_num_list.append(len(item_customer_dict[item]))                        #### 
            else:
                item_customer_num_list.append(1)                                                    #### 
        
        for i in item_list:
            for u in item_customer_dict[i]:
                item_customer_indices_list.append([i, u])
                item_customer_values_list.append(1.0/len(item_customer_dict[i]))                    #### 
                item_customer_values_weight_avg_list.append(1.0/( np.sqrt(consumed_items_num_dict[u]) *  np.sqrt(item_user_num_dict[i])  ))     #
      
        self.item_customer_indices_list = np.array(item_customer_indices_list).astype(np.int64)
        self.item_customer_values_list = np.array(item_customer_values_list).astype(np.float32)
        self.item_customer_num_list = np.array(item_customer_num_list).astype(np.int64)
        self.item_customer_values_weight_avg_list = np.array(item_customer_values_weight_avg_list).astype(np.float32)




        ################################################################################################
        ################################################################################################
        ################################################################################################
        #### u1与u2之间有共同的item
        item_item_iui_indices_list = []
        item_item_iui_values_list = []
        item_item_iui_num_list = []
        item_item_iui_values_weight_avg_list = []
        self.item_items = defaultdict(set)
        self.item_item_iui_num_dict = defaultdict(int)

        ######## item-item
        print('此时item-item阈值是{}'.format(tau_ii))

        ID = 0
        for i1 in item_list:
            i1_users_set = set(self.item_users[i1])
            for u1 in i1_users_set:
                for i2 in self.user_items[u1]:
                    if i1 == i2:
                        continue
                    i2_users_set = set(self.item_users[i2])
                    common_user_set = set()
                    for u2 in i2_users_set:
                        if u2 in i1_users_set:
                            common_user_set.add(u2)
                    if len(common_user_set) / len(i1_users_set) >= tau_ii and len(common_user_set) / len(i2_users_set) >= tau_ii:
                    # if len(common_user_set) / len(i1_users_set) >= 0.5 and len(common_user_set) / len(i2_users_set) >= 0.5:
                        # print(ID)                               ##########################################################
                        ID += 1
                        self.item_items[i1].add(i2)
        print(ID)

        for i1 in range(self.conf.num_items):
            if i1 in self.item_items:
                self.item_item_iui_num_dict[i1] = len(self.item_items[i1]) + 1
            else:
                self.item_item_iui_num_dict[i1] = 1


        for i1 in range(self.conf.num_items):
            if i1 in self.item_items:
                item_item_iui_num_list.append(len(self.item_items[i1]))
            else:
                item_item_iui_num_list.append(1)                                       


        for i1 in self.item_items.keys():
            for i2 in self.item_items[i1]:
                item_item_iui_indices_list.append([i1, i2])
                item_item_iui_values_list.append(1.0 / len(self.item_items[i1]))
                item_item_iui_values_weight_avg_list.append(1.0 / ( np.sqrt(self.item_item_iui_num_dict[i1]) * np.sqrt(self.item_item_iui_num_dict[i2]) ) )  #weight avg

                               
        self.item_item_iui_indices_list = np.array(item_item_iui_indices_list).astype(np.int64)
        self.item_item_iui_values_list = np.array(item_item_iui_values_list).astype(np.float32)
        self.item_item_iui_num_list = np.array(item_item_iui_num_list).astype(np.int64)
        self.item_item_iui_values_weight_avg_list = np.array(item_item_iui_values_weight_avg_list).astype(np.float32)

