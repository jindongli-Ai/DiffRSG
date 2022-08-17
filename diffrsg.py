# -*- coding: utf-8 -*-

from __future__ import division
from io import SEEK_CUR
from re import U
from tarfile import USTAR_FORMAT
from time import sleep
from unittest.util import unorderable_list_difference
import uu
from xmlrpc.client import _iso8601_format
import tensorflow as tf
import numpy as np

class DiffRSG():
    def __init__(self, conf):
        self.conf = conf
        self.supply_set = (
            'SOCIAL_NEIGHBORS_SPARSE_MATRIX',
            'CONSUMED_ITEMS_SPARSE_MATRIX',
            'ITEM_CUSTOMER_SPARSE_MATRIX'
        )

    def startConstructGraph(self):
        self.initializeNodes()
        self.constructTrainGraph()
        self.saveVariables()
        self.defineMap()


    def inputSupply(self, data_dict):

        self.social_neighbors_indices_input = data_dict['SOCIAL_NEIGHBORS_INDICES_INPUT']                   #### [user, frient] list转成的numpy.array()
        self.social_neighbors_values_input = data_dict['SOCIAL_NEIGHBORS_VALUES_INPUT']                     #### [1.0/len(social_neighbors_dict[user]] list转成的numpy.array()

        ###########################################################################################################
        ###########################################################################################################
        ###########################################################################################################
        self.user_user_uiu_indices_input = data_dict['user_user_uiu_indices_input']                         ######## uiu
        self.user_user_uiu_values_input = data_dict['user_user_uiu_values_input']
        self.user_user_u0u_indices_input = data_dict['user_user_u0u_indices_input']                         ######## u0u
        self.user_user_u0u_values_input = data_dict['user_user_u0u_values_input']

        self.item_item_indices_input = data_dict['item_item_iui_indices_input']
        self.item_item_values_input = data_dict['item_item_iui_values_input']


        low_att_std = 1.0               #### 正态分布的标准差
        ########  Node Attention initialization ########                #### 

        # ----------------------
        ##################################
        # user-user social network node attention initialization
                            ######## 第1层
        self.first_low_att_layer_for_social_neighbors_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='first_low_att_SN_layer1')
        self.first_low_att_layer_for_social_neighbors_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='first_low_att_SN_layer2')
                                        #### 输出空间的维度是1
        self.social_neighbors_values_input1 = tf.reduce_sum(tf.math.exp(self.first_low_att_layer_for_social_neighbors_layer1( \
                                            tf.reshape(tf.Variable( tf.random_normal([len(self.social_neighbors_indices_input)], stddev=low_att_std) ),[-1,1])   )   ), axis=1 )
                                        #### 
                                        #### 
                                        #### 
                                        #### 
                                        ####   
        # print('-----------------------------------------------------')
        # print(tf.math.exp(self.first_low_att_layer_for_social_neighbors_layer1( \
        #                                     tf.reshape(tf.Variable( tf.random_normal([len(self.social_neighbors_indices_input)], stddev=low_att_std) ),[-1,1])   )   ))
        # print('---------------------------------------------------------')
        # print(self.social_neighbors_values_input1)
        # print('=========================================================')
        ###############################################################################################################
        ###############################################################################################################
        ###############################################################################################################
        self.first_low_att_layer_for_user_user_uiu_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='first_low_att_uiu_layer1')
        self.user_user_uiu_values_input1 = tf.reduce_sum(tf.math.exp(self.first_low_att_layer_for_user_user_uiu_layer1( \
                                            tf.reshape(tf.Variable( tf.random_normal([len(self.user_user_uiu_indices_input)], stddev=low_att_std) ),[-1,1])   )   ), axis=1 )
        self.first_low_att_layer_for_user_user_u0u_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='first_low_att_u0u_layer1')
        self.user_user_u0u_values_input1 = tf.reduce_sum(tf.math.exp(self.first_low_att_layer_for_user_user_u0u_layer1( \
                                            tf.reshape(tf.Variable( tf.random_normal([len(self.user_user_u0u_indices_input)], stddev=low_att_std) ),[-1,1])   )   ), axis=1 )

                            ######## 第1层的均值和方差                 
        # first_mean_social_influ, first_var_social_influ = tf.nn.moments(self.social_neighbors_values_input1, axes=0)
        #                                 #### tf.nn.moments() 用于计算均值和方差
        # self.first_user_user_low_att = [first_mean_social_influ, first_var_social_influ]

                            ####
                            ####
                            ######## 第2层
        self.second_low_att_layer_for_social_neighbors_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='second_low_att_SN_layer1')
        self.second_low_att_layer_for_social_neighbors_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='second_low_att_SN_layer2')
        self.social_neighbors_values_input2 = tf.reduce_sum(tf.math.exp(self.second_low_att_layer_for_social_neighbors_layer1( \
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.social_neighbors_indices_input)], stddev=1.0)),[-1,1])      )   ), 1)
        ###############################################################################################################
        ###############################################################################################################
        ###############################################################################################################
        self.second_low_att_layer_for_user_user_uiu_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='second_low_att_uiu_layer1')
        self.user_user_uiu_values_input2 = tf.reduce_sum(tf.math.exp(self.second_low_att_layer_for_user_user_uiu_layer1( \
                                            tf.reshape(tf.Variable( tf.random_normal([len(self.user_user_uiu_indices_input)], stddev=low_att_std) ),[-1,1])   )   ), axis=1 )
        self.second_low_att_layer_for_user_user_u0u_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='second_low_att_u0u_layer1')
        self.user_user_u0u_values_input2 = tf.reduce_sum(tf.math.exp(self.second_low_att_layer_for_user_user_u0u_layer1( \
                                            tf.reshape(tf.Variable( tf.random_normal([len(self.user_user_u0u_indices_input)], stddev=low_att_std) ),[-1,1])   )   ), axis=1 )        

                            ####
                            ####
                            ######## 第3层
        # self.social_neighbors_values_input3 = tf.Variable(tf.random_normal([len(self.social_neighbors_indices_input)], stddev=0.01))
        # self.social_neighbors_num_input = 1.0/np.reshape(data_dict['SOCIAL_NEIGHBORS_NUM_INPUT'],[-1,1])            
                                        #### 

        self.third_low_att_layer_for_social_neighbors_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='third_low_att_SN_layer1')
        self.third_low_att_layer_for_social_neighbors_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='third_low_att_SN_layer2')
        self.social_neighbors_values_input3 = tf.reduce_sum(tf.math.exp(self.third_low_att_layer_for_social_neighbors_layer1( \
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.social_neighbors_indices_input)], stddev=1.0)),[-1,1])      )   ), 1)
        ###############################################################################################################
        ###############################################################################################################
        ###############################################################################################################
        self.third_low_att_layer_for_user_user_uiu_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='third_low_att_uiu_layer1')
        self.user_user_uiu_values_input3 = tf.reduce_sum(tf.math.exp(self.third_low_att_layer_for_user_user_uiu_layer1( \
                                            tf.reshape(tf.Variable( tf.random_normal([len(self.user_user_uiu_indices_input)], stddev=low_att_std) ),[-1,1])   )   ), axis=1 )
        self.third_low_att_layer_for_user_user_u0u_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='third_low_att_u0u_layer1')
        self.user_user_u0u_values_input3 = tf.reduce_sum(tf.math.exp(self.third_low_att_layer_for_user_user_u0u_layer1( \
                                            tf.reshape(tf.Variable( tf.random_normal([len(self.user_user_u0u_indices_input)], stddev=low_att_std) ),[-1,1])   )   ), axis=1 )    



                            ####
                            ####
                            ######## 第4层
        # self.social_neighbors_values_input4 = tf.Variable(tf.random_normal([len(self.social_neighbors_indices_input)], stddev=0.01))
        # self.social_neighbors_num_input = 1.0/np.reshape(data_dict['SOCIAL_NEIGHBORS_NUM_INPUT'],[-1,1])            
                                        #### 朋友的个数

        self.fourth_low_att_layer_for_social_neighbors_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='fourth_low_att_SN_layer1')
        self.fourth_low_att_layer_for_social_neighbors_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='fourth_low_att_SN_layer2')
        self.social_neighbors_values_input4 = tf.reduce_sum(tf.math.exp(self.fourth_low_att_layer_for_social_neighbors_layer1( \
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.social_neighbors_indices_input)], stddev=1.0)),[-1,1])      )   ), 1)
        ###############################################################################################################
        ###############################################################################################################
        ###############################################################################################################
        self.fourth_low_att_layer_for_user_user_uiu_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='fourth_low_att_uiu_layer1')
        self.user_user_uiu_values_input4 = tf.reduce_sum(tf.math.exp(self.fourth_low_att_layer_for_user_user_uiu_layer1( \
                                            tf.reshape(tf.Variable( tf.random_normal([len(self.user_user_uiu_indices_input)], stddev=low_att_std) ),[-1,1])   )   ), axis=1 )
        self.fourth_low_att_layer_for_user_user_u0u_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='fourth_low_att_u0u_layer1')
        self.user_user_u0u_values_input4 = tf.reduce_sum(tf.math.exp(self.fourth_low_att_layer_for_user_user_u0u_layer1( \
                                            tf.reshape(tf.Variable( tf.random_normal([len(self.user_user_u0u_indices_input)], stddev=low_att_std) ),[-1,1])   )   ), axis=1 )    


                            ####
                            ####
                            ######## 第5层
        # self.social_neighbors_values_input5 = tf.Variable(tf.random_normal([len(self.social_neighbors_indices_input)], stddev=0.01))
        # self.social_neighbors_num_input = 1.0/np.reshape(data_dict['SOCIAL_NEIGHBORS_NUM_INPUT'],[-1,1])            
                                        #### 朋友的个

        self.fifth_low_att_layer_for_social_neighbors_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='fifth_low_att_SN_layer1')
        self.fifth_low_att_layer_for_social_neighbors_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='fifth_low_att_SN_layer2')
        self.social_neighbors_values_input5 = tf.reduce_sum(tf.math.exp(self.fifth_low_att_layer_for_social_neighbors_layer1( \
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.social_neighbors_indices_input)], stddev=1.0)),[-1,1])      )   ), 1)
        ###############################################################################################################
        ###############################################################################################################
        ###############################################################################################################
        self.fifth_low_att_layer_for_user_user_uiu_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='fifth_low_att_uiu_layer1')
        self.user_user_uiu_values_input5 = tf.reduce_sum(tf.math.exp(self.fifth_low_att_layer_for_user_user_uiu_layer1( \
                                            tf.reshape(tf.Variable( tf.random_normal([len(self.user_user_uiu_indices_input)], stddev=low_att_std) ),[-1,1])   )   ), axis=1 )
        self.fifth_low_att_layer_for_user_user_u0u_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='fifth_low_att_u0u_layer1')
        self.user_user_u0u_values_input5 = tf.reduce_sum(tf.math.exp(self.fifth_low_att_layer_for_user_user_u0u_layer1( \
                                            tf.reshape(tf.Variable( tf.random_normal([len(self.user_user_u0u_indices_input)], stddev=low_att_std) ),[-1,1])   )   ), axis=1 )    


        # ----------------------
        # self.user_item_sparsity_dict = data_dict['USER_ITEM_SPARSITY_DICT']                     #### 
        self.consumed_items_indices_input = data_dict['CONSUMED_ITEMS_INDICES_INPUT']           #### [user, item]
        self.consumed_items_values_input = data_dict['CONSUMED_ITEMS_VALUES_INPUT']             #### 

        ##################################
        # user-item interest graph node attention initialization
                            ####
                            ######## 第1层
        self.first_low_att_layer_for_user_item_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='first_low_att_UI_layer1')
        self.first_low_att_layer_for_user_item_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='first_low_att_UI_layer2')
        #self.consumed_items_values_input1 = tf.Variable(tf.random_normal([len(self.consumed_items_indices_input)], stddev=0.01))
        self.consumed_items_values_input1 = tf.reduce_sum(tf.math.exp(self.first_low_att_layer_for_user_item_layer1( \
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.consumed_items_indices_input)], stddev=low_att_std)),[-1,1])  )   ),1)
                            ######## 第1层的均值和方差
        # first_mean_social_influ, first_var_social_influ = tf.nn.moments(self.consumed_items_values_input1, axes=0)
        # self.first_user_item_low_att = [first_mean_social_influ, first_var_social_influ]
                                        
                            ####
                            ######## 第2层
        self.second_low_att_layer_for_user_item_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='second_low_att_UI_layer1')
        self.second_low_att_layer_for_user_item_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='second_low_att_UI_layer2')
        #self.consumed_items_values_input2 = tf.Variable(tf.random_normal([len(self.consumed_items_indices_input)], stddev=0.01))
        self.consumed_items_values_input2 = tf.reduce_sum(tf.math.exp(self.second_low_att_layer_for_user_item_layer1( \
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.consumed_items_indices_input)], stddev=1.0)),[-1,1])    )   ),1)


        #                     ######## 第3层
        # self.consumed_items_values_input3 = tf.Variable(tf.random_normal([len(self.consumed_items_indices_input)], stddev=0.01))
        # self.consumed_items_num_input = 1.0/np.reshape(data_dict['CONSUMED_ITEMS_NUM_INPUT'], [-1,1])
        #                                 #### 
                            ####
                            ######## 第3层
        self.third_low_att_layer_for_user_item_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='third_low_att_UI_layer1')
        self.third_low_att_layer_for_user_item_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='third_low_att_UI_layer2')
        #self.consumed_items_values_input3 = tf.Variable(tf.random_normal([len(self.consumed_items_indices_input)], stddev=0.01))
        self.consumed_items_values_input3 = tf.reduce_sum(tf.math.exp(self.third_low_att_layer_for_user_item_layer1( \
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.consumed_items_indices_input)], stddev=1.0)),[-1,1])    )   ),1)

                            ####
                            ######## 第4层
        self.fourth_low_att_layer_for_user_item_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='fourth_low_att_UI_layer1')
        self.fourth_low_att_layer_for_user_item_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='fourth_low_att_UI_layer2')
        #self.consumed_items_values_input4 = tf.Variable(tf.random_normal([len(self.consumed_items_indices_input)], stddev=0.01))
        self.consumed_items_values_input4 = tf.reduce_sum(tf.math.exp(self.fourth_low_att_layer_for_user_item_layer1( \
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.consumed_items_indices_input)], stddev=1.0)),[-1,1])    )   ),1)

                            ####
                            ######## 第5层
        self.fifth_low_att_layer_for_user_item_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='fifth_low_att_UI_layer1')
        self.fifth_low_att_layer_for_user_item_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='fifth_low_att_UI_layer2')
        #self.consumed_items_values_input4 = tf.Variable(tf.random_normal([len(self.consumed_items_indices_input)], stddev=0.01))
        self.consumed_items_values_input5 = tf.reduce_sum(tf.math.exp(self.fifth_low_att_layer_for_user_item_layer1( \
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.consumed_items_indices_input)], stddev=1.0)),[-1,1])    )   ),1)


        # ----------------------
        self.item_customer_indices_input = data_dict['ITEM_CUSTOMER_INDICES_INPUT']         #### [item, user] 列表
        self.item_customer_values_input = data_dict['ITEM_CUSTOMER_VALUES_INPUT']           ####

        # item-user graph node attention initialization
                            ####
                            ######## 第1层
        self.first_low_att_layer_for_item_user_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='first_low_att_IU_layer1')
        self.first_low_att_layer_for_item_user_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='first_low_att_IU_layer2')

        #self.item_customer_values_input1 = tf.Variable(tf.random_normal([len(self.item_customer_indices_input)], stddev=0.01))
        self.item_customer_values_input1 = tf.reduce_sum(tf.math.exp(self.first_low_att_layer_for_item_user_layer1( \
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.item_customer_indices_input)], stddev=low_att_std)),[-1,1])    )   ),1)

                            ######## 第1层的均值和方差
        # first_mean_social_influ, first_var_social_influ = tf.nn.moments(self.item_customer_values_input1, axes=0)
        # self.first_item_user_low_att = [first_mean_social_influ, first_var_social_influ]

                            ####
                            ######## 第2层
        self.second_low_att_layer_for_item_user_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='second_low_att_IU_layer1')
        self.second_low_att_layer_for_item_user_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='second_low_att_IU_layer2')
        #self.item_customer_values_input2 = tf.Variable(tf.random_normal([len(self.item_customer_indices_input)], stddev=0.01))
        self.item_customer_values_input2 = tf.reduce_sum(tf.math.exp(self.second_low_att_layer_for_item_user_layer1( \
                                   
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.item_customer_indices_input)], stddev=0.01)),[-1,1]) )   ),1)
        #                     ######## 第3层
        # self.item_customer_values_input3 = tf.Variable(tf.random_normal([len(self.item_customer_indices_input)], stddev=0.01))
        # self.item_customer_num_input = 1.0/np.reshape(data_dict['ITEM_CUSTOMER_NUM_INPUT'],[-1,1])

                            ####
                            ######## 第3层
        self.third_low_att_layer_for_item_user_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='third_low_att_IU_layer1')
        self.third_low_att_layer_for_item_user_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='third_low_att_IU_layer2')
        #self.item_customer_values_input3 = tf.Variable(tf.random_normal([len(self.item_customer_indices_input)], stddev=0.01))
        self.item_customer_values_input3 = tf.reduce_sum(tf.math.exp(self.third_low_att_layer_for_item_user_layer1( \
                               
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.item_customer_indices_input)], stddev=0.01)),[-1,1]) )   ),1)

                            ####
                            ######## 第4层
        self.fourth_low_att_layer_for_item_user_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='fourth_low_att_IU_layer1')
        self.fourth_low_att_layer_for_item_user_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='fourth_low_att_IU_layer2')
        #self.item_customer_values_input4 = tf.Variable(tf.random_normal([len(self.item_customer_indices_input)], stddev=0.01))
        self.item_customer_values_input4 = tf.reduce_sum(tf.math.exp(self.fourth_low_att_layer_for_item_user_layer1( \
                               
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.item_customer_indices_input)], stddev=0.01)),[-1,1]) )   ),1)

                            ####
                            ######## 第5层
        self.fifth_low_att_layer_for_item_user_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='fifth_low_att_IU_layer1')
        self.fifth_low_att_layer_for_item_user_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='fifth_low_att_IU_layer2')
        #self.item_customer_values_input4 = tf.Variable(tf.random_normal([len(self.item_customer_indices_input)], stddev=0.01))
        self.item_customer_values_input5 = tf.reduce_sum(tf.math.exp(self.fifth_low_att_layer_for_item_user_layer1( \
                               
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.item_customer_indices_input)], stddev=0.01)),[-1,1]) )   ),1)




        #############################################################################################################
        #############################################################################################################
        #############################################################################################################
        #--------------------------- item-item
        # item-user graph node attention initialization
                            ######## 第1层
        self.first_low_att_layer_for_item_item_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='first_low_att_II_layer1')
        self.first_low_att_layer_for_item_item_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='first_low_att_II_layer2')

        #self.item_item_values_input1 = tf.Variable(tf.random_normal([len(self.item_item_indices_input)], stddev=0.01))
        self.item_item_values_input1 = tf.reduce_sum(tf.math.exp(self.first_low_att_layer_for_item_item_layer1( \
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.item_item_indices_input)], stddev=low_att_std)),[-1,1])    )   ),1)

                            ######## 第1层的均值和方差
        # first_mean_item_item_influ, first_var_item_item_influ = tf.nn.moments(self.item_item_values_input1, axes=0)
        # self.first_item_item_low_att = [first_mean_item_item_influ, first_var_item_item_influ]

                            ####
                            ######## 第2层
        self.second_low_att_layer_for_item_item_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='second_low_att_II_layer1')
        self.second_low_att_layer_for_item_item_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='second_low_att_II_layer2')
        #self.item_item_values_input2 = tf.Variable(tf.random_normal([len(self.item_item_indices_input)], stddev=0.01))
        self.item_item_values_input2 = tf.reduce_sum(tf.math.exp(self.second_low_att_layer_for_item_item_layer1(\
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.item_item_indices_input)], stddev=0.01)),[-1,1]) )   ),1)
                            ####
                            ######## 第3层
        self.third_low_att_layer_for_item_item_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='third_low_att_II_layer1')
        self.third_low_att_layer_for_item_item_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='third_low_att_II_layer2')
        #self.item_item_values_input3 = tf.Variable(tf.random_normal([len(self.item_item_indices_input)], stddev=0.01))
        self.item_item_values_input3 = tf.reduce_sum(tf.math.exp(self.third_low_att_layer_for_item_item_layer1(\
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.item_item_indices_input)], stddev=0.01)),[-1,1]) )   ),1)


                            ####
                            ######## 第4层
        self.fourth_low_att_layer_for_item_item_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='fourth_low_att_II_layer1')
        self.fourth_low_att_layer_for_item_item_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='fourth_low_att_II_layer2')
        #self.item_item_values_input4 = tf.Variable(tf.random_normal([len(self.item_item_indices_input)], stddev=0.01))
        self.item_item_values_input4 = tf.reduce_sum(tf.math.exp(self.fourth_low_att_layer_for_item_item_layer1(\
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.item_item_indices_input)], stddev=0.01)),[-1,1]) )   ),1)

                            ####
                            ######## 第5层
        self.fifth_low_att_layer_for_item_item_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='fifth_low_att_II_layer1')
        self.fifth_low_att_layer_for_item_item_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='fifth_low_att_II_layer2')
        #self.item_item_values_input5 = tf.Variable(tf.random_normal([len(self.item_item_indices_input)], stddev=0.01))
        self.item_item_values_input5 = tf.reduce_sum(tf.math.exp(self.fifth_low_att_layer_for_item_item_layer1(\
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.item_item_indices_input)], stddev=0.01)),[-1,1]) )   ),1)



        ########  Rough Graph Attention initialization 
        def get_two_random_probability():           ######## 
            # @function: 
            ab = tf.Variable(tf.random_normal([1, 2], stddev=0.01))
            ab_prob = tf.div(tf.math.exp(ab), tf.reduce_sum(tf.math.exp(ab), 1))
            return ( tf.slice(ab_prob, [0,0], [1,1]), tf.slice(ab_prob, [0,1], [1,1])  )

        self.first_user_userneighbor_attention_value, self.first_user_itemneighbor_attention_value = get_two_random_probability()
        self.second_user_userneighbor_attention_value, self.second_user_itemneighbor_attention_value = get_two_random_probability()
        self.third_user_userneighbor_attention_value, self.third_user_itemneighbor_attention_value = get_two_random_probability()
        self.fourth_user_userneighbor_attention_value, self.fourth_user_itemneighbor_attention_value = get_two_random_probability()
        self.fifth_user_userneighbor_attention_value, self.fifth_user_itemneighbor_attention_value = get_two_random_probability()

        self.first_item_itself_attention_value, self.first_item_userneighbor_attention_value = get_two_random_probability()
        self.second_item_itself_attention_value, self.second_item_userneighbor_attention_value = get_two_random_probability()
        self.third_item_itself_attention_value, self.third_item_userneighbor_attention_value  = get_two_random_probability()
        self.fourth_item_itself_attention_value, self.fourth_item_userneighbor_attention_value  = get_two_random_probability()
        self.fifth_item_itself_attention_value, self.fifth_item_userneighbor_attention_value  = get_two_random_probability()

        # # ---------------------- 
        # # User part
        # # First Layer Influence:
        # self.first_layer_user_attention_ini = tf.Variable(tf.random_normal([1, 2], stddev=0.01), name = "first_layer_user_part_influence_attention")
        #                                 #### 
        # first_layer_user_attention_norm_denominator = tf.reduce_sum(tf.math.exp(self.first_layer_user_attention_ini), 1)
        #                                 #### 
        # self.first_layer_user_attention = tf.div(tf.math.exp(self.first_layer_user_attention_ini), first_layer_user_attention_norm_denominator)
        #                                 #### 
        #                                 #### 
        # self.first_user_userneighbor_attention_value = tf.slice(self.first_layer_user_attention,[0,0],[1,1])
        #                                 #### 
        # self.first_user_itemneighbor_attention_value = tf.slice(self.first_layer_user_attention,[0,1],[1,1])
        #                                 #### 
        #                 ######## 

        # # Second Layer Influence:
        # self.second_layer_user_attention_ini = tf.Variable(tf.random_normal([1, 2], stddev=0.01), name = "second_layer_user_part_influence_attention")
        # second_layer_user_attention_norm_denominator = tf.reduce_sum(tf.math.exp(self.second_layer_user_attention_ini),1)
        # second_layer_user_attention = tf.div(tf.math.exp(self.second_layer_user_attention_ini), second_layer_user_attention_norm_denominator)
        # self.second_user_userneighbor_attention_value = tf.slice(second_layer_user_attention,[0,0],[1,1])
        # self.second_user_itemneighbor_attention_value = tf.slice(second_layer_user_attention,[0,1],[1,1])        

        # # Third Layer Influence:
        # self.third_layer_user_attention_ini = tf.Variable(tf.random_normal([1, 2], stddev=0.01), name = "third_layer_user_part_influence_attention")
        # third_layer_user_attention_norm_denominator = tf.reduce_sum(tf.math.exp(self.third_layer_user_attention_ini),1)
        # third_layer_user_attention = tf.div(tf.math.exp(self.third_layer_user_attention_ini), third_layer_user_attention_norm_denominator)
        # self.third_user_userneighbor_attention_value = tf.slice(third_layer_user_attention,[0,0],[1,1])
        # self.third_user_itemneighbor_attention_value = tf.slice(third_layer_user_attention,[0,1],[1,1])     


        # # ----------------------
        # # Item part
        # # First Layer Influence:
        # self.first_layer_item_attention_ini = tf.Variable(tf.random_normal([1, 2], stddev=0.01), name = "first_layer_item_part_influence_attention")
        # first_layer_item_attention_norm_denominator = tf.reduce_sum(tf.math.exp(self.first_layer_item_attention_ini),1)
        # self.first_layer_item_attention = tf.div(tf.math.exp(self.first_layer_item_attention_ini), first_layer_item_attention_norm_denominator)

        # self.first_item_itself_attention_value = tf.slice(self.first_layer_item_attention,[0,0],[1,1])
        # self.first_item_userneighbor_attention_value = tf.slice(self.first_layer_item_attention,[0,1],[1,1])

        # # Second Layer Influence:
        # self.second_layer_item_attention_ini = tf.Variable(tf.random_normal([1, 2], stddev=0.01), name = "second_layer_item_part_influence_attention")
        # second_layer_item_attention_norm_denominator = tf.reduce_sum(tf.math.exp(self.second_layer_item_attention_ini),1)
        # second_layer_item_attention = tf.div(tf.math.exp(self.second_layer_item_attention_ini), second_layer_item_attention_norm_denominator)

        # self.second_item_itself_attention_value = tf.slice(second_layer_item_attention,[0,0],[1,1])
        # self.second_item_userneighbor_attention_value = tf.slice(second_layer_item_attention,[0,1],[1,1])        

        # # Third Layer Influence:
        # self.third_layer_item_attention_ini = tf.Variable(tf.random_normal([1, 2], stddev=0.01), name = "third_layer_item_part_influence_attention")
        # third_layer_item_attention_norm_denominator = tf.reduce_sum(tf.math.exp(self.third_layer_item_attention_ini),1)
        # third_layer_item_attention = tf.div(tf.math.exp(self.third_layer_item_attention_ini), third_layer_item_attention_norm_denominator)

        # self.third_item_itself_attention_value = tf.slice(third_layer_item_attention,[0,0],[1,1])
        # self.third_item_userneighbor_attention_value = tf.slice(third_layer_item_attention,[0,1],[1,1])      


        ######## Generate Sparse Matrices with/without attention #########

        # ----------------------
        # prepare the shape of sparse matrice
                                        #### 
        self.social_neighbors_dense_shape = np.array([self.conf.num_users, self.conf.num_users]).astype(np.int64)
        self.consumed_items_dense_shape = np.array([self.conf.num_users, self.conf.num_items]).astype(np.int64)
        self.item_customer_dense_shape = np.array([self.conf.num_items, self.conf.num_users]).astype(np.int64)
        #######################################################################################
        #######################################################################################
        #######################################################################################
        self.item_item_dense_shape = np.array([self.conf.num_items, self.conf.num_items]).astype(np.int64)


        # ----------------------
        ####
        #### Frist Layer

        self.social_neighbors_sparse_matrix_avg = tf.SparseTensor(
            indices = self.social_neighbors_indices_input,                      #### [user, user], 
            values = self.social_neighbors_values_input,                        #### [1.0/friend的个数]
            dense_shape=self.social_neighbors_dense_shape                       #### 
        )
        self.first_layer_social_neighbors_sparse_matrix = tf.SparseTensor(
            indices = self.social_neighbors_indices_input,                      #### [user, user]
            values = self.social_neighbors_values_input1,                       #### 
            dense_shape=self.social_neighbors_dense_shape                       #### 
        )
        #######################################################################################
        #######################################################################################
        #######################################################################################
        self.first_layer_user_user_uiu_sparse_matrix = tf.SparseTensor(
            indices = self.user_user_uiu_indices_input,                         ############ uiu
            values = self.user_user_uiu_values_input1,
            dense_shape=self.social_neighbors_dense_shape
        )
        self.first_layer_user_user_u0u_sparse_matrix = tf.SparseTensor(
            indices = self.user_user_u0u_indices_input,                         ############ u0u
            values = self.user_user_u0u_values_input1,
            dense_shape=self.social_neighbors_dense_shape
        )


        self.consumed_items_sparse_matrix_avg = tf.SparseTensor(
            indices = self.consumed_items_indices_input,                        #### [user, item]
            values = self.consumed_items_values_input,                          #### 
            dense_shape=self.consumed_items_dense_shape                         #### 
        )
        self.first_layer_consumed_items_sparse_matrix = tf.SparseTensor(
            indices = self.consumed_items_indices_input,                        #### [user, item]
            values = self.consumed_items_values_input1,                         #### 
            dense_shape=self.consumed_items_dense_shape                         #### 
        )
        self.item_customer_sparse_matrix_avg = tf.SparseTensor(
            indices = self.item_customer_indices_input,                         #### [item, user]
            values = self.item_customer_values_input,                           #### 
            dense_shape=self.item_customer_dense_shape                          #### 
        )
        self.first_layer_item_customer_sparse_matrix = tf.SparseTensor(
            indices = self.item_customer_indices_input,                         #### [user, item]
            values = self.item_customer_values_input1,                          #### 
            dense_shape=self.item_customer_dense_shape                          ####
        )

        #######################################################################################
        #######################################################################################
        #######################################################################################
        # self.first_layer_item_item_sparse_matrix = tf.SparseTensor(
        #     indices = self.item_item_indices_input,                             #### [item, item]
        #     values = self.item_item_values_input,                              #### 
        #     dense_shape = self.item_item_dense_shape
        # )
        self.first_layer_item_item_sparse_matrix = tf.SparseTensor(
            indices=self.item_item_indices_input,                               #### [item, item]
            values=self.item_item_values_input1,                                #### 
            dense_shape=self.item_item_dense_shape
        )


        self.first_social_neighbors_low_level_att_matrix = tf.sparse.softmax(self.first_layer_social_neighbors_sparse_matrix) 
        #######################################################################################
        #######################################################################################
        #######################################################################################
        self.first_user_user_uiu_low_level_att_matrix = tf.sparse.softmax(self.first_layer_user_user_uiu_sparse_matrix)         ######## uiu
        self.first_user_user_u0u_low_level_att_matrix = tf.sparse.softmax(self.first_layer_user_user_u0u_sparse_matrix)         ######## u0u

        self.first_consumed_items_low_level_att_matrix = tf.sparse.softmax(self.first_layer_consumed_items_sparse_matrix) 
        self.first_items_users_neighborslow_level_att_matrix = tf.sparse.softmax(self.first_layer_item_customer_sparse_matrix) 
        ######################################################################################
        ######################################################################################
        ######################################################################################
        self.first_item_item_low_level_att_matrix = tf.sparse.softmax(self.first_layer_item_item_sparse_matrix)                 ######## item-item

        # ----------------------
        ####
        ##### Second layer 

        self.second_layer_social_neighbors_sparse_matrix = tf.SparseTensor(
            indices = self.social_neighbors_indices_input,                      #### [user, user]
            values = self.social_neighbors_values_input2,                       #### 
            dense_shape=self.social_neighbors_dense_shape
        )
        #######################################################################################
        #######################################################################################
        #######################################################################################
        self.second_layer_user_user_uiu_sparse_matrix = tf.SparseTensor(
            indices = self.user_user_uiu_indices_input,                         ############ uiu
            values = self.user_user_uiu_values_input2,
            dense_shape=self.social_neighbors_dense_shape
        )
        self.second_layer_user_user_u0u_sparse_matrix = tf.SparseTensor(
            indices = self.user_user_u0u_indices_input,                         ############ u0u
            values = self.user_user_u0u_values_input2,
            dense_shape=self.social_neighbors_dense_shape
        )


        self.second_layer_consumed_items_sparse_matrix = tf.SparseTensor(
            indices = self.consumed_items_indices_input,                        #### [user, item]
            values = self.consumed_items_values_input2,                         #### 
            dense_shape=self.consumed_items_dense_shape
        )
        self.second_layer_item_customer_sparse_matrix = tf.SparseTensor(
            indices = self.item_customer_indices_input,                         #### [item, item] 
            values = self.item_customer_values_input2,                          #### 
            dense_shape=self.item_customer_dense_shape
        )
        #######################################################################################
        #######################################################################################
        #######################################################################################
        self.second_layer_item_item_sparse_matrix = tf.SparseTensor(
            indices = self.item_item_indices_input,                             #### [item, item]
            values = self.item_item_values_input2,                              #### 
            dense_shape = self.item_item_dense_shape
        )

        
        self.second_social_neighbors_low_level_att_matrix = tf.sparse.softmax(self.second_layer_social_neighbors_sparse_matrix) 
        #######################################################################################
        #######################################################################################
        #######################################################################################
        self.second_user_user_uiu_low_level_att_matrix = tf.sparse.softmax(self.second_layer_user_user_uiu_sparse_matrix)         ######## uiu
        self.second_user_user_u0u_low_level_att_matrix = tf.sparse.softmax(self.second_layer_user_user_u0u_sparse_matrix)         ######## u0u
        
        self.second_consumed_items_low_level_att_matrix = tf.sparse.softmax(self.second_layer_consumed_items_sparse_matrix) 
        self.second_items_users_neighborslow_level_att_matrix = tf.sparse.softmax(self.second_layer_item_customer_sparse_matrix) 
        ######################################################################################
        ######################################################################################
        ######################################################################################
        self.second_item_item_low_level_att_matrix = tf.sparse.softmax(self.second_layer_item_item_sparse_matrix)                 ######## item-item

        
        # # ----------------------
        # # Third layer 

        # self.third_layer_social_neighbors_sparse_matrix = tf.SparseTensor(
        #     indices = self.social_neighbors_indices_input,                      #### [user, user]
        #     values = self.social_neighbors_values_input3,                       #### 
        #     dense_shape=self.social_neighbors_dense_shape
        # )
        # self.third_layer_consumed_items_sparse_matrix = tf.SparseTensor(
        #     indices = self.consumed_items_indices_input,                        #### [user, item]  
        #     values = self.consumed_items_values_input3,                         #### 
        #     dense_shape=self.consumed_items_dense_shape
        # )
        # self.third_layer_item_customer_sparse_matrix = tf.SparseTensor(
        #     indices = self.item_customer_indices_input,                         #### [item, user]
        #     values = self.item_customer_values_input3,                          #### 
        #     dense_shape=self.item_customer_dense_shape
        # )

        # self.third_social_neighbors_low_level_att_matrix = tf.sparse.softmax(self.third_layer_social_neighbors_sparse_matrix)
        # self.third_consumed_items_low_level_att_matrix = tf.sparse.softmax(self.third_layer_consumed_items_sparse_matrix)
        # self.third_items_users_neighborslow_level_att_matrix = tf.sparse.softmax(self.third_layer_item_customer_sparse_matrix)




        # ----------------------
        ####
        #### Third layer 

        self.third_layer_social_neighbors_sparse_matrix = tf.SparseTensor(
            indices = self.social_neighbors_indices_input,                      #### [user, user]
            values = self.social_neighbors_values_input3,                       #### 
            dense_shape=self.social_neighbors_dense_shape
        )
        #######################################################################################
        #######################################################################################
        #######################################################################################
        self.third_layer_user_user_uiu_sparse_matrix = tf.SparseTensor(
            indices = self.user_user_uiu_indices_input,                         ############ uiu
            values = self.user_user_uiu_values_input3,
            dense_shape=self.social_neighbors_dense_shape
        )
        self.third_layer_user_user_u0u_sparse_matrix = tf.SparseTensor(
            indices = self.user_user_u0u_indices_input,                         ############ u0u
            values = self.user_user_u0u_values_input3,
            dense_shape=self.social_neighbors_dense_shape
        )


        self.third_layer_consumed_items_sparse_matrix = tf.SparseTensor(
            indices = self.consumed_items_indices_input,                        #### [user, item]
            values = self.consumed_items_values_input3,                         #### 
            dense_shape=self.consumed_items_dense_shape
        )
        self.third_layer_item_customer_sparse_matrix = tf.SparseTensor(
            indices = self.item_customer_indices_input,                         #### [item, item] 
            values = self.item_customer_values_input3,                          #### 
            dense_shape=self.item_customer_dense_shape
        )
        #######################################################################################
        #######################################################################################
        #######################################################################################
        self.third_layer_item_item_sparse_matrix = tf.SparseTensor(
            indices = self.item_item_indices_input,                             #### [item, item]
            values = self.item_item_values_input3,                              #### 
            dense_shape = self.item_item_dense_shape
        )

        
        self.third_social_neighbors_low_level_att_matrix = tf.sparse.softmax(self.third_layer_social_neighbors_sparse_matrix) 
        #######################################################################################
        #######################################################################################
        #######################################################################################
        self.third_user_user_uiu_low_level_att_matrix = tf.sparse.softmax(self.third_layer_user_user_uiu_sparse_matrix)         ######## uiu
        self.third_user_user_u0u_low_level_att_matrix = tf.sparse.softmax(self.third_layer_user_user_u0u_sparse_matrix)         ######## u0u
        
        self.third_consumed_items_low_level_att_matrix = tf.sparse.softmax(self.third_layer_consumed_items_sparse_matrix) 
        self.third_items_users_neighborslow_level_att_matrix = tf.sparse.softmax(self.third_layer_item_customer_sparse_matrix) 
        ######################################################################################
        ######################################################################################
        ######################################################################################
        self.third_item_item_low_level_att_matrix = tf.sparse.softmax(self.third_layer_item_item_sparse_matrix)                 ######## item-item




        # ----------------------
        ####
        #### Fourth layer 

        self.fourth_layer_social_neighbors_sparse_matrix = tf.SparseTensor(
            indices = self.social_neighbors_indices_input,                      #### [user, user]
            values = self.social_neighbors_values_input4,                       #### 
            dense_shape=self.social_neighbors_dense_shape
        )
        #######################################################################################
        #######################################################################################
        #######################################################################################
        self.fourth_layer_user_user_uiu_sparse_matrix = tf.SparseTensor(
            indices = self.user_user_uiu_indices_input,                         ############ uiu
            values = self.user_user_uiu_values_input4,
            dense_shape=self.social_neighbors_dense_shape
        )
        self.fourth_layer_user_user_u0u_sparse_matrix = tf.SparseTensor(
            indices = self.user_user_u0u_indices_input,                         ############ u0u
            values = self.user_user_u0u_values_input4,
            dense_shape=self.social_neighbors_dense_shape
        )


        self.fourth_layer_consumed_items_sparse_matrix = tf.SparseTensor(
            indices = self.consumed_items_indices_input,                        #### [user, item]
            values = self.consumed_items_values_input4,                         #### 
            dense_shape=self.consumed_items_dense_shape
        )
        self.fourth_layer_item_customer_sparse_matrix = tf.SparseTensor(
            indices = self.item_customer_indices_input,                         #### [item, item] 
            values = self.item_customer_values_input4,                          #### 
            dense_shape=self.item_customer_dense_shape
        )
        #######################################################################################
        #######################################################################################
        #######################################################################################
        self.fourth_layer_item_item_sparse_matrix = tf.SparseTensor(
            indices = self.item_item_indices_input,                             #### [item, item]
            values = self.item_item_values_input4,                              #### 
            dense_shape = self.item_item_dense_shape
        )

        
        self.fourth_social_neighbors_low_level_att_matrix = tf.sparse.softmax(self.fourth_layer_social_neighbors_sparse_matrix) 
        #######################################################################################
        #######################################################################################
        #######################################################################################
        self.fourth_user_user_uiu_low_level_att_matrix = tf.sparse.softmax(self.fourth_layer_user_user_uiu_sparse_matrix)         ######## uiu
        self.fourth_user_user_u0u_low_level_att_matrix = tf.sparse.softmax(self.fourth_layer_user_user_u0u_sparse_matrix)         ######## u0u
        
        self.fourth_consumed_items_low_level_att_matrix = tf.sparse.softmax(self.fourth_layer_consumed_items_sparse_matrix) 
        self.fourth_items_users_neighborslow_level_att_matrix = tf.sparse.softmax(self.fourth_layer_item_customer_sparse_matrix) 
        ######################################################################################
        ######################################################################################
        ######################################################################################
        self.fourth_item_item_low_level_att_matrix = tf.sparse.softmax(self.fourth_layer_item_item_sparse_matrix)                 ######## item-item


        # ----------------------
        ####
        #### Fifth layer 

        self.fifth_layer_social_neighbors_sparse_matrix = tf.SparseTensor(
            indices = self.social_neighbors_indices_input,                      #### [user, user]
            values = self.social_neighbors_values_input5,                       #### 
            dense_shape=self.social_neighbors_dense_shape
        )
        #######################################################################################
        #######################################################################################
        #######################################################################################
        self.fifth_layer_user_user_uiu_sparse_matrix = tf.SparseTensor(
            indices = self.user_user_uiu_indices_input,                         ############ uiu
            values = self.user_user_uiu_values_input5,
            dense_shape=self.social_neighbors_dense_shape
        )
        self.fifth_layer_user_user_u0u_sparse_matrix = tf.SparseTensor(
            indices = self.user_user_u0u_indices_input,                         ############ u0u
            values = self.user_user_u0u_values_input5,
            dense_shape=self.social_neighbors_dense_shape
        )


        self.fifth_layer_consumed_items_sparse_matrix = tf.SparseTensor(
            indices = self.consumed_items_indices_input,                        #### [user, item]
            values = self.consumed_items_values_input5,                         #### 
            dense_shape=self.consumed_items_dense_shape
        )
        self.fifth_layer_item_customer_sparse_matrix = tf.SparseTensor(
            indices = self.item_customer_indices_input,                         #### [item, item] 
            values = self.item_customer_values_input5,                          #### 
            dense_shape=self.item_customer_dense_shape
        )
        #######################################################################################
        #######################################################################################
        #######################################################################################
        self.fifth_layer_item_item_sparse_matrix = tf.SparseTensor(
            indices = self.item_item_indices_input,                             #### [item, item]
            values = self.item_item_values_input5,                              #### 
            dense_shape = self.item_item_dense_shape
        )

        
        self.fifth_social_neighbors_low_level_att_matrix = tf.sparse.softmax(self.fifth_layer_social_neighbors_sparse_matrix) 
        #######################################################################################
        #######################################################################################
        #######################################################################################
        self.fifth_user_user_uiu_low_level_att_matrix = tf.sparse.softmax(self.fifth_layer_user_user_uiu_sparse_matrix)         ######## uiu
        self.fifth_user_user_u0u_low_level_att_matrix = tf.sparse.softmax(self.fifth_layer_user_user_u0u_sparse_matrix)         ######## u0u
        
        self.fifth_consumed_items_low_level_att_matrix = tf.sparse.softmax(self.fifth_layer_consumed_items_sparse_matrix) 
        self.fifth_items_users_neighborslow_level_att_matrix = tf.sparse.softmax(self.fifth_layer_item_customer_sparse_matrix) 
        ######################################################################################
        ######################################################################################
        ######################################################################################
        self.fifth_item_item_low_level_att_matrix = tf.sparse.softmax(self.fifth_layer_item_item_sparse_matrix)                 ######## item-item



        

    # ----------------------
    # Operations for Diffusion
                ####
                ######## 第1层
    def generateUserEmbeddingFromSocialNeighbors(self, current_user_embedding):
        user_embedding_from_social_neighbors = tf.sparse_tensor_dense_matmul(
            self.social_neighbors_sparse_matrix_avg, current_user_embedding                 # avg
        )
        return user_embedding_from_social_neighbors


    def generateUserEmbeddingFromSocialNeighbors1(self, current_user_embedding):            #### [user, user]
        user_embedding_from_social_neighbors = tf.sparse_tensor_dense_matmul(
            self.first_social_neighbors_low_level_att_matrix, current_user_embedding
        )
        return user_embedding_from_social_neighbors
   
    ##################################################################################################
    ##################################################################################################
    ##################################################################################################
    def generateUserEmbeddingFrom_uiu_1(self, current_user_embedding):                      #### uiu
        user_embedding_from_uiu = tf.sparse_tensor_dense_matmul(
            self.first_user_user_uiu_low_level_att_matrix, current_user_embedding
        )
        return user_embedding_from_uiu
    def generateUserEmbeddingFrom_u0u_1(self, current_user_embedding):                      #### u0u
        user_embedding_from_u0u = tf.sparse_tensor_dense_matmul(
            self.first_user_user_u0u_low_level_att_matrix, current_user_embedding
        )
        return user_embedding_from_u0u


    # def generateUserEmebddingFromConsumedItems(self, current_item_embedding):              
    #     user_embedding_from_consumed_items = tf.sparse_tensor_dense_matmul(
    #         self.consumed_items_sparse_matrix_avg, current_item_embedding                   # avg
    #     )
    #     return user_embedding_from_consumed_items


    def generateUserEmebddingFromConsumedItems1(self, current_item_embedding):              #### [user, item]
        user_embedding_from_consumed_items = tf.sparse_tensor_dense_matmul(
            self.first_consumed_items_low_level_att_matrix, current_item_embedding
        )
        return user_embedding_from_consumed_items


    # def generateItemEmebddingFromCustomer(self, current_user_embedding):
    #     item_embedding_from_customer = tf.sparse_tensor_dense_matmul(
    #         self.item_customer_sparse_matrix_avg, current_user_embedding                    # avg
    #     )
    #     return item_embedding_from_customer


    def generateItemEmebddingFromCustomer1(self, current_user_embedding):                   #### [item, user]
        item_embedding_from_customer = tf.sparse_tensor_dense_matmul(
            self.first_items_users_neighborslow_level_att_matrix, current_user_embedding
        )
        return item_embedding_from_customer

    ##################################################################################################
    ##################################################################################################
    ##################################################################################################
    def generateItemEmbeddingFromItem1(self, current_item_embedding):                       #### [item, item]
        item_embedding_from_item = tf.sparse_tensor_dense_matmul(
            self.first_item_item_low_level_att_matrix, current_item_embedding
        )
        return item_embedding_from_item
                

                ########
                ########
                ######## 第2层
    def generateUserEmbeddingFromSocialNeighbors2(self, current_user_embedding):            #### [user, user]
        user_embedding_from_social_neighbors = tf.sparse_tensor_dense_matmul(
            self.second_social_neighbors_low_level_att_matrix, current_user_embedding           
        )
        return user_embedding_from_social_neighbors
    ##################################################################################################
    ##################################################################################################
    ##################################################################################################
    def generateUserEmbeddingFrom_uiu_2(self, current_user_embedding):                      #### uiu
        user_embedding_from_uiu = tf.sparse_tensor_dense_matmul(
            self.second_user_user_uiu_low_level_att_matrix, current_user_embedding
        )
        return user_embedding_from_uiu
    def generateUserEmbeddingFrom_u0u_2(self, current_user_embedding):                      #### u0u
        user_embedding_from_u0u = tf.sparse_tensor_dense_matmul(
            self.second_user_user_u0u_low_level_att_matrix, current_user_embedding
        )
        return user_embedding_from_u0u
    
    
    def generateUserEmebddingFromConsumedItems2(self, current_item_embedding):              #### [user, item]
        user_embedding_from_consumed_items = tf.sparse_tensor_dense_matmul(
            self.second_consumed_items_low_level_att_matrix, current_item_embedding
        )
        return user_embedding_from_consumed_items

    def generateItemEmebddingFromCustomer2(self, current_user_embedding):                   #### [item, user]
        item_embedding_from_customer = tf.sparse_tensor_dense_matmul(
            self.second_items_users_neighborslow_level_att_matrix, current_user_embedding
        )
        return item_embedding_from_customer
    ##################################################################################################
    ##################################################################################################
    ##################################################################################################
    def generateItemEmbeddingFromItem2(self, current_item_embedding):                       #### [item, item]
        item_embedding_from_item = tf.sparse_tensor_dense_matmul(
            self.second_item_item_low_level_att_matrix, current_item_embedding
        )
        return item_embedding_from_item


                ########
                ########
                ######## 第3层
    def generateUserEmbeddingFromSocialNeighbors3(self, current_user_embedding):            #### [user, user]
        user_embedding_from_social_neighbors = tf.sparse_tensor_dense_matmul(
            self.third_social_neighbors_low_level_att_matrix, current_user_embedding           
        )
        return user_embedding_from_social_neighbors
    ##################################################################################################
    ##################################################################################################
    ##################################################################################################
    def generateUserEmbeddingFrom_uiu_3(self, current_user_embedding):                      #### uiu
        user_embedding_from_uiu = tf.sparse_tensor_dense_matmul(
            self.third_user_user_uiu_low_level_att_matrix, current_user_embedding
        )
        return user_embedding_from_uiu
    def generateUserEmbeddingFrom_u0u_3(self, current_user_embedding):                      #### u0u
        user_embedding_from_u0u = tf.sparse_tensor_dense_matmul(
            self.third_user_user_u0u_low_level_att_matrix, current_user_embedding
        )
        return user_embedding_from_u0u
    
    
    def generateUserEmebddingFromConsumedItems3(self, current_item_embedding):              #### [user, item]
        user_embedding_from_consumed_items = tf.sparse_tensor_dense_matmul(
            self.third_consumed_items_low_level_att_matrix, current_item_embedding
        )
        return user_embedding_from_consumed_items

    def generateItemEmebddingFromCustomer3(self, current_user_embedding):                   #### [item, user]
        item_embedding_from_customer = tf.sparse_tensor_dense_matmul(
            self.third_items_users_neighborslow_level_att_matrix, current_user_embedding
        )
        return item_embedding_from_customer
    ##################################################################################################
    ##################################################################################################
    ##################################################################################################
    def generateItemEmbeddingFromItem3(self, current_item_embedding):                       #### [item, item]
        item_embedding_from_item = tf.sparse_tensor_dense_matmul(
            self.third_item_item_low_level_att_matrix, current_item_embedding
        )
        return item_embedding_from_item




                ########
                ########
                ######## 第4层
    def generateUserEmbeddingFromSocialNeighbors4(self, current_user_embedding):            #### [user, user]
        user_embedding_from_social_neighbors = tf.sparse_tensor_dense_matmul(
            self.fourth_social_neighbors_low_level_att_matrix, current_user_embedding           
        )
        return user_embedding_from_social_neighbors
    ##################################################################################################
    ##################################################################################################
    ##################################################################################################
    def generateUserEmbeddingFrom_uiu_4(self, current_user_embedding):                      #### uiu
        user_embedding_from_uiu = tf.sparse_tensor_dense_matmul(
            self.fourth_user_user_uiu_low_level_att_matrix, current_user_embedding
        )
        return user_embedding_from_uiu
    def generateUserEmbeddingFrom_u0u_4(self, current_user_embedding):                      #### u0u
        user_embedding_from_u0u = tf.sparse_tensor_dense_matmul(
            self.fourth_user_user_u0u_low_level_att_matrix, current_user_embedding
        )
        return user_embedding_from_u0u
    
    
    def generateUserEmebddingFromConsumedItems4(self, current_item_embedding):              #### [user, item]
        user_embedding_from_consumed_items = tf.sparse_tensor_dense_matmul(
            self.fourth_consumed_items_low_level_att_matrix, current_item_embedding
        )
        return user_embedding_from_consumed_items

    def generateItemEmebddingFromCustomer4(self, current_user_embedding):                   #### [item, user]
        item_embedding_from_customer = tf.sparse_tensor_dense_matmul(
            self.fourth_items_users_neighborslow_level_att_matrix, current_user_embedding
        )
        return item_embedding_from_customer
    ##################################################################################################
    ##################################################################################################
    ##################################################################################################
    def generateItemEmbeddingFromItem4(self, current_item_embedding):                       #### [item, item]
        item_embedding_from_item = tf.sparse_tensor_dense_matmul(
            self.fourth_item_item_low_level_att_matrix, current_item_embedding
        )
        return item_embedding_from_item



                ########
                ########
                ######## 第4层
    def generateUserEmbeddingFromSocialNeighbors5(self, current_user_embedding):            #### [user, user]
        user_embedding_from_social_neighbors = tf.sparse_tensor_dense_matmul(
            self.fifth_social_neighbors_low_level_att_matrix, current_user_embedding           
        )
        return user_embedding_from_social_neighbors
    ##################################################################################################
    ##################################################################################################
    ##################################################################################################
    def generateUserEmbeddingFrom_uiu_5(self, current_user_embedding):                      #### uiu
        user_embedding_from_uiu = tf.sparse_tensor_dense_matmul(
            self.fifth_user_user_uiu_low_level_att_matrix, current_user_embedding
        )
        return user_embedding_from_uiu
    def generateUserEmbeddingFrom_u0u_5(self, current_user_embedding):                      #### u0u
        user_embedding_from_u0u = tf.sparse_tensor_dense_matmul(
            self.fifth_user_user_u0u_low_level_att_matrix, current_user_embedding
        )
        return user_embedding_from_u0u
    
    
    def generateUserEmebddingFromConsumedItems5(self, current_item_embedding):              #### [user, item]
        user_embedding_from_consumed_items = tf.sparse_tensor_dense_matmul(
            self.fifth_consumed_items_low_level_att_matrix, current_item_embedding
        )
        return user_embedding_from_consumed_items

    def generateItemEmebddingFromCustomer5(self, current_user_embedding):                   #### [item, user]
        item_embedding_from_customer = tf.sparse_tensor_dense_matmul(
            self.fifth_items_users_neighborslow_level_att_matrix, current_user_embedding
        )
        return item_embedding_from_customer
    ##################################################################################################
    ##################################################################################################
    ##################################################################################################
    def generateItemEmbeddingFromItem5(self, current_item_embedding):                       #### [item, item]
        item_embedding_from_item = tf.sparse_tensor_dense_matmul(
            self.fifth_item_item_low_level_att_matrix, current_item_embedding
        )
        return item_embedding_from_item




    def initializeNodes(self):
        self.item_input = tf.placeholder("int32", [None, 1]) # Get item embedding from the core_item_input
        self.user_input = tf.placeholder("int32", [None, 1]) # Get user embedding from the core_user_input
        self.labels_input = tf.placeholder("float32", [None, 1])
                                #### 
                                #### 

        self.user_embedding = tf.Variable(
            tf.random_normal([self.conf.num_users, self.conf.dimension], stddev=0.01), name='user_embedding')
        self.item_embedding = tf.Variable(
            tf.random_normal([self.conf.num_items, self.conf.dimension], stddev=0.01), name='item_embedding')


        self.user_review_vector_matrix = tf.constant(\
            np.load(self.conf.user_review_vector_matrix), dtype=tf.float32)
        self.item_review_vector_matrix = tf.constant(\
            np.load(self.conf.item_review_vector_matrix), dtype=tf.float32)
        self.reduce_dimension_layer = tf.layers.Dense(\
            self.conf.dimension, activation=tf.nn.sigmoid, name='reduce_dimension_layer')

        ########  Fine-grained Graph Attention initialization ########
        # ----------------------
        # User part

        # ----------------------
        ####
        #### First diffusion layer
        self.first_user_part_social_graph_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='firstGCN_UU_user_MLP_first_layer')

        self.first_user_part_social_graph_att_layer2 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='firstGCN_UU_user_MLP_sencond_layer')              #### [user, user]

        ######################################################################################
        ######################################################################################
        ######################################################################################
        self.first_user_user_uiu_subgraph_att_layer1 = tf.layers.Dense(1, activation=tf.nn.tanh, name='firstGCN_user_user_uiu_MLP_first_layer')
        self.first_user_user_uiu_subgraph_att_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='firstGCN_user_user_uiu_MLP_second_layer')

        self.first_user_user_u0u_subgraph_att_layer1 = tf.layers.Dense(1, activation=tf.nn.tanh, name='firstGCN_user_user_u0u_MLP_first_layer')
        self.first_user_user_u0u_subgraph_att_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='firstGCN_user_user_u0u_MLP_second_layer')

        self.first_user_user_self_att_layer1 = tf.layers.Dense(1, activation=tf.nn.tanh, name='firstGCN_user_user_self_MLP_first_layer')
        self.first_user_user_self_att_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='firstGCN_user_user_self_MLP_second_layer')


        self.first_user_part_interest_graph_att_layer1 = tf.layers.Dense(1, activation=tf.nn.tanh, name='firstGCN_UI_user_MLP_first_layer')
        self.first_user_part_interest_graph_att_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='firstGCN_UI_user_MLP_second_layer')               #### [user, item]


        # ----------------------
        ####
        #### Second diffusion layer
        self.second_user_part_social_graph_att_layer1 = tf.layers.Dense(1, activation=tf.nn.tanh, name='secondGCN_UU_user_MLP_first_layer')

        self.second_user_part_social_graph_att_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='secondGCN_UU_user_MLP_second_layer')              #### [user, user]

        ######################################################################################
        ######################################################################################
        ######################################################################################
        self.second_user_user_uiu_subgraph_att_layer1 = tf.layers.Dense(1, activation=tf.nn.tanh, name='secondGCN_user_user_uiu_MLP_first_layer')
        self.second_user_user_uiu_subgraph_att_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='secondGCN_user_user_uiu_MLP_second_layer')

        self.second_user_user_u0u_subgraph_att_layer1 = tf.layers.Dense(1, activation=tf.nn.tanh, name='secondGCN_user_user_u0u_MLP_first_layer')
        self.second_user_user_u0u_subgraph_att_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='secondGCN_user_user_u0u_MLP_second_layer')

        self.second_user_user_self_att_layer1 = tf.layers.Dense(1, activation=tf.nn.tanh, name='secondGCN_user_user_self_MLP_first_layer')
        self.second_user_user_self_att_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='secondGCN_user_user_self_MLP_second_layer')

        self.second_user_part_interest_graph_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='secondGCN_UI_user_MLP_first_layer')

        self.second_user_part_interest_graph_att_layer2 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='secondGCN_UI_user_MLP_second_layer')              #### [user, item]


        # ----------------------
        ####
        #### Third diffusion layer
        self.third_user_part_social_graph_att_layer1 = tf.layers.Dense(1, activation=tf.nn.tanh, name='thirdGCN_UU_user_MLP_first_layer')

        self.third_user_part_social_graph_att_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='thirdGCN_UU_user_MLP_second_layer')              #### [user, user]

        ######################################################################################
        ######################################################################################
        ######################################################################################
        self.third_user_user_uiu_subgraph_att_layer1 = tf.layers.Dense(1, activation=tf.nn.tanh, name='thirdGCN_user_user_uiu_MLP_first_layer')
        self.third_user_user_uiu_subgraph_att_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='thirdGCN_user_user_uiu_MLP_second_layer')

        self.third_user_user_u0u_subgraph_att_layer1 = tf.layers.Dense(1, activation=tf.nn.tanh, name='thirdGCN_user_user_u0u_MLP_first_layer')
        self.third_user_user_u0u_subgraph_att_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='thirdGCN_user_user_u0u_MLP_second_layer')

        self.third_user_user_self_att_layer1 = tf.layers.Dense(1, activation=tf.nn.tanh, name='thirdGCN_user_user_self_MLP_first_layer')
        self.third_user_user_self_att_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='thirdGCN_user_user_self_MLP_second_layer')

        self.third_user_part_interest_graph_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='thirdGCN_UI_user_MLP_first_layer')

        self.third_user_part_interest_graph_att_layer2 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='thirdGCN_UI_user_MLP_second_layer')              #### [user, item]



        # ----------------------
        ####
        #### Fourth diffusion layer
        self.fourth_user_part_social_graph_att_layer1 = tf.layers.Dense(1, activation=tf.nn.tanh, name='fourthGCN_UU_user_MLP_first_layer')

        self.fourth_user_part_social_graph_att_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='fourthGCN_UU_user_MLP_second_layer')              #### [user, user]

        ######################################################################################
        ######################################################################################
        ######################################################################################
        self.fourth_user_user_uiu_subgraph_att_layer1 = tf.layers.Dense(1, activation=tf.nn.tanh, name='fourthGCN_user_user_uiu_MLP_first_layer')
        self.fourth_user_user_uiu_subgraph_att_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='fourthGCN_user_user_uiu_MLP_second_layer')

        self.fourth_user_user_u0u_subgraph_att_layer1 = tf.layers.Dense(1, activation=tf.nn.tanh, name='fourthGCN_user_user_u0u_MLP_first_layer')
        self.fourth_user_user_u0u_subgraph_att_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='fourthGCN_user_user_u0u_MLP_second_layer')

        self.fourth_user_user_self_att_layer1 = tf.layers.Dense(1, activation=tf.nn.tanh, name='fourthGCN_user_user_self_MLP_first_layer')
        self.fourth_user_user_self_att_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='fourthGCN_user_user_self_MLP_second_layer')

        self.fourth_user_part_interest_graph_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='fourthGCN_UI_user_MLP_first_layer')

        self.fourth_user_part_interest_graph_att_layer2 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='fourthGCN_UI_user_MLP_second_layer')              #### [user, item]




        # ----------------------
        ####
        #### Fifth diffusion layer
        self.fifth_user_part_social_graph_att_layer1 = tf.layers.Dense(1, activation=tf.nn.tanh, name='fifthGCN_UU_user_MLP_first_layer')

        self.fifth_user_part_social_graph_att_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='fifthGCN_UU_user_MLP_second_layer')              #### [user, user]

        ######################################################################################
        ######################################################################################
        ######################################################################################
        self.fifth_user_user_uiu_subgraph_att_layer1 = tf.layers.Dense(1, activation=tf.nn.tanh, name='fifthGCN_user_user_uiu_MLP_first_layer')
        self.fifth_user_user_uiu_subgraph_att_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='fifthGCN_user_user_uiu_MLP_second_layer')

        self.fifth_user_user_u0u_subgraph_att_layer1 = tf.layers.Dense(1, activation=tf.nn.tanh, name='fifthGCN_user_user_u0u_MLP_first_layer')
        self.fifth_user_user_u0u_subgraph_att_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='fifthGCN_user_user_u0u_MLP_second_layer')

        self.fifth_user_user_self_att_layer1 = tf.layers.Dense(1, activation=tf.nn.tanh, name='fifthGCN_user_user_self_MLP_first_layer')
        self.fifth_user_user_self_att_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='fifthGCN_user_user_self_MLP_second_layer')

        self.fifth_user_part_interest_graph_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='fifthGCN_UI_user_MLP_first_layer')

        self.fifth_user_part_interest_graph_att_layer2 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='fifthGCN_UI_user_MLP_second_layer')              #### [user, item]




        # ----------------------
        # Item part
                        ####
                        ####
                        #### 第1层
        self.first_item_part_itself_graph_att_layer1 = tf.layers.Dense(1, activation=tf.nn.tanh, name='firstGCN_IU_itemself_MLP_first_layer')

        self.first_item_part_itself_graph_att_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='firstGCN_IU_itemself_MLP_second_layer')           #### item-self

        self.first_item_part_user_graph_att_layer1 = tf.layers.Dense(1, activation=tf.nn.tanh, name='firstGCN_IU_customer_MLP_first_layer')

        self.first_item_part_user_graph_att_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='firstGCN_IU_customer_MLP_second_layer')           #### [item, user]

        ######################################################################################
        ######################################################################################
        ######################################################################################
        self.first_item_part_item_graph_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='firstGCN_II_item_MLP_first_layer')
        self.first_item_part_item_graph_att_layer2 = tf.layers.Dense(\
            1,activation=tf.nn.leaky_relu, name='firstGCN_II_item_MLP_second_layer')                #### [item, item]

                        ####
                        ####
                        #### 第2层
        self.second_item_part_itself_graph_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='secondGCN_IU_itemself_MLP_first_layer')

        self.second_item_part_itself_graph_att_layer2 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='secondGCN_IU_itemself_MLP_second_layer')          #### item-self

        self.second_item_part_user_graph_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='secondGCN_IU_customer_MLP_first_layer')

        self.second_item_part_user_graph_att_layer2 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='secondGCN_IU_customer_MLP_second_layer')          #### [item, user]
        ######################################################################################
        ######################################################################################
        ######################################################################################
        self.second_item_part_item_graph_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='secondGCN_II_item_MLP_first_layer')
        self.second_item_part_item_graph_att_layer2 = tf.layers.Dense(\
            1,activation=tf.nn.leaky_relu, name='secondGCN_II_item_MLP_second_layer')                #### [item, item]



                        ####
                        ####
                        #### 第3层
        self.third_item_part_itself_graph_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='thirdGCN_IU_itemself_MLP_first_layer')

        self.third_item_part_itself_graph_att_layer2 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='thirdGCN_IU_itemself_MLP_second_layer')          #### item-self

        self.third_item_part_user_graph_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='thirdGCN_IU_customer_MLP_first_layer')

        self.third_item_part_user_graph_att_layer2 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='thirdGCN_IU_customer_MLP_second_layer')          #### [item, user]
        ######################################################################################
        ######################################################################################
        ######################################################################################
        self.third_item_part_item_graph_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='thirdGCN_II_item_MLP_first_layer')
        self.third_item_part_item_graph_att_layer2 = tf.layers.Dense(\
            1,activation=tf.nn.leaky_relu, name='thirdGCN_II_item_MLP_second_layer')                #### [item, item]




                        ####
                        ####
                        #### 第4层
        self.fourth_item_part_itself_graph_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='fourthGCN_IU_itemself_MLP_first_layer')

        self.fourth_item_part_itself_graph_att_layer2 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='fourthGCN_IU_itemself_MLP_second_layer')          #### item-self

        self.fourth_item_part_user_graph_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='fourthGCN_IU_customer_MLP_first_layer')

        self.fourth_item_part_user_graph_att_layer2 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='fourthGCN_IU_customer_MLP_second_layer')          #### [item, user]
        ######################################################################################
        ######################################################################################
        ######################################################################################
        self.fourth_item_part_item_graph_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='fourthGCN_II_item_MLP_first_layer')
        self.fourth_item_part_item_graph_att_layer2 = tf.layers.Dense(\
            1,activation=tf.nn.leaky_relu, name='fourthGCN_II_item_MLP_second_layer')                #### [item, item]



                        ####
                        ####
                        #### 第5层
        self.fifth_item_part_itself_graph_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='fifthGCN_IU_itemself_MLP_first_layer')

        self.fifth_item_part_itself_graph_att_layer2 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='fifthGCN_IU_itemself_MLP_second_layer')          #### item-self

        self.fifth_item_part_user_graph_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='fifthGCN_IU_customer_MLP_first_layer')

        self.fifth_item_part_user_graph_att_layer2 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='fifthGCN_IU_customer_MLP_second_layer')          #### [item, user]
        ######################################################################################
        ######################################################################################
        ######################################################################################
        self.fifth_item_part_item_graph_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='fifthGCN_II_item_MLP_first_layer')
        self.fifth_item_part_item_graph_att_layer2 = tf.layers.Dense(\
            1,activation=tf.nn.leaky_relu, name='fifthGCN_II_item_MLP_second_layer')                #### [item, item]




    def convertDistribution(self, x):
        mean, var = tf.nn.moments(x, axes=[0, 1])
                            #### 
        y = (x - mean) * 0.1 / tf.sqrt(var)
        return y

    def constructTrainGraph(self):

        ########  Fusion Layer ########

        first_user_review_vector_matrix = self.convertDistribution(self.user_review_vector_matrix)              #### user
        first_item_review_vector_matrix = self.convertDistribution(self.item_review_vector_matrix)              #### item
        
        self.user_reduce_dim_vector_matrix = self.reduce_dimension_layer(first_user_review_vector_matrix)       #### 
        self.item_reduce_dim_vector_matrix = self.reduce_dimension_layer(first_item_review_vector_matrix)

        second_user_review_vector_matrix = self.convertDistribution(self.user_reduce_dim_vector_matrix)
        second_item_review_vector_matrix = self.convertDistribution(self.item_reduce_dim_vector_matrix)

        # self.fusion_item_embedding = tf.Variable(self.item_embedding + second_item_review_vector_matrix, name='self.fusion_item_embedding')                     #### 
        # self.fusion_user_embedding = tf.Variable(self.user_embedding + second_user_review_vector_matrix, name='self.fusion_user_embedding')                     ####

        self.i_0 = self.item_embedding + second_item_review_vector_matrix                    #### 
        self.u_0 = self.user_embedding + second_user_review_vector_matrix                    #### 

        # self.u_0 = self.user_embedding
        # self.i_0 = self.item_embedding

        ######## Influence and Interest Diffusion Layer ########

        # ----------------------
        ####
        ####
        #### First Layer
                            ######## user
        u_from_u_1 = self.generateUserEmbeddingFromSocialNeighbors1(self.u_0)                   #### [user, user]
        u_from_i_1 = self.generateUserEmebddingFromConsumedItems1(self.i_0)                       #### [user, item]

        #############################################################################################
        #############################################################################################
        #############################################################################################
        u_from_u_uiu = self.generateUserEmbeddingFrom_uiu_1(self.u_0)                  ######## uiu
        u_from_u_u0u = self.generateUserEmbeddingFrom_u0u_1(self.u_0)                  ######## u0u

        uiu_att_1 = tf.math.exp(self.first_user_user_uiu_subgraph_att_layer2(self.first_user_user_uiu_subgraph_att_layer1( \
                                    tf.concat([self.u_0, u_from_u_uiu], 1))))  + 0.7               #### uiu
        u0u_att_1 = tf.math.exp(self.first_user_user_u0u_subgraph_att_layer2(self.first_user_user_u0u_subgraph_att_layer1( \
                                    tf.concat([self.u_0, u_from_u_u0u], 1)))) + 0.3                    #### u0u
        # u_from_u_att_sum_1 = (uiu_att_1 + u0u_att_1) / 2
        # u_from_u_1 = uiu_att_1 / u_from_u_att_sum_1 * u_from_u_uiu \
        #             + u0u_att_1 / u_from_u_att_sum_1 * u_from_u_u0u

        u_from_u_1 = u_from_u_uiu * 1.1 + u_from_u_u0u * 0.9


        u_u_att_1 = tf.math.exp(self.first_user_part_social_graph_att_layer2(self.first_user_part_social_graph_att_layer1(\
                                    tf.concat([self.u_0, u_from_u_1], 1)))) + 0.3                       #### [user, user]
                                        #### 
        u_i_att_1 = tf.math.exp(self.first_user_part_interest_graph_att_layer2(self.first_user_part_interest_graph_att_layer1(\
                                    tf.concat([self.u_0, u_from_i_1], axis = 1)))) + 0.7                  #### [user, item]
                                        #### 
        uu_att_1 = tf.math.exp(self.first_user_user_self_att_layer2(self.first_user_user_self_att_layer1( \
                                    tf.concat([self.u_0, self.u_0], 1)))) + 1.0                                 #### user self
        
        # user_sum_attention_1 = social_neighbors_attention_1 + consumed_items_attention_1 + user_self_attention_1
        # first_gcn_user_embedding = social_neighbors_attention_1 / user_sum_attention_1 * user_embedding_from_social_neighbors_1 \
        #                          + consumed_items_attention_1 / user_sum_attention_1 * user_embedding_from_consumed_items_1 \
        #                          + user_self_attention_1 / user_sum_attention_1 * self.fusion_user_embedding    

        u_att_sum_1 = u_u_att_1 + u_i_att_1
        u_1 = 0.5 * self.u_0 + \
                            0.5 * ( u_u_att_1 / u_att_sum_1 * u_from_u_1 \
                                    + u_i_att_1 / u_att_sum_1 * u_from_i_1)                                        #### USER_1

        # u_1 = u_from_u_1
        # u_att_sum_1 = u_u_att_1 + u_i_att_1 + uu_att_1
        # u_1 = uu_att_1 / u_att_sum_1 * self.u_0 + u_u_att_1 / u_att_sum_1 * u_from_u_1 + u_i_att_1 / u_att_sum_1 * u_from_i_1


        # first_mean_social_influ, first_var_social_influ = tf.nn.moments(self.social_neighbors_attention_1, axes=0)                      #### user-user 
        # first_mean_interest_influ, first_var_interest_influ = tf.nn.moments(self.consumed_items_attention_1, axes=0)                    #### user-item 
        # self.first_layer_analy = [first_mean_social_influ, first_var_social_influ,\
        #                           first_mean_interest_influ, first_var_interest_influ]                      
                            ######## item 为中心
                   
        
        i_from_u_1 = self.generateItemEmebddingFromCustomer1(self.u_0)
                                     
        #############################################################################################
        #############################################################################################
        #############################################################################################
        i_from_i_1 = self.generateItemEmbeddingFromItem1(self.i_0)
      
        i_i_att_1 = tf.math.exp(self.first_item_part_item_graph_att_layer2(self.first_item_part_item_graph_att_layer1(\
                                    tf.concat([self.i_0, i_from_i_1], 1)))) + 0.3
        i_u_att_1 = tf.math.exp(self.first_item_part_item_graph_att_layer2(self.first_item_part_item_graph_att_layer1(\
                                    tf.concat([self.i_0, i_from_u_1], 1)))) + 0.7
        ii_att_1 = tf.math.exp(self.first_item_part_itself_graph_att_layer2(\
                            self.first_item_part_itself_graph_att_layer1(self.i_0))) + 1.0                              #### 

        i_att_sum_1 = ii_att_1 + i_i_att_1 + i_u_att_1
        i_1 = ii_att_1 / i_att_sum_1 * self.i_0 \
                                    + i_i_att_1 / i_att_sum_1 * i_from_i_1 \
                                    + i_u_att_1 / i_att_sum_1 * i_from_u_1


        # i_u_att_1 = tf.math.exp(self.first_item_part_item_graph_att_layer2(self.first_item_part_item_graph_att_layer1(\
        #                             tf.concat([self.i_0, i_from_u_1], 1)))) + 1.0
        # ii_att_1 = tf.math.exp(self.first_item_part_itself_graph_att_layer2(\
        #                     self.first_item_part_itself_graph_att_layer1(self.i_0))) + 1.0                              #### 
        # i_att_sum_1 = ii_att_1 + i_u_att_1
        # i_1 = ii_att_1 / i_att_sum_1 * self.i_0 \
        #                             + i_u_att_1 / i_att_sum_1 * i_from_u_1     

        # i_1 = i_from_i_1

        # i_att_sum_1 = i_i_att_1 + i_u_att_1
        # i_1 = 1/2 * self.i_0 + 1/2 * (i_i_att_1 / i_att_sum_1 * i_from_i_1 + i_u_att_1 / i_att_sum_1 * i_from_u_1)

        # item_sum_attention_1 = item_itself_att_1 + item_customer_attenton_1    
        # first_gcn_item_embedding = item_itself_att_1 / item_sum_attention_1 * self.fusion_item_embedding + \
        #                 item_customer_attenton_1 / item_sum_attention_1 * item_embedding_from_consumer_1              #### ITEM_2

        # first_mean_social_influ1, first_var_social_influ1 = tf.nn.moments(self.item_itself_att1, axes=0)                                #### item-self      
        # first_mean_interest_influ1, first_var_interest_influ1 = tf.nn.moments(self.item_customer_attenton1, axes=0)                     #### item-user 
        # self.first_layer_item_analy = [first_mean_social_influ1, first_var_social_influ1,\
        #                           first_mean_interest_influ1, first_var_interest_influ1]

        # ----------------------
        ####
        ####
        ##### Second Layer
                            ######## user为中心
        u_from_u_2 = self.generateUserEmbeddingFromSocialNeighbors2(u_1)
        u_from_i_2 = self.generateUserEmebddingFromConsumedItems2(i_1)

        #############################################################################################
        #############################################################################################
        #############################################################################################
        u_from_u_uiu = self.generateUserEmbeddingFrom_uiu_2(u_1)                  ######## uiu
        u_from_u_u0u = self.generateUserEmbeddingFrom_u0u_2(u_1)                  ######## u0u

        uiu_att_2 = tf.math.exp(self.second_user_user_uiu_subgraph_att_layer2(self.second_user_user_uiu_subgraph_att_layer1(\
                                    tf.concat([u_1, u_from_u_uiu], 1)))) + 0.7                   #### uiu
        u0u_att_2 = tf.math.exp(self.second_user_user_u0u_subgraph_att_layer2(self.second_user_user_u0u_subgraph_att_layer1(\
                                    tf.concat([u_1, u_from_u_u0u], 1)))) + 0.3                    #### u0u
        # u_from_u_att_sum_2 = (uiu_att_2 + u0u_att_2) / 2
        # u_from_u_2 = uiu_att_2 / u_from_u_att_sum_2 * u_from_u_uiu \
        #     + u0u_att_2 / u_from_u_att_sum_2 * u_from_u_u0u

        u_from_u_2 = u_from_u_uiu * 1.1 + u_from_u_u0u * 0.9


        u_u_att_2 = tf.math.exp(self.second_user_part_social_graph_att_layer2(self.second_user_part_social_graph_att_layer1(\
                                    tf.concat([u_1, u_from_u_2], 1)))) + 0.3
        u_i_att_2 = tf.math.exp(self.second_user_part_interest_graph_att_layer2(self.second_user_part_interest_graph_att_layer1(\
                                    tf.concat([u_1, u_from_i_2], 1)))) + 0.7
        uu_att_2 = tf.math.exp(self.second_user_user_self_att_layer2(self.second_user_user_self_att_layer1( \
                                    tf.concat([u_1, u_1], 1))))                                 #### user self
        
        # user_sum_attention_2 = social_neighbors_attention_2 + consumed_items_attention_2 + user_self_attention_2
        # second_gcn_user_embedding = social_neighbors_attention_2 / user_sum_attention_2 * user_embedding_from_social_neighbors \
        #                          + consumed_items_attention_2 / user_sum_attention_2 * user_embedding_from_consumed_items \
        #                          + user_self_attention_2 / user_sum_attention_2 * first_gcn_user_embedding    

        u_att_sum_2 = u_u_att_2 + u_i_att_2
        u_2 = 0.5 * u_1 \
                + 0.5 * (u_u_att_2 / u_att_sum_2 * u_from_u_2 \
                    + u_i_att_2 / u_att_sum_2 * u_from_i_2 )                                      #### USER_2
        # u_2 = u_from_u_2
            
        # u_att_sum_2 = u_u_att_2 + u_i_att_2 + uu_att_2
        # u_2 = uu_att_2 / u_att_sum_2 * u_1 + u_u_att_2 / u_att_sum_2 * u_from_u_2 + u_i_att_2 / u_att_sum_2 * u_from_i_2

        # second_mean_social_influ, second_var_social_influ = tf.nn.moments(self.social_neighbors_attention_2, axes=0)                    #### user-user 
        # second_mean_interest_influ, second_var_interest_influ = tf.nn.moments(self.consumed_items_attention_2, axes=0)                  #### user-item 
        # self.second_layer_analy = [second_mean_social_influ, second_var_social_influ,\
        #                            second_mean_interest_influ,second_var_interest_influ]


        i_from_u_2 = self.generateItemEmebddingFromCustomer2(u_1)                                     
        #############################################################################################
        #############################################################################################
        #############################################################################################
        i_from_i_2 = self.generateItemEmbeddingFromItem2(i_1)

        # i_u_att_2 = tf.math.exp(self.second_item_part_user_graph_att_layer2(\
        #                   self.second_item_part_user_graph_att_layer1(i_from_u_2))) + 1.0                           #### [item, user]
        i_u_att_2 = tf.math.exp(self.second_item_part_item_graph_att_layer2(self.second_item_part_item_graph_att_layer1(\
                                    tf.concat([i_1, i_from_u_2], 1)))) + 0.3
        i_i_att_2 = tf.math.exp(self.second_item_part_item_graph_att_layer2(self.second_item_part_item_graph_att_layer1(\
                                    tf.concat([i_1, i_from_i_2], 1)))) + 0.7
        ii_att_2 = tf.math.exp(self.second_item_part_itself_graph_att_layer2(\
                          self.second_item_part_itself_graph_att_layer1(i_1))) + 1.0                              ####   

        # item_sum_attention_2 = item_item_attention_2 + item_user_attention_2
        # second_gcn_item_embedding = 1/2 * first_gcn_item_embedding \
        #      +  1/2 * ( item_item_attention_2 / item_sum_attention_2 * item_embedding_from_item_2 \
        #              + item_user_attention_2 /item_sum_attention_2 * item_embedding_from_consumer_2)                                        #### USER_2

        i_att_sum_2 = i_u_att_2 + i_i_att_2 + ii_att_2
        i_2 = ii_att_2 / i_att_sum_2 * i_1 \
                                    + i_i_att_2 / i_att_sum_2 * i_from_i_2 \
                                    + i_u_att_2 / i_att_sum_2 * i_from_u_2



        # i_u_att_2 = tf.math.exp(self.second_item_part_item_graph_att_layer2(self.second_item_part_item_graph_att_layer1(\
        #                             tf.concat([i_1, i_from_u_2], 1)))) + 1.0
        # ii_att_2 = tf.math.exp(self.second_item_part_itself_graph_att_layer2(\
        #                   self.second_item_part_itself_graph_att_layer1(i_1))) + 1.0                              ####  
        # i_att_sum_2 = ii_att_2 + i_u_att_2
        # i_2 = ii_att_2 / i_att_sum_2 * i_1 \
        #                             + i_u_att_2 / i_att_sum_2 * i_from_u_2  

        # i_2 = i_from_i_2

        # i_att_sum_2 = i_i_att_2 + i_u_att_2
        # i_2 = 1/2 * i_1 + 1/2 * (i_i_att_2 / i_att_sum_2 * i_from_i_2 + i_u_att_2 / i_att_sum_2 * i_from_u_2)

        # item_sum_attention = item_itself_att + item_customer_attenton

        # self.item_itself_att2 = item_itself_att / item_sum_attention
        # self.item_customer_attenton2 = item_customer_attenton / item_sum_attention

        # second_gcn_item_embedding = self.item_itself_att2 * first_gcn_item_embedding + \
        #     self.item_customer_attenton2 * self.generateItemEmebddingFromCustomer2(first_gcn_user_embedding)                            #### ITEM_2

        # first_mean_social_influ2, first_var_social_influ2 = tf.nn.moments(self.item_itself_att2, axes=0)                                #### item-self 
        # first_mean_interest_influ2, first_var_interest_influ2 = tf.nn.moments(self.item_customer_attenton2, axes=0)                     #### item_user 
        # self.second_layer_item_analy = [first_mean_social_influ2, first_var_social_influ2,\
        #                           first_mean_interest_influ2, first_var_interest_influ2]




        # # ----------------------
        # ####
        # ####
        # #### Third Layer
        #                     ######## user为中心
        # u_from_i_3 = self.generateUserEmebddingFromConsumedItems3(i_2)

        # #############################################################################################
        # #############################################################################################
        # #############################################################################################
        # u_from_u_uiu = self.generateUserEmbeddingFrom_uiu_3(u_2)                  ######## uiu
        # u_from_u_u0u = self.generateUserEmbeddingFrom_u0u_3(u_2)                  ######## u0u

        # uiu_att_3 = tf.math.exp(self.third_user_user_uiu_subgraph_att_layer2(self.third_user_user_uiu_subgraph_att_layer1(\
        #                             tf.concat([u_2, u_from_u_uiu], 1)))) + 0.7                   #### uiu
        # u0u_att_3 = tf.math.exp(self.third_user_user_u0u_subgraph_att_layer2(self.third_user_user_u0u_subgraph_att_layer1(\
        #                             tf.concat([u_2, u_from_u_u0u], 1)))) + 0.3                    #### u0u

        # u_from_u_3 = u_from_u_uiu * 1.1 + u_from_u_u0u * 0.9


        # u_u_att_3 = tf.math.exp(self.third_user_part_social_graph_att_layer2(self.third_user_part_social_graph_att_layer1(\
        #                             tf.concat([u_2, u_from_u_3], 1)))) + 0.3
        # u_i_att_3 = tf.math.exp(self.third_user_part_interest_graph_att_layer2(self.third_user_part_interest_graph_att_layer1(\
        #                             tf.concat([u_2, u_from_i_3], 1)))) + 0.7
        # uu_att_3 = tf.math.exp(self.third_user_user_self_att_layer2(self.third_user_user_self_att_layer1( \
        #                             tf.concat([u_2, u_2], 1))))                                 #### user self

        # u_att_sum_3 = u_u_att_3 + u_i_att_3
        # u_3 = 1/2 * u_2 \
        #         + 1/2 * (u_u_att_3 / u_att_sum_3 * u_from_u_3 \
        #             + u_i_att_3 / u_att_sum_3 * u_from_i_3 )                                      #### USER_2



        # i_from_u_3 = self.generateItemEmebddingFromCustomer3(u_2)                                     
        # #############################################################################################
        # #############################################################################################
        # #############################################################################################
        # i_from_i_3 = self.generateItemEmbeddingFromItem3(i_2)

        # i_u_att_3 = tf.math.exp(self.third_item_part_item_graph_att_layer2(self.third_item_part_item_graph_att_layer1(\
        #                             tf.concat([i_2, i_from_u_3], 1)))) + 0.3
        # i_i_att_3 = tf.math.exp(self.third_item_part_item_graph_att_layer2(self.third_item_part_item_graph_att_layer1(\
        #                             tf.concat([i_2, i_from_i_3], 1)))) + 0.7
        # ii_att_3 = tf.math.exp(self.third_item_part_itself_graph_att_layer2(\
        #                   self.third_item_part_itself_graph_att_layer1(i_2))) + 1.0                              #### 

        # i_att_sum_3 = i_u_att_3 + i_i_att_3 + ii_att_3
        # i_3 = ii_att_3 / i_att_sum_3 * i_2 \
        #                             + i_i_att_3 / i_att_sum_3 * i_from_i_3 \
        #                             + i_u_att_3 / i_att_sum_3 * i_from_u_3

        # # i_att_sum_3 = i_u_att_3 + ii_att_3
        # # i_3 = ii_att_3 / i_att_sum_3 * i_2 \
        # #                             + i_u_att_3 / i_att_sum_3 * i_from_u_3


        # # ----------------------
        # ####
        # #### Fourth Layer
        #                     ######## user为中心
        # u_from_i_4 = self.generateUserEmebddingFromConsumedItems4(i_3)
        # #############################################################################################
        # #############################################################################################
        # #############################################################################################
        # u_from_u_uiu = self.generateUserEmbeddingFrom_uiu_4(u_3)                  ######## uiu
        # u_from_u_u0u = self.generateUserEmbeddingFrom_u0u_4(u_3)                  ######## u0u

        # uiu_att_4 = tf.math.exp(self.fourth_user_user_uiu_subgraph_att_layer2(self.fourth_user_user_uiu_subgraph_att_layer1(\
        #                             tf.concat([u_3, u_from_u_uiu], 1)))) + 0.7                   #### uiu
        # u0u_att_4 = tf.math.exp(self.fourth_user_user_u0u_subgraph_att_layer2(self.fourth_user_user_u0u_subgraph_att_layer1(\
        #                             tf.concat([u_3, u_from_u_u0u], 1)))) + 0.3                    #### u0u

        # u_from_u_4 = u_from_u_uiu * 1.1 + u_from_u_u0u * 0.9


        # u_u_att_4 = tf.math.exp(self.fourth_user_part_social_graph_att_layer2(self.fourth_user_part_social_graph_att_layer1(\
        #                             tf.concat([u_3, u_from_u_4], 1)))) + 0.3
        # u_i_att_4 = tf.math.exp(self.fourth_user_part_interest_graph_att_layer2(self.fourth_user_part_interest_graph_att_layer1(\
        #                             tf.concat([u_3, u_from_i_4], 1)))) + 0.7
        # uu_att_4 = tf.math.exp(self.fourth_user_user_self_att_layer2(self.fourth_user_user_self_att_layer1( \
        #                             tf.concat([u_3, u_3], 1))))                                 #### user self
        

        # u_att_sum_4 = u_u_att_4 + u_i_att_4
        # u_4 = 1/2 * u_3 \
        #         + 1/2 * (u_u_att_4 / u_att_sum_4 * u_from_u_4 \
        #             + u_i_att_4 / u_att_sum_4 * u_from_i_4 )                                      #### USER_2




        # i_from_u_4 = self.generateItemEmebddingFromCustomer4(u_3)                                     
        # #############################################################################################
        # #############################################################################################
        # #############################################################################################
        # i_from_i_4 = self.generateItemEmbeddingFromItem4(i_3)


        # i_u_att_4 = tf.math.exp(self.fourth_item_part_item_graph_att_layer2(self.fourth_item_part_item_graph_att_layer1(\
        #                             tf.concat([i_3, i_from_u_4], 1)))) + 0.3
        # i_i_att_4 = tf.math.exp(self.fourth_item_part_item_graph_att_layer2(self.fourth_item_part_item_graph_att_layer1(\
        #                             tf.concat([i_3, i_from_i_4], 1)))) + 0.7
        # ii_att_4 = tf.math.exp(self.fourth_item_part_itself_graph_att_layer2(\
        #                   self.fourth_item_part_itself_graph_att_layer1(i_3))) + 1.0                              ####    

        # i_att_sum_4 = i_u_att_4 + i_i_att_4 + ii_att_4
        # i_4 = ii_att_4 / i_att_sum_4 * i_3 \
        #                             + i_i_att_4 / i_att_sum_4 * i_from_i_4 \
        #                             + i_u_att_4 / i_att_sum_4 * i_from_u_4

        # # i_att_sum_4 = i_u_att_4 + ii_att_4
        # # i_4 = ii_att_4 / i_att_sum_4 * i_3 \
        # #                             + i_u_att_4 / i_att_sum_4 * i_from_u_4


        # # ----------------------
        # ####
        # #### Fifth Layer
        #                     ######## user为中心
        # u_from_i_5 = self.generateUserEmebddingFromConsumedItems5(i_4)
        # #############################################################################################
        # #############################################################################################
        # #############################################################################################
        # u_from_u_uiu = self.generateUserEmbeddingFrom_uiu_5(u_4)                  ######## uiu
        # u_from_u_u0u = self.generateUserEmbeddingFrom_u0u_5(u_4)                  ######## u0u

        # uiu_att_5 = tf.math.exp(self.fifth_user_user_uiu_subgraph_att_layer2(self.fifth_user_user_uiu_subgraph_att_layer1(\
        #                             tf.concat([u_4, u_from_u_uiu], 1)))) + 0.7                   #### uiu
        # u0u_att_5 = tf.math.exp(self.fifth_user_user_u0u_subgraph_att_layer2(self.fifth_user_user_u0u_subgraph_att_layer1(\
        #                             tf.concat([u_4, u_from_u_u0u], 1)))) + 0.3                    #### u0u

        # u_from_u_5 = u_from_u_uiu * 1.1 + u_from_u_u0u * 0.9


        # u_u_att_5 = tf.math.exp(self.fifth_user_part_social_graph_att_layer2(self.fifth_user_part_social_graph_att_layer1(\
        #                             tf.concat([u_4, u_from_u_5], 1)))) + 0.3
        # u_i_att_5 = tf.math.exp(self.fifth_user_part_interest_graph_att_layer2(self.fifth_user_part_interest_graph_att_layer1(\
        #                             tf.concat([u_4, u_from_i_5], 1)))) + 0.7
        # uu_att_5 = tf.math.exp(self.fifth_user_user_self_att_layer2(self.fifth_user_user_self_att_layer1( \
        #                             tf.concat([u_4, u_4], 1))))                                 #### user self
        

        # u_att_sum_5 = u_u_att_5 + u_i_att_5
        # u_5 = 1/2 * u_4 \
        #         + 1/2 * (u_u_att_5 / u_att_sum_5 * u_from_u_5 \
        #             + u_i_att_5 / u_att_sum_5 * u_from_i_5 )                                      #### USER_2




        # i_from_u_5 = self.generateItemEmebddingFromCustomer5(u_4)                                     
        # #############################################################################################
        # #############################################################################################
        # #############################################################################################
        # i_from_i_5 = self.generateItemEmbeddingFromItem4(i_4)


        # i_u_att_5 = tf.math.exp(self.fifth_item_part_item_graph_att_layer2(self.fifth_item_part_item_graph_att_layer1(\
        #                             tf.concat([i_4, i_from_u_5], 1)))) + 0.3
        # i_i_att_5 = tf.math.exp(self.fifth_item_part_item_graph_att_layer2(self.fifth_item_part_item_graph_att_layer1(\
        #                             tf.concat([i_4, i_from_i_5], 1)))) + 0.7
        # ii_att_5 = tf.math.exp(self.fifth_item_part_itself_graph_att_layer2(\
        #                   self.fifth_item_part_itself_graph_att_layer1(i_4))) + 1.0                              ####    

        # i_att_sum_5 = i_u_att_5 + i_i_att_5 + ii_att_5
        # i_5 = ii_att_5 / i_att_sum_5 * i_4 \
        #                             + i_i_att_5 / i_att_sum_5 * i_from_i_5 \
        #                             + i_u_att_5 / i_att_sum_5 * i_from_u_5

        # i_att_sum_5 = i_u_att_5 + ii_att_5
        # i_5 = ii_att_5 / i_att_sum_5 * i_4 \
        #                             + i_u_att_5 / i_att_sum_5 * i_from_u_5


        ######## Prediction Layer ########
        # self.final_user_embedding = \
        #     tf.concat([first_gcn_user_embedding, second_gcn_user_embedding, self.user_embedding, second_user_review_vector_matrix], 1)
        # self.final_item_embedding = \
        #     tf.concat([first_gcn_item_embedding, second_gcn_item_embedding, self.item_embedding, second_item_review_vector_matrix], 1)
        
        # self.final_user_embedding = second_gcn_user_embedding
        # self.final_item_embedding = second_gcn_item_embedding
        
        # print(self.final_user_embedding)        # shape: (17237, 256)
        # print(self.user_input)                  # shape: (?, 1)      

        ##################################################################################################
        ##################################################################################################
        ##################################################################################################
        # def a_add_b(a, b):
        #     return tf.Variable(a + b)

        # uv1 = a_add_b(first_gcn_user_embedding, second_user_review_vector_matrix)
        # uv2 = a_add_b(second_gcn_user_embedding, second_user_review_vector_matrix)
        # u01 = a_add_b(self.fusion_user_embedding, first_gcn_user_embedding)
        # u02 = a_add_b(self.fusion_user_embedding, second_gcn_user_embedding)
        # u12 = a_add_b(first_gcn_user_embedding, second_gcn_user_embedding)

        # print(uv1)
        # print(u01)
        # print(self.user_embedding)
        # print(second_user_review_vector_matrix)

        # iv1 = a_add_b(first_gcn_item_embedding, second_item_review_vector_matrix)
        # iv2 = a_add_b(second_gcn_item_embedding, second_item_review_vector_matrix)
        # i01 = a_add_b(self.fusion_item_embedding, first_gcn_item_embedding)
        # i02 = a_add_b(self.fusion_item_embedding, second_gcn_item_embedding)
        # i12 = a_add_b(first_gcn_item_embedding, second_gcn_item_embedding)

        # self.final_user_embedding = \
        #     tf.concat([u_1, u_2, self.user_embedding, second_user_review_vector_matrix, self.u_0], 1)
        # self.final_item_embedding = \
        #     tf.concat([i_1, i_2, self.item_embedding, second_item_review_vector_matrix, self.i_0], 1)

        # self.final_user_embedding = tf.concat([self.u_0, u_1, u_2, self.u_0+u_1+u_2], 1)
        # self.final_item_embedding = tf.concat([self.i_0, i_1, i_2, self.i_0+i_1+i_2], 1)

        # self.final_user_embedding = tf.concat([self.u_0 + u_1 + u_2, u_from_i_1, u_from_i_2], 1)
        # self.final_item_embedding = tf.concat([self.i_0 + i_1 + i_2, i_from_u_1, i_from_u_2], 1)


        ####
        ##### depth K = 0
        # self.final_user_embedding = tf.concat([self.u_0, self.u_0, second_user_review_vector_matrix], 1)
        # self.final_item_embedding = tf.concat([self.i_0, self.i_0, second_item_review_vector_matrix], 1)

        ##### depth K = 1
        # self.final_user_embedding = tf.concat([self.u_0, u_1, self.u_0+u_1, second_user_review_vector_matrix], 1)
        # self.final_item_embedding = tf.concat([self.i_0, i_1, self.i_0+i_1, second_item_review_vector_matrix], 1)

        ####  depth K = 2
        self.final_user_embedding = tf.concat([self.u_0, u_1, u_2, self.u_0+u_1+u_2, second_user_review_vector_matrix], 1)
        self.final_item_embedding = tf.concat([self.i_0, i_1, i_2, self.i_0+i_1+i_2, second_item_review_vector_matrix], 1)

        # ##### depth K = 3
        # self.final_user_embedding = tf.concat([self.u_0, u_1, u_2, u_3, self.u_0+u_1+u_2+u_3, second_user_review_vector_matrix], 1)
        # self.final_item_embedding = tf.concat([self.i_0, i_1, i_2, i_3, self.i_0+i_1+i_2+i_3, second_item_review_vector_matrix], 1)

        # ##### depth K = 4
        # self.final_user_embedding = tf.concat([self.u_0, u_1, u_2, u_3, u_4, self.u_0+u_1+u_2+u_3+u_4, second_user_review_vector_matrix], 1)
        # self.final_item_embedding = tf.concat([self.i_0, i_1, i_2, i_3, i_4, self.i_0+i_1+i_2+i_3+i_4, second_item_review_vector_matrix], 1)

        # ##### depth K = 5
        # self.final_user_embedding = tf.concat([self.u_0, u_1, u_2, u_3, u_4, u_5, self.u_0+u_1+u_2+u_3+u_4+u_5, second_user_review_vector_matrix], 1)
        # self.final_item_embedding = tf.concat([self.i_0, i_1, i_2, i_3, i_4, i_5, self.i_0+i_1+i_2+i_3+i_4+i_5, second_item_review_vector_matrix], 1)



        latest_user_latent = tf.gather_nd(self.final_user_embedding, self.user_input)
        latest_item_latent = tf.gather_nd(self.final_item_embedding, self.item_input)
                                #### 

        self.predict_vector = tf.multiply(latest_user_latent, latest_item_latent)
                                        #### 
        self.prediction = tf.sigmoid(tf.reduce_sum(self.predict_vector, 1, keepdims=True))
                                        #### tf.reduce_sum()  
                                        # 
                                        # keepdims=True:

        # ----------------------
        # Optimazation

        self.loss = tf.nn.l2_loss(self.labels_input - self.prediction)
                            #### l2
        self.opt_loss = tf.nn.l2_loss(self.labels_input - self.prediction)
        self.opt = tf.train.AdamOptimizer(self.conf.learning_rate).minimize(self.opt_loss)
                            #### Adam
        self.init = tf.global_variables_initializer()

    def saveVariables(self):
        ############################# Save Variables #################################
        variables_dict = {}
        variables_dict[self.user_embedding.op.name] = self.user_embedding
        variables_dict[self.item_embedding.op.name] = self.item_embedding

        for v in self.reduce_dimension_layer.variables:
            variables_dict[v.op.name] = v
                
        self.saver = tf.train.Saver(variables_dict)
                        #### 

        ############################# Save Variables #################################
    
    def defineMap(self):
        map_dict = {}
        map_dict['train'] = {
            self.user_input: 'USER_LIST', 
            self.item_input: 'ITEM_LIST', 
            self.labels_input: 'LABEL_LIST'
        }
        
        map_dict['val'] = {
            self.user_input: 'USER_LIST', 
            self.item_input: 'ITEM_LIST', 
            self.labels_input: 'LABEL_LIST'
        }

        map_dict['test'] = {
            self.user_input: 'USER_LIST', 
            self.item_input: 'ITEM_LIST', 
            self.labels_input: 'LABEL_LIST'
        }

        map_dict['eva'] = {
            self.user_input: 'EVA_USER_LIST', 
            self.item_input: 'EVA_ITEM_LIST'
        }

        map_dict['out'] = {
            'train': self.loss,
            'val': self.loss,
            'test': self.loss,
            'eva': self.prediction, 
            # 'first_layer_ana': self.first_layer_analy, 
            # 'second_layer_ana': self.second_layer_analy,
            # 'first_layer_item_ana': self.first_layer_item_analy,
            # 'second_layer_item_ana': self.second_layer_item_analy,                                ##### 
            'prediction': self.predict_vector,
            'user':self.final_user_embedding,
            'item':self.final_item_embedding,
            # 'low_att_user_user': self.first_user_user_low_att,              ####
            # 'low_att_user_item': self.first_user_item_low_att,
            # 'low_att_user_user': self.first_item_user_low_att,
            'first_social_neighbors_low_att_matrix': self.first_social_neighbors_low_level_att_matrix,
            'second_social_neighbors_low_att_matrix': self.second_social_neighbors_low_level_att_matrix,
            'first_consumed_items_low_level_att_matrix':self.first_consumed_items_low_level_att_matrix,
            'second_consumed_items_low_level_att_matrix':self.second_consumed_items_low_level_att_matrix,
            'first_items_users_neighborslow_level_att_matrix':self.first_items_users_neighborslow_level_att_matrix,
            'second_items_users_neighborslow_level_att_matrix':self.second_items_users_neighborslow_level_att_matrix,

            #########################################################################################
            #########################################################################################
            #########################################################################################
            # 'first_user_user_uiu_low_level_att_matrix':self.first_user_user_uiu_low_level_att_matrix,
            # 'second_user_user_uiu_low_level_att_matrix':self.second_user_user_uiu_low_level_att_matrix,
            # 'first_user_user_u0u_low_level_att_matrix':self.first_user_user_u0u_low_level_att_matrix,
            # 'second_user_user_u0u_low_level_att_matrix':self.second_user_user_u0u_low_level_att_matrix,
            # 'first_item_item_low_level_att_matrix': self.first_item_item_low_level_att_matrix,
            # 'second_item_item_low_level_att_matrix': self.second_item_item_low_level_att_matrix,

        }

        self.map_dict = map_dict


