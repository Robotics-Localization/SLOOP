# -*- coding:UTF-8 -*-
import copy
# from multiprocessing import popen_spawn_win32
import cv2
import math
import random
import time, sys, os

from ros import rosbag
import roslib
roslib.load_manifest('sensor_msgs')
from sensor_msgs.msg import Image#, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pcl2

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

from mpl_toolkits.mplot3d import Axes3D
from math import *
from datetime import datetime

from sk2semantic_array import SKPreprocess
import open3d as o3d

dataset_is_scenenet_209 = 0
dataset_is_semantic_kitti = 1
debug = 0
src_frame = 0  #set the source frame

'''write the graph matching results to the file'''

dateTime_p =  datetime.now() # get the current time
str_p =  datetime.strftime(dateTime_p,'%Y-%m-%d-%h-%m-%s')
    
if dataset_is_scenenet_209 == 1:
    curr_path = os.path.abspath(os.path.dirname(__file__))
    f_GM_information = open(f"{curr_path}/../data/log/{str_p}ours1.txt", "a+")
if dataset_is_semantic_kitti == 1:
    curr_path = os.path.abspath(os.path.dirname(__file__))
    f_GM_information = open(f"{curr_path}/../data/log/05_neg100_{str_p}ours.txt", "a+")
    
query_frame_origin = 0  #query the frame from this number   ：）
if_search_specially_query_frame = 1 # whether select the start  query frame 

'''flag variable'''
show_ = 0 #the switch,  whether draw some graph results
show_vertex_graph = 1     #whether show the vertex graph
show_pointcloud =0 #show point cloud
points_src =[]               #point cloud of the src frame
points_query =[]        #point cloud of the query frame
only_show_once = 0   #show once and stop
show_common_semantic = 1  #whether the ratio of same semantics
#GM_first_parameter
frames = []
common_RGB_perceptions = []
GM_switch = 1 #the switch for graph matching


#SSGM_thresholds
max_two_stage_distance_thres = 3
w1 =0.5 ;  w2 = 0.5
first_threhold = 0.5  #the threshold of : the ratio of same semantics
spatial_consistency_threhold = 0.3     #m   0.3
r_threshold = 0.5 #m  2
theta_threshold = 2      #degree  8
fai_threshold_1 = 2       #degree  8
# fai_threshold_2 = 4      #degree  
distance_threshold = 15    #the distance threshold for measure the local features of selected vertex pairs
# local_dis_feature_threshold_1 = 3  # threshold of the local distance features: 构建顶点对
# local_dis_feature_threshold_2 = 5  # threshold of the local distance features：选二阶矩阵最好的
local_semantic_feature_threshold_ = 0.95         #这个也能用   0.95
angle_interval =3 # angle sample interval for selecting axes 

vector_z_axis_src = [] ;  vector_x_axis_src = [] ;  vector_y_axis_src = []     #three axes
vector_z_axis_query = [] ;  vector_x_axis_query = [] ;  vector_y_axis_query = []
P1 = 0  #the ratio of same semantics
P2 = 0  #max two order spatial consistency value
P3 = 0   #similar Spherical coordinate vertex pair

#scannet_10_01 random RGB
RGB_wall = [187,47,155]
RGB_desk = [243,233,81]
RGB_windows = [37,174,197]
RGB_floor = [7,4,196]

'''Visualization initialization'''
if show_:
    #ax  3D
    if not only_show_once :
        plt.ion()
    fig=plt.figure()    
    ax=Axes3D(fig)    
    
    #ax_query  3D
    fig_query=plt.figure()    
    ax_query=Axes3D(fig_query)    
    
    #ax1  2D
    fig_1=plt.figure()  
    #draw 2D curve：the ratio of same semantics
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax1 = fig_1.add_axes([left, bottom, width, height])
    

# get the fourth element of elem
def takeFourth(elem):
    return elem[3]

# get the seventh element of elem
def takeSeven(elem):
    return elem[6]

#remove the repeat element in list  according to the value in specific dimension
def DelRepeat(data,key):   #data: list ,  key:dimension 
    new_data = [] #   new list after duplicate removal
    key_values = []  #  store current values
    for d in data:
        if d[key] not in key_values:
            new_data.append(d)
            key_values.append(d[key])
    return new_data, key_values

#Average, maximum and minimum values of specific dimensions of elements within a certain range in the list
def Mean_Max_Min(data,key,start_index, end_index):   #data: list ,  key:dimensions  start_index, end_index: 
    sum_value = 0
    #search maximum and minimum values
    max_coordinate = 0
    min_coordinate = 0
    num_x0y0z0 =0 
    for d in data[start_index:end_index]:
        if float(d[0]) == 0 and float(d[1]) == 0 and float(d[2]) == 0:
            num_x0y0z0 = num_x0y0z0 + 1
            continue
        if max_coordinate < d[key]:
            max_coordinate = d[key]
        if min_coordinate > d[key]:
            min_coordinate = d[key]
        sum_value =sum_value + d[key]
    if end_index - start_index-num_x0y0z0 != 0:
        mean_value = sum_value / (end_index - start_index-num_x0y0z0) 
        continue_flag = 0       #  Whether to skip  generating the vertex, the flag variable = 0 means not to skip
    else:
        mean_value = 0
        continue_flag = 1     
        # print("mean_value, max_coordinate, min_coordinate", mean_value, max_coordinate, min_coordinate)
    return mean_value, max_coordinate, min_coordinate, continue_flag


#Downsampling a frame of point cloud (reducing the display time) and visualization
def down_sample_and_show(data,down_rate, ax_num):   #data: list,  key:dimensions
    down_sample_data_x = [] 
    down_sample_data_y = [] 
    down_sample_data_z = [] 
    down_sample_data_RGB = [] 
    for d in data[0::down_rate]:
        down_sample_data_x.append(d[0])
        down_sample_data_y.append(d[1])
        down_sample_data_z.append(d[2])
        down_sample_data_RGB.append( (float(d[3]/255) ,float(d[4]/255) ,float(d[5]/255) ,1) )
    down_sample_data_x= np.array(down_sample_data_x)
    down_sample_data_y= np.array(down_sample_data_y)
    down_sample_data_z= np.array(down_sample_data_z)
    #visualize the point cloud
    ax_num.scatter(down_sample_data_x, down_sample_data_z, -down_sample_data_y, c = down_sample_data_RGB, s = 1)

#Find the RGB value with the most points of the same RGB within a certain range in the list
def SearchRGB(data,start_index, end_index):   #data: list ,  key:dimensions 
    RGB_list = []  # store existing RGB
    for d in data[start_index:end_index]:
        # RGB = d[3:6]
        RGB_list.append(d[3:6])
    number = Counter(RGB_list) #Sort the quantity from large to small
    result = number.most_common()   # result like: [((0,0,0), 44340), ((10,100,23),2340), ...]
    # print("RGB_sort", result)
    if result[0][0] == (0,0,0): #If the most semantic points are unknown semantic points, it depends on what the second most semantic points are and the number of the second most semantic points
        if len(result) >1 :
            if result[0][1] >= 0.5*( result[1][1] + result[0][1] ) : 
                return result[0][0], result[0][1]
        else:  #If there is only unknown point cloud, return directly
            return result[0][0], result[0][1]
    else:
        return result[0][0], result[0][1]

#for each value in instance_list, find where it first appeared in data, with the dimension of key
def SearchIndex(data,instance_list, key):     #data: list , instance_list  ,key:dimensions
    ins_index_list = []  # store existing value
    instance_num = 0   #
    for Index in range(len(data)):
        if instance_num == len(instance_list):
            ins_index_list.append(len(data)-1)       #The index of the last point of the last instance
            break
        if data[Index][key] == instance_list[instance_num]:
            ins_index_list.append(Index)
            instance_num = instance_num + 1
    return  ins_index_list

def  get_RGBlist_from_vertexslist(vertexs):        #Vextex = [x_mean, y_mean, z_mean, R, G, B, instance,x_max, y_max, z_max, x_min, y_min, z_min]
    RGB_list = []
    for vertex in vertexs:
        R, G, B= vertex[3:6]
        RGB_list.append((R,G,B))
    return RGB_list

def print_vertexs_information(Vertexs):
    print("vertexs_information of the current frame: ")
    print("x_mean, y_mean, z_mean, R, G, B, instance,x_max, y_max, z_max, x_min, y_min, z_min")
    for v in Vertexs:
        print(v)

def show_vertexs_graph(vertexs_x_, vertexs_y_,vertexs_z_, color_vertexs_, src_flag, query_flag):
    vertexs_x_ = np.array(vertexs_x_)
    vertexs_y_ = np.array(vertexs_y_)
    vertexs_z_ = np.array(vertexs_z_)
    if src_flag ==1 :
        ax.scatter(vertexs_x_ , vertexs_z_, -vertexs_y_ , c = color_vertexs_ , s=100, alpha = 0.5, norm = 0.5)
    elif query_flag == 1:
        ax_query.scatter(vertexs_x_ , vertexs_z_, -vertexs_y_ , c = color_vertexs_ , s=100, alpha = 0.5, norm = 0.5)

def distance(x,y,z):    
    return math.sqrt(pow(x,2)+pow(y,2)+pow(z,2))

def plot_ax_parameter(only_show_once, ax_num):
    ax_num.set_xlabel('X')  
    ax_num.set_ylabel('Y')  
    ax_num.set_zlabel('Z')  
    ax_num.view_init(elev=0, azim=5)                # ax_num.view_init(elev=20, azim=0 ）          
    if not only_show_once :      
        # plt.savefig(str(frame)+".png")
        fig.canvas.draw() #update
        # time.sleep(2)
        ax_num.cla()   #remove old data
    else:   #
        plt.show()
        time.sleep(50)

def plot_ax_parameter_GM_Spherical(only_show_once, ax_num, ax_num1):
    ax_num.set_xlabel('X')  
    ax_num.set_ylabel('Y')  
    ax_num.set_zlabel('Z')  
    ax_num1.set_xlabel('X')  
    ax_num1.set_ylabel('Y')  
    ax_num1.set_zlabel('Z')  
    ax_num.view_init(elev=0, azim=5)                # ax_num.view_init(elev=20, azim=0 ） 
    ax_num1.view_init(elev=0, azim=5)                # ax_num.view_init(elev=20, azim=0 ）   

    if not only_show_once :      
        # plt.savefig(str(frame)+".png")
        fig.canvas.draw() #update
        # time.sleep(2)
        ax_num.cla()  #remove old data
        ax_num1.cla()  #remove old data
    else:  
        plt.show()
        time.sleep(5)


def  GM_first(vertexs_src, vertexs_query, frame_src, frame_query):        #V.append(Vextex)   Vextex = [x_mean, y_mean, z_mean, R, G, B, instance,x_max, y_max, z_max, x_min, y_min, z_min, density]
    first_remained_flag = 0   
    #obtain RGB values
    global RGBlist_src, RGBlist_query
    RGBlist_src = get_RGBlist_from_vertexslist(vertexs_src)
    RGBlist_query = get_RGBlist_from_vertexslist(vertexs_query)
    '''calculate common semantics / semantics in src or query (IOU)'''
    src_RGB_sort = set(RGBlist_src) 
    query_RGB_sort = set(RGBlist_query) 
    global common_RGB_sort
    common_RGB_sort = src_RGB_sort & query_RGB_sort

    tem_denominator = float( len( src_RGB_sort ) ) + float( len( query_RGB_sort ) ) - float( len( common_RGB_sort ) ) 
    common_RGB_perception =float( len(  common_RGB_sort) ) /   tem_denominator
    global P1 
    P1 = common_RGB_perception
    f_GM_information.write("P1 "+str(common_RGB_perception)+"\n")
    # print("The ",frame_src, "frame and the ", frame_query,"frame has the common RGB perception of :", len( common_RGB_sort ), " / ",len( src_RGB_sort )," : ",common_RGB_perception)
    frames.append(frame_query)
    common_RGB_perceptions.append(common_RGB_perception)

    if show_ and show_common_semantic:
        ax1.plot(frames, common_RGB_perceptions)
    #compare with the threshold
    if  common_RGB_perception >= first_threhold:
        first_remained_flag = 1
    
    return first_remained_flag, common_RGB_perceptions

#find the vertexs close to point ，return their semantics and distances
def most_near_class(point, vertexs):
    distance_list = []
    class_list = []
    for i in range(len(vertexs)):
        tem_point_i_src = np.array( vertexs[i] )
        tem_distance = np.linalg.norm(point[0:3] - tem_point_i_src[0:3])
        if tem_distance < distance_threshold and tem_distance != 0:
            distance_list.append(tem_distance)
            class_list.append(tem_point_i_src[3:6].tolist())
            # print(class_list)
        # print(point - tem_point_i_src)
    return class_list , distance_list

def matrix(m,n,initial):  
    return [[initial for j in range(n)] for i in range(m)]

def angle(v1, v2): #the angle of two vectors
    cos = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
    return np.arccos(cos)

def count(list):           #Return the different elements in the list and their occurrence times
    # count = 0 
    no_repeat_list = []  #A new list composed of non-repeating elements in list
    corresponding_num = [0 for i in range(len(list))] 
    if len(list) == 0 :
        return no_repeat_list, corresponding_num
    for i in range(len(list)):
        if list[i] not in no_repeat_list:
            no_repeat_list.append(list[i])
            corresponding_num[len(no_repeat_list)-1] += 1 
        else:
            Index = no_repeat_list.index(list[i])
            corresponding_num[Index] += 1 
    return no_repeat_list, corresponding_num

#输入周围点的语义列表和对应的距离列表
#输出  语义种类列表，对应的语义个数列表，各语义的平均距离列表
def count_new(semantic_list, distance_list):   
    # 统计语义的颜色列表
    unique_semantics_list, counts_list = count(semantic_list)
    counts_list = counts_list[:len(unique_semantics_list)]
    # 计算语义的个数和语义距离的平均值
    num_semantics = len(unique_semantics_list)
    mean_distances_list = []
    for s in unique_semantics_list:
        indices = [i for i, sublist in enumerate(semantic_list) if sublist == s]
        mean_distances_list.append( np.mean(np.array(distance_list)[indices]) )
    # print(unique_semantics_list, counts_list, mean_distances_list)
    return unique_semantics_list, counts_list, mean_distances_list

#cosine similarity
def similarity_dot_compare(no_repeat_class_src, corresponding_sem_feature_src, no_repeat_class_query, corresponding_sem_feature_query):
    import operator
    normalized_dot_product = 0 
    normalized_dot_product_numerator = 0  #th numerator of the dot
    normalized_dot_product_denominator = 0  #denominator of the dot
    normalized_dot_product_denominator += np.linalg.norm( np.array(corresponding_sem_feature_src) ) * np.linalg.norm( np.array(corresponding_sem_feature_query) )
    for i in no_repeat_class_src:
        for j in no_repeat_class_query:
            # print(i, j)
            # i = list(i)
            # j = list(j)
            if operator.eq(i,j) == True:
                normalized_dot_product_numerator += corresponding_sem_feature_src[no_repeat_class_src.index(i)] * corresponding_sem_feature_query[no_repeat_class_query.index(j)]
    if normalized_dot_product_denominator != 0 :
        normalized_dot_product = float(normalized_dot_product_numerator) / float(normalized_dot_product_denominator)
    else:
        normalized_dot_product = 0
    return normalized_dot_product

def show_vertexs_and_systems(vertexs,origin, z_axis, y_axis, x_axis, origin_2):
    vertexs_src_array = np.array(vertexs)
    points_xyz = vertexs_src_array[:,:3]
    #有些颜色显示不出来，需要颜色归一化
    points_rgb = vertexs_src_array[:,3:6]
    # Create an Open3D point cloud object from the numpy array
    point_cloud_0 = o3d.geometry.PointCloud()
    point_cloud_0.points = o3d.utility.Vector3dVector(points_xyz)
    point_cloud_0.colors = o3d.utility.Vector3dVector(points_rgb)
    vis_src_list = [point_cloud_0]
    
    
    # 创建坐标系的几何体
    # origin_geom = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=origin)
    axis_geom = o3d.geometry.LineSet()

    # 设置顶点和线索引   
    extened_f =10
    axis_geom.points = o3d.utility.Vector3dVector([origin, origin + extened_f * x_axis, origin, origin + extened_f * y_axis, origin, origin + extened_f * z_axis])
    axis_geom.lines = o3d.utility.Vector2iVector([[0, 1], [2, 3], [4, 5]])

    # 设置线的颜色和线宽
    axis_geom.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 设置 x、y、z 轴的颜色
    # axis_geom.line_width = 5.0  # 设置线宽为 5
    # vis_src_list.append (origin_geom) 
    vis_src_list.append(axis_geom)
    
    point_origin = o3d.geometry.PointCloud()
    origin = np.array([origin])
    
    point_origin.points = o3d.utility.Vector3dVector(origin)
    point_origin.paint_uniform_color([1, 0, 0])  # 将点的颜色设置为红色
    point_origin_2 = o3d.geometry.PointCloud()
    origin_2 = np.array([origin_2])
    print(origin, origin_2)
    point_origin_2.points = o3d.utility.Vector3dVector(origin_2)
    point_origin_2.paint_uniform_color([1, 0, 0])  # 将点的颜色设置为红色
    vis_src_list.append(point_origin)
    vis_src_list.append(point_origin_2)
    
    o3d.visualization.draw_geometries(vis_src_list)

#构建顶点的语义和语义距离 局部图特征     
# 输入为顶点, 图  输出：顶点的语义和距离 局部图特征
def extract_local_feature_of_vertex(point_src, vertexs_src):
    point_src_class_list  , dis1_point_src_list  =  most_near_class(point_src, vertexs_src)   # Returns the semantic class and average distance of the nearest vertices
    no_repeat_class_point_src, corresponding_num_point_src, corresponding_ave_distance = count_new(point_src_class_list, dis1_point_src_list)
    return corresponding_ave_distance, no_repeat_class_point_src, corresponding_num_point_src

#判断两个顶点的语义和距离 局部图特征是否相似；低于相似阈值时 不构建顶点对   #
# 输入为顶点的距离和语义特征  输出：True / False
def compare_feature_of_two_vertexs(no_repeat_class_point_src, corresponding_seman_feature_point_src, no_repeat_class_point_query, corresponding_seman_feature_point_query):
    # #local feature of distances: comparison
    # if abs(dis1_point_src - dis2_point_query) > local_dis_feature_threshold_1:
    #     return False
    #local feature of semantic: comparison
    point_similarity = similarity_dot_compare(no_repeat_class_point_src, corresponding_seman_feature_point_src, no_repeat_class_point_query, corresponding_seman_feature_point_query)
    if point_similarity < local_semantic_feature_threshold_:
        return False
    
    return True

#找列表中最大和第二大所有元素的下标
#输出：最大值，最大值下标，第二大值，第二大的下标
def find_max_second_max_indices(lst):
    max_val = 0
    second_max_val = 0
    max_indices = []
    second_max_indices = []

    for i in range(len(lst)):
        if lst[i] > max_val:
            second_max_val = max_val
            second_max_indices = max_indices.copy()
            max_val = lst[i]
            max_indices = [i]
        elif lst[i] == max_val:
            max_indices.append(i)
        elif lst[i] > second_max_val:
            second_max_val = lst[i]
            second_max_indices = [i]
        elif lst[i] == second_max_val:
            second_max_indices.append(i)

    return max_val, max_indices, second_max_val, second_max_indices

#Build a spherical coordinate system, measure the spatial distribution of two spatial semantic graph
def GM_Spherical_plane(vertexs_src, vertexs_query, frame_src, frame_query, normal_src, normal_query, first_remained_flag, common_RGB_perception, use_ground_normal=True):     #V.append(Vextex)   Vextex =  [x_mean, y_mean, z_mean, R, G, B, instance,x_max, y_max, z_max, x_min, y_min, z_min, instance_points, volume, density]
    if first_remained_flag == 1:
        
        # f_GM_information.write("第"+str(frame_query)+"帧query\n")
        global vertex_pairs
        vertex_pairs = []        # vertex_pairs =  [[i1,j1],[i2,j2],...]
        
        #obtain RGB list of source and query，common_RGB_sort
        global RGBlist_src, RGBlist_query, common_RGB_sort
        # print(RGBlist_src)
        i_volumn_difference_min = 0
        j_volumn_difference_min = 0
        
        '''Construct vertex pairs with the same semantics in two vertex graphs'''  
        
        #many-to-many 
        start_time = time.perf_counter()
        most_num_true_vertex_pair = 0  #Maximum number of vertex pairs that may be correct

        # #为每个顶点局部图特征
        # local_sem_feature_lis_query = []  ;  local_sem_feature_class_lis_query = []
        # for vertex in vertexs_query:
        #     local_sem_feature_dis, local_sem_feature_class, local_sem_feature_num = extract_local_feature_of_vertex(vertex, vertexs_query)

        #     local_sem_feature_dis_normalized = np.array(local_sem_feature_dis) / np.sum(local_sem_feature_dis)
        #     local_sem_feature_num_normalized = np.array(local_sem_feature_num) / np.sum(local_sem_feature_num)
        #     local_sem_feature_normalized = w1*local_sem_feature_dis_normalized + w2*local_sem_feature_num_normalized
        #     local_sem_feature_lis_query.append(local_sem_feature_normalized); local_sem_feature_class_lis_query.append(local_sem_feature_class)

        end_time = time.perf_counter()
        # print("432: ",end_time-start_time,"s")
            
        start_time = time.perf_counter()
        #顶点的顺序是语义RGB的顺序
        for common_RGB in common_RGB_sort:        
            index_vertexs_src = [index  for index, RGB_value in enumerate(RGBlist_src) if RGB_value == common_RGB]   
            index_vertexs_query = [index  for index, RGB_value in enumerate(RGBlist_query) if RGB_value == common_RGB]
            # print(index_vertexs_src, index_vertexs_query)
            most_num_true_vertex_pair = most_num_true_vertex_pair + min( len(index_vertexs_src), len(index_vertexs_query) )

            #构建匹配对
            for i in index_vertexs_src:
                for j in index_vertexs_query:
                    #判断两个点的局部特征是否相似------>构建匹配对与否
                    if compare_feature_of_two_vertexs(local_sem_feature_class_lis_src[i],local_sem_feature_lis_src[i],
                                                       local_sem_feature_class_lis_query[j], local_sem_feature_lis_query[j]) == True:
                        vertex_pairs.append([i,j])
        end_time = time.perf_counter()
        # print("446: ",end_time-start_time,"s")
        # if vertex_pairs != []:
        #     print(vertex_pairs)
        # print('most_num_true_vertex_pair: ', most_num_true_vertex_pair)
        '''second order spatial_consistency'''

        # spatial_consistency_sum = [0]*len(vertex_pairs)
        # one_stage_spatial_consistency_matrix = matrix( len(vertex_pairs), len(vertex_pairs), 0)
        global consistent_list_row
        consistent_list_row = []
        consistent_list_col = []   #      corresponding to consistent_list_row，record the consistent element in one_order_spatial_consistency matrix
        two_stage_spatial_consistency_list = [] # record the second order spatial_consistency value corresponding to consistent_list_row， consistent_list_col


        global spatial_consistency_threhold
        # two_stage_spatial_consistency_matrix = matrix( len(vertex_pairs), len(vertex_pairs), 0)
        
        start_time = time.perf_counter()
        #first order spatial_consistency
        for i  in range( len(vertex_pairs) ):                              #pair_1_index_src , pair_1_index_query ;   pair_2_index_src ；  pair_2_index_query 
            for j in range((i+1),len(vertex_pairs)):
                pair_1_index_src = vertex_pairs[i][0]
                pair_1_index_query = vertex_pairs[i][1]
                pair_2_index_src = vertex_pairs[j ][0]
                pair_2_index_query = vertex_pairs[j ][1]      
                if pair_1_index_src == pair_2_index_src or pair_1_index_query == pair_2_index_query:
                    continue
                line_src = np.array(vertexs_src[pair_1_index_src][0:3]) - np.array(vertexs_src[pair_2_index_src][0:3])  
                line_query = np.array(vertexs_query[pair_1_index_query][0:3]) - np.array(vertexs_query[pair_2_index_query][0:3])  
                if abs( np.linalg.norm(line_src) - np.linalg.norm(line_query) ) < spatial_consistency_threhold:
                    # one_stage_spatial_consistency_matrix[i][j] = 1
                    # one_stage_spatial_consistency_matrix[j][i] = 1
                    consistent_list_row.append(i)   
                    consistent_list_col.append(j)
        end_time = time.perf_counter()
        # print("480: ",end_time-start_time,"s")
        
        # print(consistent_list_row, consistent_list_col)
        if len(consistent_list_row) == 0:
            f_GM_information.write("P2 0"+"\n")
            f_GM_information.write("P3 0"+"\n")
            print('consistent_list_row= 0   no one stage consistent pairs!')
            # f_GM_information.write("P2: 0.000"+"\n")
            # f_GM_information.write("P3:   max_same_angle_count is : 0.0000" +  "\n")
            # f_GM_information.write("final score: 0.0000"+", P1: "+str(0)+", P2: "+str(0)+", P3: "+str(0)+"\n")
            # print('consistent_list_row= 0')
            return -1
        print("the number of 1 in one stage matrix: ",len(consistent_list_row))

        #constructing sysmetric matrix
        global consistent_list_row_col
        consistent_list_row_col = consistent_list_row + consistent_list_col
        consistent_list_col_row = consistent_list_col + consistent_list_row
        consistent_list_row_col = np.array(consistent_list_row_col)
        consistent_list_col_row = np.array(consistent_list_col_row)
        start_time = time.perf_counter()
        
        #caluculate two_order_spatial_consistency_matrix values       构建二阶占主要时间
        for index_ in range(len(consistent_list_row)):          #( consistent_list_row[index_], consistent_list_col[index_] ) record the position of element =1  in one_stage_spatial_consistency matrix
            current_row_1 = consistent_list_row[index_]  ;   current_row_1_corresponding_consistent_col = []
            current_row_2 = consistent_list_col[index_] ;  current_row_2_corresponding_consistent_col = []
            
            #新的二阶矩阵生成方法
            current_row_1_corresponding_consistent_col = consistent_list_col_row[consistent_list_row_col == current_row_1]
            current_row_2_corresponding_consistent_col = consistent_list_col_row[consistent_list_row_col == current_row_2]
            # print(current_row_1_corresponding_consistent_col)   #打印也会花时间
            # print(np.intersect1d(current_row_1_corresponding_consistent_col, current_row_2_corresponding_consistent_col).size)
            two_stage_spatial_consistency_list.append(np.intersect1d(current_row_1_corresponding_consistent_col, current_row_2_corresponding_consistent_col).size )
            
            #原先二阶矩阵生成方法
            # for index_1 in [i for i, x in enumerate(consistent_list_row_col) if x == current_row_1]:
            #     current_row_1_corresponding_consistent_col.append(consistent_list_col_row[index_1])
            # for index_2 in [i for i, x in enumerate(consistent_list_row_col) if x == current_row_2]:
            #     current_row_2_corresponding_consistent_col.append(consistent_list_col_row[index_2])

            # tem_two_stage_value = 0
            # for i in current_row_1_corresponding_consistent_col:
            #     if i in current_row_2_corresponding_consistent_col:
            #         tem_two_stage_value +=1
            # two_stage_spatial_consistency_list.append(tem_two_stage_value)
        end_time = time.perf_counter()
        # print("516: ",end_time-start_time,"s")
        
        start_time = time.perf_counter()
        #find the max one in two_order_spatial_consistency_matrix values   方法1
        # 处理一下二阶矩阵，两点如果离得太近，直接二阶矩阵值为0----->即不参与竞选
        
        
        max_in_two_stage_spatial_consistency_list = 0 ; row =[] ; col =[]
        for index_ in range(len(two_stage_spatial_consistency_list)):
            #排除距离过近的两个点
            vertex_for_src_1 = vertexs_src[vertex_pairs[consistent_list_row[index_]][0] ][0:3] ;     vertex_for_src_2 = vertexs_src[vertex_pairs[ consistent_list_col[index_]][0]][0:3]
            vertex_for_src_1 = np.array(vertex_for_src_1);              vertex_for_src_2 = np.array(vertex_for_src_2)
            if np.linalg.norm(vertex_for_src_1 - vertex_for_src_2) < max_two_stage_distance_thres :
                two_stage_spatial_consistency_list[index_] = 0
            
            if max_in_two_stage_spatial_consistency_list < two_stage_spatial_consistency_list[index_]:
                row = []
                col = []
                max_in_two_stage_spatial_consistency_list = two_stage_spatial_consistency_list[index_]
                row.append(consistent_list_row[index_])
                col.append(consistent_list_col[index_])
            elif max_in_two_stage_spatial_consistency_list == two_stage_spatial_consistency_list[index_]:
                row.append(consistent_list_row[index_])
                col.append(consistent_list_col[index_])
        print("vertex pairs number: ", len(vertex_pairs), "maximum value of two stage: ", max_in_two_stage_spatial_consistency_list)        
        
        sum_without_ones = sum([x for x in two_stage_spatial_consistency_list if x != 1])
        print("sum two stage ")
        
        if max_in_two_stage_spatial_consistency_list ==0 or max_in_two_stage_spatial_consistency_list ==1:   
            f_GM_information.write("P2 0"+"\n")
            f_GM_information.write("P3 0"+"\n")
            print("max_in_two_stage_spatial_consistency_list = 0 or 1")
            return -1
        print("numbers of max two stage matrix value: ",len(row))        
        end_time = time.perf_counter()
        # print("535: ",end_time-start_time,"s")
        
        #find all max and second max values in two_order_spatial_consistency_matrix values   
        # max_in_two_stage_spatial_consistency_list, max_indies, second_max_val, second_max_indies  = find_max_second_max_indices(two_stage_spatial_consistency_list)
        # row =[] ; col =[]
        # print("vertex pairs number: ", len(vertex_pairs), "maximum value of two stage: ", max_in_two_stage_spatial_consistency_list, "second max value of two stage: ",second_max_val)
        # if max_in_two_stage_spatial_consistency_list ==0 or max_in_two_stage_spatial_consistency_list ==1 or second_max_val ==0 or second_max_val ==1:   
        #     f_GM_information.write("P2 0"+"\n")
        #     f_GM_information.write("P3 0"+"\n")
        #     print("max_or_second_max_in_two_stage_spatial_consistency_list = 0 or 1")
        #     return -1
        # merged_max_indies_in_two_stage = max_indies + second_max_indies
        #     #去除merged_max_indies_in_two_stage中相距过近的两点------->会使球坐标系偏差大，影响P3的值   filtered_merged_max_indies_in_two_stage
        # flag_lst = []
        # for i in merged_max_indies_in_two_stage:
        #     vertex_for_src_1 = vertexs_src[vertex_pairs[consistent_list_row[i]][0] ][0:3] ;     vertex_for_src_2 = vertexs_src[vertex_pairs[ consistent_list_col[i]][0]][0:3]
        #     vertex_for_src_1 = np.array(vertex_for_src_1);              vertex_for_src_2 = np.array(vertex_for_src_2)
        #     if np.linalg.norm(vertex_for_src_1 - vertex_for_src_2) < 2 :
        #         flag_lst.append(0)
        #     else:
        #         flag_lst.append(1)
        #     print(np.linalg.norm(vertex_for_src_1 - vertex_for_src_2))
        # filtered_merged_max_indies_in_two_stage = [x for x, flag in zip(merged_max_indies_in_two_stage, flag_lst) if flag == 1]
        # for i in filtered_merged_max_indies_in_two_stage:
        #     row.append(consistent_list_row[i])
        #     col.append(consistent_list_col[i])
        # print("numbers of max two stage matrix value: ",len(row))
        
        # print(row,col)
        

        
        
        # most_SC_index =row[0]
        # second_most_SC_index = col[0]
        #search the most likely correct vertex pairs    方法2  计算几个二阶矩阵最大值  对应顶点对的一阶矩阵行和情况
        one_stage_row_sum_of_two_stage_max_lis = []
        for i in range(len(row)):
            row_i_sum = len(consistent_list_col_row[consistent_list_row_col == row[i]])
            col_i_sum = len(consistent_list_col_row[consistent_list_row_col == col[i]])
            one_stage_row_sum_of_two_stage_max_lis.append(row_i_sum+col_i_sum)
        first_max_index = one_stage_row_sum_of_two_stage_max_lis.index(max(one_stage_row_sum_of_two_stage_max_lis))
        most_SC_index = row[first_max_index]
        second_most_SC_index = col[first_max_index]
        
        #search the most likely correct vertex pairs    方法1  计算几个二阶矩阵最大值 对应顶点对的局部图特征相似度 
        # similarity_score = 0
        # max_similarity_score = 0
        # for i in range(len(row)):
        #     point_origin_src = np.array(vertexs_src[vertex_pairs[row[i]][0] ])
        #     point_z_src = np.array(vertexs_src[vertex_pairs[col[i]][0] ])
        #     point_origin_query = np.array(vertexs_query[vertex_pairs[row[i]][1] ])
        #     point_z_query = np.array(vertexs_query[vertex_pairs[col[i]][1] ])

        #     origin_src_class_list  , dis1      =  most_near_class(point_origin_src, vertexs_src)   # Returns the semantic class and average distance of the nearest vertices
        #     z_src_class_list  , dis2               =  most_near_class(point_z_src, vertexs_src, 3)
        #     origin_query_class_list , dis3 =  most_near_class(point_origin_query, vertexs_query, 3)
        #     z_query_class_list ,  dis4          =  most_near_class(point_z_query, vertexs_query, 3)

        #     no_repeat_class_origin_src, corresponding_num_origin_src = count(origin_src_class_list)

        #     # print(origin_src_class_list, no_repeat_class_origin_src, corresponding_num_origin_src)

        #     no_repeat_class_z_src, corresponding_num_z_src = count(z_src_class_list)
        #     no_repeat_class_origin_query, corresponding_num_origin_query = count(origin_query_class_list)
        #     no_repeat_class_z_query, corresponding_num_z_query = count(z_query_class_list)

        #     origin_similarity = similarity_dot_compare(no_repeat_class_origin_src, corresponding_num_origin_src, no_repeat_class_origin_query, corresponding_num_origin_query)
        #     z_similarity = similarity_dot_compare(no_repeat_class_z_src, corresponding_num_z_src, no_repeat_class_z_query, corresponding_num_z_query)
        #     # print(origin_similarity, z_similarity)

        #     #local feature of distances: comparison
        #     if abs(dis1 - dis3) >local_dis_feature_threshold_2 or abs(dis2 - dis4) >local_dis_feature_threshold_2:
        #         continue
        #    #local features of semantics:  cosine similarity
        #     if origin_similarity != 0 and z_similarity != 0:
        #         similarity_score = min(origin_similarity, z_similarity)
        #     if origin_similarity != 0 and z_similarity == 0:
        #         if z_src_class_list ==[] and z_query_class_list == []:
        #             similarity_score = origin_similarity
        #         else:
        #             similarity_score = 0
        #     if origin_similarity == 0 and z_similarity != 0:
        #         if origin_src_class_list ==[] and origin_query_class_list == []:
        #             similarity_score = z_similarity
        #         else:
        #             similarity_score = 0

        #     if max_similarity_score < similarity_score:
        #         max_similarity_score = similarity_score
        #         most_SC_index =row[i]
        #         second_most_SC_index = col[i]
                
        # ###########################---for--- end
        # #这个应该也可以当成最终的评价指标
        # # print(max_similarity_score)
        # if max_similarity_score == 0:
        #     f_GM_information.write("P2 0"+"\n")
        #     f_GM_information.write("P3 0"+"\n")
        #     print("For all candidates, local feature of average distance exceeded!")
        #     return -1
        # # # print(origin_src_class_list, z_src_class_list, origin_query_class_list, z_query_class_list)

        #P2
        great_vertex_pair_by_consistency = 0  #
        # sorted_nums = one_stage_spatial_consistency_matrix[0]
        sorted_nums = 0
        global P2
        P2 = max_in_two_stage_spatial_consistency_list       
        # print("P2: ",P2, "great_vertex_pair_by_consistency: ", great_vertex_pair_by_consistency, "len(sorted_nums): ",len(sorted_nums))
        # f_GM_information.write("P2: "+str(P2)+"\n")
                
        
        if not use_ground_normal:
            # ? 获得原点和z轴
            '''origin, x, y ,z axis'''
            #origin  : most_SC_index，z axis ×  xaxis  = y axis
            vertex_for_original_src = vertexs_src[vertex_pairs[most_SC_index][0] ] ;     vertex_for_original_query = vertexs_query[vertex_pairs[most_SC_index][1]]             #origin
            
            point_original_src = [ vertex_for_original_src[0], vertex_for_original_src[1], vertex_for_original_src[2]]     #src  origin
            point_original_query = [vertex_for_original_query[0], vertex_for_original_query[1], vertex_for_original_query[2] ]  #query  origin

            global vector_z_axis_src, vector_x_axis_src, vector_y_axis_src
            global vector_z_axis_query, vector_x_axis_query, vector_y_axis_query
            #z axis ： second_most_SC_index
            vertex_for_z_axis_src = vertexs_src[vertex_pairs[second_most_SC_index][0] ] ;     vertex_for_z_axis_query = vertexs_query[vertex_pairs[second_most_SC_index][1]]     
            vector_z_axis_src = np.array([vertex_for_z_axis_src[0] - vertex_for_original_src[0] , vertex_for_z_axis_src[1] - vertex_for_original_src[1] , vertex_for_z_axis_src[2] - vertex_for_original_src[2]])        #src z axis
            vector_z_axis_src = vector_z_axis_src / np.linalg.norm(vector_z_axis_src)     #z axis : Unitization
            vector_z_axis_query = [vertex_for_z_axis_query[0] - vertex_for_original_query[0], vertex_for_z_axis_query[1] - vertex_for_original_query[1], vertex_for_z_axis_query[2] - vertex_for_original_query[2]]  #query z axis
            vector_z_axis_query = vector_z_axis_query / np.linalg.norm(vector_z_axis_query)     #z axis : Unitization
            array_z_axis_src_unit = np.array(vector_z_axis_src  )                             #src unit z axis
            array_z_axis_query_unit = np.array(vector_z_axis_query)      #query unit z axis
            
            # ? 构造一个处于y=z平面内的x轴，这个应该怎么取都行，然后叉乘构造y轴
                #x axis : Perpendicular to z-axis
            if array_z_axis_src_unit[0] != 0 :   # ? 这里又涉及了浮点数的比较，应该全部都会落入第一个if
                vector_x_axis_src = [- ( array_z_axis_src_unit[1] + array_z_axis_src_unit[2] ) / array_z_axis_src_unit[0],1,1]
            elif array_z_axis_src_unit[1] != 0 :
                vector_x_axis_src = [1, - ( array_z_axis_src_unit[0] + array_z_axis_src_unit[2] ) / array_z_axis_src_unit[1],1]
            elif array_z_axis_src_unit[2] != 0 :
                vector_x_axis_src = [1,1, - ( array_z_axis_src_unit[0] + array_z_axis_src_unit[1] ) / array_z_axis_src_unit[2]]
            vector_x_axis_src = vector_x_axis_src / np.linalg.norm(vector_x_axis_src)     #x axis : Unitization
            
            array_x_axis_src_unit  = np.array(vector_x_axis_src)                                                #src unit x axis
            
            if array_z_axis_query_unit[0] != 0 :
                vector_x_axis_query = [- ( array_z_axis_query_unit[1] + array_z_axis_query_unit[2] ) / array_z_axis_query_unit[0],1,1]
            elif array_z_axis_query_unit[1] != 0 :
                vector_x_axis_query = [1, - ( array_z_axis_query_unit[0] + array_z_axis_query_unit[2] ) / array_z_axis_query_unit[1],1]
            elif array_z_axis_query_unit[2] != 0 :
                vector_x_axis_query = [1,1, - ( array_z_axis_query_unit[0] + array_z_axis_query_unit[1] ) / array_z_axis_query_unit[2]]
            vector_x_axis_query = vector_x_axis_query / np.linalg.norm(vector_x_axis_query)     #x axis : Unitization
            
            array_x_axis_query_unit  = np.array(vector_x_axis_query )                 #query unit x axis
            
                #y axis   y axis(unit)  = z axis (unit)  ×  x axis (unit)  
            vector_y_axis_src = np.cross(array_z_axis_src_unit , array_x_axis_src_unit)
            vector_y_axis_query = np.cross(array_z_axis_query_unit , array_x_axis_query_unit)
            
            array_y_axis_src_unit = np.array(vector_y_axis_src )                  #src unit y axis
            array_y_axis_query_unit = np.array(vector_y_axis_query)  #query unit y axis

            # print( angle(array_z_axis_src_unit, array_x_axis_src_unit) , angle(array_z_axis_src_unit, array_y_axis_src_unit) ,angle(array_y_axis_src_unit, array_x_axis_src_unit))
            # print( angle(array_z_axis_query_unit, array_x_axis_query_unit) , angle(array_z_axis_query_unit, array_y_axis_query_unit) ,angle(array_y_axis_query_unit, array_x_axis_query_unit))

            # ? 将各顶点转换到各自的球坐标下，方便后边旋转
            #    r^2 = x^2 +y^2 +z^2 ; z = r*cos(theta) ; x=rsinθcosφ ；  y=rsinθsinφ
            spherical_parameters_src = [ ] ;    spherical_parameters_query = [ ]        
            
            array_point_original_src = np.array(point_original_src)              
            array_point_original_query = np.array(point_original_query)  
            
            #calculate theta  fai  r
            for index in range(len(vertex_pairs)):
                # if index == row or index == col :
                #     continue
                vertex_n_src = vertexs_src[vertex_pairs[index][0] ] ;     vertex_n_query = vertexs_query[vertex_pairs[index][1]]                                                                           
                point_n_src = [ vertex_n_src[0], vertex_n_src[1], vertex_n_src[2]]  ;        point_n_query = [ vertex_n_query[0], vertex_n_query[1], vertex_n_query[2]]         
                array_point_n_src =np.array(point_n_src);        array_point_n_query =np.array(point_n_query)           

                # print(vector_x_axis_query, array_point_n_src , array_point_original_src)
                point_n_src_new_x = np.dot(array_x_axis_src_unit, array_point_n_src - array_point_original_src)   ;        point_n_query_new_x = np.dot(array_x_axis_query_unit, array_point_n_query - array_point_original_query)
                point_n_src_new_y = np.dot(array_y_axis_src_unit, array_point_n_src - array_point_original_src)   ;        point_n_query_new_y = np.dot(array_y_axis_query_unit, array_point_n_query - array_point_original_query)
                point_n_src_new_z = np.dot(array_z_axis_src_unit, array_point_n_src - array_point_original_src)   ;        point_n_query_new_z = np.dot(array_z_axis_query_unit, array_point_n_query - array_point_original_query)
                #r
                r_point_n_src = np.linalg.norm(array_point_n_src - array_point_original_src)                                             ;        r_point_n_query = np.linalg.norm(array_point_n_query - array_point_original_query)        
                
                #theta:0-180°，  fai:-180-180°
                if r_point_n_src == 0:
                    theta_point_n_src = 0
                else:
                    theta_point_n_src = np.arccos( min(1, point_n_src_new_z / r_point_n_src) )                                                                     
                if r_point_n_query == 0:
                    theta_point_n_query =0
                else:
                    theta_point_n_query = np.arccos( min(1, point_n_query_new_z / r_point_n_query) )        #theta = arccos( z/r )

                fai_point_n_src = np.arctan2(point_n_src_new_y , point_n_src_new_x)                    ;        fai_point_n_query = np.arctan2(point_n_query_new_y , point_n_query_new_x)        #fai = arctan2( y/x )    
                if r_point_n_src == 0:
                    theta_point_n_src = 0
                    
                if r_point_n_query == 0:
                    theta_point_n_query = 0
                spherical_parameters_src.append([theta_point_n_src/math.pi*180, fai_point_n_src/math.pi*180, r_point_n_src, vertex_pairs[index][0], vertex_pairs[index][1], index])    
                spherical_parameters_query.append([theta_point_n_query/math.pi*180, fai_point_n_query/math.pi*180, r_point_n_query, vertex_pairs[index][0], vertex_pairs[index][1], index])
                # print([theta_point_n_src/math.pi*180, fai_point_n_src/math.pi*180])   ;           print([theta_point_n_query/math.pi*180, fai_point_n_query/math.pi*180])   ;  print("\n")
                
                # f_GM_information.write(str(theta_point_n_src/math.pi*180)+" " + str(fai_point_n_src/math.pi*180)+"   "+str(theta_point_n_query/math.pi*180)+" " + str(fai_point_n_query/math.pi*180)+"\n")   

            # TODO 旋转找最大相似顶点个数，有了法向量后需要改一改
            '''theta and fai'''
            #1. loop ：fai_angle : sample
            max_same_angle_count = 0  
            max_incre_fai_angle = 0    
            
            # vertex_pairs_with_same_sphercial_parameters = []
            
            # global spatial_consistency_threhold
            global theta_threshold
            global fai_threshold_1
            # global fai_threshold_2

            for incre_fai_angle in range(0,360,angle_interval):
                same_angle_count = 0   
                #2.loop：compare theta and fai
                for i in range(len(spherical_parameters_src)):
                    theta_i_src = spherical_parameters_src[i][0]   ;  fai_i_src = spherical_parameters_src[i][1]    ;   r_i_src = spherical_parameters_src[i][2]           
                    theta_i_query = spherical_parameters_query[i][0]   ;  fai_i_query = spherical_parameters_query[i][1] ;r_i_query = spherical_parameters_query[i][2]    
                    fai_i_src = fai_i_src + incre_fai_angle
                    if abs( theta_i_src - theta_i_query ) < theta_threshold and abs(r_i_src - r_i_query) < r_threshold:
                        
                        if fai_i_src > 180:  
                            fai_i_src = fai_i_src -360
                        
                        abs_delta_fai = abs( fai_i_src - fai_i_query )   
                        
                        if abs_delta_fai > 180:     
                            abs_delta_fai = 360 - abs_delta_fai
                        
                        if abs_delta_fai < fai_threshold_1:         
                            same_angle_count = same_angle_count + 1
            
                        # elif abs_delta_fai < fai_threshold_2:        
                        #     same_angle_count = same_angle_count + 0.5
                    
                if same_angle_count > max_same_angle_count :
                    max_same_angle_count = same_angle_count
                    max_incre_fai_angle = incre_fai_angle
                if max_same_angle_count > most_num_true_vertex_pair:   
                    max_same_angle_count = most_num_true_vertex_pair
                    break
                # if float(max_same_angle_count) / most_num_true_vertex_pair > 0.6:
                #     break
        else:
            max_same_angle_count = 0
            vertexs_axis_src = np.array([
                vertexs_src[vertex_pairs[most_SC_index][0]],
                vertexs_src[vertex_pairs[second_most_SC_index][0]]
            ])
            axis_vertex_query = np.array([
                vertexs_query[vertex_pairs[most_SC_index][1]],
                vertexs_query[vertex_pairs[second_most_SC_index][1]]
            ])
            global z_axis_src, y_axis_src, x_axis_src, origin_src, origin_2_src
            z_axis_src = normal_src / np.linalg.norm(normal_src)
            vertex_diff = vertexs_axis_src[1][0:3] - vertexs_axis_src[0][0:3]
            # 计算垂直分量
            projection = np.dot(vertex_diff, z_axis_src) / np.linalg.norm(z_axis_src)
            perpendicular_component = vertex_diff - projection * z_axis_src
            y_axis_src = perpendicular_component / np.linalg.norm(perpendicular_component)
            x_axis_src = np.cross(y_axis_src, z_axis_src)
            origin_src = np.array(vertexs_axis_src[0][0:3]) ;   origin_2_src = np.array(vertexs_axis_src[1][0:3])
            # print(origin_src, origin_2_src, x_axis_src, y_axis_src, z_axis_src)
            # print("原点的index: ", most_SC_index)
            # for i in range(len(consistent_list_row)):
            #     print("consistent_list_row and consistent_list_col： ", consistent_list_row[i], consistent_list_col[i])
            
            global z_axis_query, y_axis_query, x_axis_query, origin_query, origin_2_query
            z_axis_query = normal_query / np.linalg.norm(normal_query)
            vertex_diff = axis_vertex_query[1][0:3] - axis_vertex_query[0][0:3]
            # 计算垂直分量
            projection = np.dot(vertex_diff, z_axis_query) / np.linalg.norm(z_axis_query)
            perpendicular_component = vertex_diff - projection * z_axis_query
            y_axis_query = perpendicular_component / np.linalg.norm(perpendicular_component)
            x_axis_query = np.cross(y_axis_query, z_axis_query)
            origin_query = np.array(axis_vertex_query[0][0:3])   ;  origin_2_query = np.array(axis_vertex_query[1][0:3])
            # print(origin_query, origin_2_query, x_axis_query, y_axis_query, z_axis_query)
            #可视化 顶点 和 球坐标系
            # print(vertexs_src)
            
            # show_vertexs_and_systems(vertexs_src,origin_src, z_axis_src, y_axis_src, x_axis_src, origin_2_src)
            # show_vertexs_and_systems(vertexs_query,origin_query, z_axis_query, y_axis_query, x_axis_query, origin_2_query)
            
            #calculate theta  fai  r
            spherical_parameters_src = [ ] ;    spherical_parameters_query = [ ]     
            for index in range(len(vertex_pairs)):
                # if index == row or index == col :
                #     continue
                vertex_n_src = np.array(vertexs_src[vertex_pairs[index][0] ][0:3]) ;     vertex_n_query = np.array(vertexs_query[vertex_pairs[index][1]][0:3])                                                                              
                # print(vector_x_axis_query, array_point_n_src , array_point_original_src)
                point_n_src_new_x = np.dot(x_axis_src, vertex_n_src - origin_src)   ;        point_n_query_new_x = np.dot(x_axis_query, vertex_n_query - origin_query)
                point_n_src_new_y = np.dot(y_axis_src, vertex_n_src - origin_src)   ;        point_n_query_new_y = np.dot(y_axis_query, vertex_n_query - origin_query)
                point_n_src_new_z = np.dot(z_axis_src, vertex_n_src - origin_src)   ;        point_n_query_new_z = np.dot(z_axis_query, vertex_n_query - origin_query)
                #r
                r_point_n_src = np.linalg.norm(vertex_n_src - origin_src)                                             ;        r_point_n_query = np.linalg.norm(vertex_n_query - origin_query)        
                
                #theta:0-180°，  fai:-180-180°
                if r_point_n_src == 0:
                    theta_point_n_src = 0
                else:
                    theta_point_n_src = np.arccos( min(1, point_n_src_new_z / r_point_n_src) )                                                                     
                if r_point_n_query == 0:
                    theta_point_n_query =0
                else:
                    theta_point_n_query = np.arccos( min(1, point_n_query_new_z / r_point_n_query) )        #theta = arccos( z/r )

                fai_point_n_src = np.arctan2(point_n_src_new_y , point_n_src_new_x)                    ;        fai_point_n_query = np.arctan2(point_n_query_new_y , point_n_query_new_x)        #fai = arctan2( y/x )    
                if r_point_n_src == 0:
                    theta_point_n_src = 0
                    
                if r_point_n_query == 0:
                    theta_point_n_query = 0
                # print([theta_point_n_src/math.pi*180, fai_point_n_src/math.pi*180, r_point_n_src, vertex_pairs[index][0], vertex_pairs[index][1], index], [theta_point_n_query/math.pi*180, fai_point_n_query/math.pi*180, r_point_n_query, vertex_pairs[index][0], vertex_pairs[index][1], index])
                spherical_parameters_src.append([theta_point_n_src/math.pi*180, fai_point_n_src/math.pi*180, r_point_n_src, vertex_pairs[index][0], vertex_pairs[index][1], index])    
                spherical_parameters_query.append([theta_point_n_query/math.pi*180, fai_point_n_query/math.pi*180, r_point_n_query, vertex_pairs[index][0], vertex_pairs[index][1], index])
            
            # print(len(vertex_pairs),len(spherical_parameters_src))
            #比较球坐标
            same_angle_count = 0 
            global similar_spherical_index        #保存有相似球坐标的vertex_pairs下标
            similar_spherical_index = []  
            for i in range(len(spherical_parameters_src)):
                theta_i_src = spherical_parameters_src[i][0]   ;  fai_i_src = spherical_parameters_src[i][1]    ;   r_i_src = spherical_parameters_src[i][2]           
                theta_i_query = spherical_parameters_query[i][0]   ;  fai_i_query = spherical_parameters_query[i][1] ;r_i_query = spherical_parameters_query[i][2]    
                if abs( theta_i_src - theta_i_query ) < theta_threshold and abs(r_i_src - r_i_query) < r_threshold:
                    
                    if fai_i_src > 180:  
                        fai_i_src = fai_i_src -360
                    
                    abs_delta_fai = abs( fai_i_src - fai_i_query )   
                    
                    if abs_delta_fai > 180:     
                        abs_delta_fai = 360 - abs_delta_fai
                    
                    if abs_delta_fai < fai_threshold_1:         
                        same_angle_count = same_angle_count + 1
                        similar_spherical_index.append(i)
        
                    # elif abs_delta_fai < fai_threshold_2:        
                    #     same_angle_count = same_angle_count + 0.5
                    #     similar_spherical_index.append(i)
        # print("P3: max_same_angle_count is : " + str(max_same_angle_count) + ",  most_num_true_vertex_pair is : "+ str(most_num_true_vertex_pair)+" ," + str(float(max_same_angle_count) / most_num_true_vertex_pair) + " ," + str(max_incre_fai_angle) + "angle\n")
        #P3 record
        # f_GM_information.write("P3:   same_angle_count is : " + str(same_angle_count) + "\n")
        global P3
        P3 = same_angle_count
        print("P3: ",P3)
        
        # f_GM_information.write("P3:   max_same_angle_count is : " + str(max_same_angle_count) + ",  most_num_true_vertex_pair is : "+ str(most_num_true_vertex_pair)+", float(max_same_angle_count) / most_num_true_vertex_pair is : " + str(float(max_same_angle_count) / most_num_true_vertex_pair) + ", max_incre_fai_angle is : " + str(max_incre_fai_angle) + "\n")
        #P2的正常写入在P3的正常写入之前
        f_GM_information.write("P2 "+str(P2)+"\n")
        f_GM_information.write("P3 " + str(P3) + "\n")
        # print(vertex_pairs_with_same_sphercial_parameters)
        
        # visualization
        if show_ == 1:
            for index in range(len(vertex_pairs)):
                vertex_n_src = vertexs_src[vertex_pairs[index][0] ] ;     vertex_n_query = vertexs_query[vertex_pairs[index][1]]    
                color_vertex_show_src =  ( float(vertex_n_src[3]/255) ,float(vertex_n_src[4]/255) ,float(vertex_n_src[5]/255) ,1)
                color_vertex_show_query =  ( float(vertex_n_query[3]/255) ,float(vertex_n_query[4]/255) ,float(vertex_n_query[5]/255) ,1)
                if index == most_SC_index:        #origin
                    # print(float(vertex_n_query[3]) ,float(vertex_n_query[4]) ,float(vertex_n_query[5]))
                    ax.scatter(vertex_n_src[0] , vertex_n_src[2], -vertex_n_src[1] , c = color_vertex_show_src , s=500, alpha = 0.5, norm = 0.5)
                    ax_query.scatter(vertex_n_query[0] , vertex_n_query[2], -vertex_n_query[1] , c = color_vertex_show_query , s=500, alpha = 0.5, norm = 0.5)
                    #src
                    line_z_src = [];  line_x_src = [];      line_y_src = []
                    line_z_src.append( [ vertex_n_src[0], vertex_n_src[0]+array_z_axis_src_unit[0] ] );            line_z_src.append([ vertex_n_src[1], vertex_n_src[1]+array_z_axis_src_unit[1]]);   line_z_src.append([ vertex_n_src[2], vertex_n_src[2]+array_z_axis_src_unit[2]])
                    line_x_src.append( [ vertex_n_src[0], vertex_n_src[0]+array_x_axis_src_unit[0] ] );            line_x_src.append([ vertex_n_src[1], vertex_n_src[1]+array_x_axis_src_unit[1]]);   line_x_src.append([ vertex_n_src[2], vertex_n_src[2]+array_x_axis_src_unit[2]])
                    line_y_src.append( [ vertex_n_src[0], vertex_n_src[0]+array_y_axis_src_unit[0] ] );            line_y_src.append([ vertex_n_src[1], vertex_n_src[1]+array_y_axis_src_unit[1]]);   line_y_src.append([ vertex_n_src[2], vertex_n_src[2]+array_y_axis_src_unit[2]])
                    
                    ax.plot(np.array(line_z_src[0]), np.array(line_z_src[2]), -np.array(line_z_src[1]), c='r')
                    ax.plot( np.array(line_x_src[0]), np.array(line_x_src[2]), -np.array(line_x_src[1]), c='g')
                    ax.plot(np.array(line_y_src[0]), np.array(line_y_src[2]), -np.array(line_y_src[1]), c='b')
                    #query
                    line_z_query = [];  line_x_query = [];      line_y_query = []
                    line_z_query.append( [ vertex_n_query[0], vertex_n_query[0]+array_z_axis_query_unit[0] ] );            line_z_query.append([ vertex_n_query[1], vertex_n_query[1]+array_z_axis_query_unit[1]]);   line_z_query.append([ vertex_n_query[2], vertex_n_query[2]+array_z_axis_query_unit[2]])
                    line_x_query.append( [ vertex_n_query[0], vertex_n_query[0]+array_x_axis_query_unit[0] ] );            line_x_query.append([ vertex_n_query[1], vertex_n_query[1]+array_x_axis_query_unit[1]]);   line_x_query.append([ vertex_n_query[2], vertex_n_query[2]+array_x_axis_query_unit[2]])
                    line_y_query.append( [ vertex_n_query[0], vertex_n_query[0]+array_y_axis_query_unit[0] ] );            line_y_query.append([ vertex_n_query[1], vertex_n_query[1]+array_y_axis_query_unit[1]]);   line_y_query.append([ vertex_n_query[2], vertex_n_query[2]+array_y_axis_query_unit[2]])
                    
                    ax_query.plot(np.array(line_z_query[0]), np.array(line_z_query[2]), -np.array(line_z_query[1]), c='r')
                    ax_query.plot(np.array(line_x_query[0]), np.array(line_x_query[2]), -np.array(line_x_query[1]), c='g')
                    ax_query.plot(np.array(line_y_query[0]), np.array(line_y_query[2]), -np.array(line_y_query[1]), c='b')
                # elif index in vertex_pairs_with_same_sphercial_parameters:
                elif index == second_most_SC_index:
                    ax.scatter(vertex_n_src[0] , vertex_n_src[2], -vertex_n_src[1] , c = color_vertex_show_src , s=300, alpha = 0.5, norm = 0.5)
                    ax_query.scatter(vertex_n_query[0] , vertex_n_query[2], -vertex_n_query[1] , c = color_vertex_show_query , s=300, alpha = 0.5, norm = 0.5)
                else:
                    ax.scatter(vertex_n_src[0] , vertex_n_src[2], -vertex_n_src[1] , c = color_vertex_show_src , s=50, alpha = 0.5, norm = 0.5)
                    ax_query.scatter(vertex_n_query[0] , vertex_n_query[2], -vertex_n_query[1] , c = color_vertex_show_query , s=50, alpha = 0.5, norm = 0.5)
    
    if first_remained_flag == 0 :       
    #     P = 0
    #     f_GM_information.write("final score: "+str(P)+"\n")
    # else:
    #     global P3
    #     P3 = max_same_angle_count
    #     f_GM_information.write("final score: 0.0000"+", P1: "+str(P1)+", P2: "+str(P2)+", P3: "+str(P3)+"\n")
        f_GM_information.write("P2 0"+"\n")
        f_GM_information.write("P3 0"+"\n")


def get_vertex_pair(points_frame,frame,point_num_threhold):                

    '''Point clouds are sorted by instance'''
    instance_list = []                                                  #store instance categories
    points_frame.sort(key=takeSeven)            #point cloud sorted by instance
    ins_index_in_pc = []                                          
    a, instance_list = DelRepeat(points_frame,6)                                       
    ins_index_in_pc = SearchIndex(points_frame,instance_list, 6)   #The first subscript of each instance in a frame of instance sorting point cloud
    V= []  #  V=[ [x, y, z, RGB, instance, number_of_points], [], ... ]
    
    vertexs_x_show = []
    vertexs_y_show = []
    vertexs_z_show = []
    color_vertexs_show = []
    
    '''generating vertex for each instance'''
    all_points_num = len(points_frame)
    for i in range(len(ins_index_in_pc)):
        # the first and last subscripts in the frame of each  instance
        if i == len(ins_index_in_pc) - 1:
            instance_start_index = ins_index_in_pc[i]
            instance_end_index = all_points_num
            instance_points = instance_end_index- instance_start_index                   #Number of points for an instance
        else:
            instance_start_index = ins_index_in_pc[i]
            instance_end_index = ins_index_in_pc[i+1]
            instance_points = instance_end_index- instance_start_index                   #Number of points for an instance

        # print(instance_points)
        if instance_points < point_num_threhold:                                                                                           
            continue
        
        (R, G, B), num_RGB = SearchRGB(points_frame, instance_start_index, instance_end_index)     #num_ RGB is the number of points corresponding to the RGB value

        x_mean, x_max, x_min, continue_flag = Mean_Max_Min(points_frame, 0, instance_start_index, instance_end_index) 
        if continue_flag == 1:  #continue_flag = 1 ，Skip generating the vertex, because its points are the origin
            continue
        y_mean, y_max, y_min, continue_flag = Mean_Max_Min(points_frame, 1, instance_start_index, instance_end_index)
        z_mean, z_max, z_min, continue_flag = Mean_Max_Min(points_frame, 2, instance_start_index, instance_end_index)

        Vertex = [x_mean, y_mean, z_mean, R, G, B]

        global dataset_is_scenenet_209
        if dataset_is_scenenet_209 == 1:
            if R == 0 and G == 217 and B == 0 :  #ceiling
                continue
            if R == 0 and G == 139 and B == 249 :  #wall
                continue
            if R == 23 and G == 241 and B == 222 :  #floor
                continue
            if R == 194 and G == 228 and B == 225 :  #wall
                continue

        vertexs_z_show.append(Vertex[2])
        '''visualization'''
        if show_vertex_graph:
            vertexs_x_show.append(Vertex[0])
            vertexs_y_show.append(Vertex[1])
            
            color_vertex_show =  ( float(Vertex[3]/255) ,float(Vertex[4]/255) ,float(Vertex[5]/255) ,1)
            color_vertexs_show.append( color_vertex_show )     
        

        V.append(Vertex)

    # print(color_vertexs_show)
    return  V, vertexs_x_show , vertexs_y_show, vertexs_z_show, color_vertexs_show

def main():
    ###################################
    vertexs_src = []
    point_num_threhold = 10     

    if dataset_is_scenenet_209 == 1:
        bag_name = "/media/tang/yujie2/dataset/train_0/train/scenenet0_209.bag"
        bag = rosbag.Bag(bag_name, 'r')
    ####################################
        try:    
            #src 

            bag_data = bag.read_messages('/camera_point')
            frame = -1
            for topic, msg, t in bag_data:
                frame = frame + 1
                if src_frame == frame:
                    #read point cloud
                    lidar = pcl2.read_points(msg)
                    points_frame = list(lidar)
                    points_src = points_frame
                    print("src: the "+str(frame)+"th frame")
                    vertexs_src, vertexs_src_x_show , vertexs_src_y_show, vertexs_src_z_show, color_src_vertexs_show = get_vertex_pair(points_frame,frame,point_num_threhold)
                    print("vertex:", len(vertexs_src))
                    # print(vertexs_src)
                    break
            #query
            bag_data = bag.read_messages('/camera_point')
            frame = -1
            for topic, msg, t in bag_data:
                frame = frame + 1
                print("the "+str(frame)+"th frame")
                if if_search_specially_query_frame and  frame < query_frame_origin:  
                    continue 
                
                #read point cloud
                lidar = pcl2.read_points(msg)
                points_frame = list(lidar)
                
                V, vertexs_query_x_show , vertexs_query_y_show, vertexs_query_z_show, color_query_vertexs_show = get_vertex_pair(points_frame,frame,point_num_threhold)
                print("vertex:", len(V))

            #----------------------------------------------------inner 'for' loop end--------------------------------------------------------#
                '''visualization'''
                if show_:
                    if show_pointcloud:       
                            points_query = points_frame
                            down_sample_and_show(points_query,5,ax_query)
                            down_sample_and_show(points_src,5,ax)
                    if show_vertex_graph:
                        show_vertexs_graph(vertexs_query_x_show, vertexs_query_y_show,vertexs_query_z_show, color_query_vertexs_show, src_flag =0, query_flag =1)
                        show_vertexs_graph(vertexs_src_x_show , vertexs_src_y_show, vertexs_src_z_show, color_src_vertexs_show, src_flag =1, query_flag =0)
                        
                
                '''grapg matching'''
                if GM_switch == 1 and vertexs_src != [] and frame>=query_frame_origin:
                    f_GM_information.write("the "+str(frame)+" frame query\n")
                    start_time = time.perf_counter()
                    first_remained_flag, common_RGB_perception = GM_first( vertexs_src, V, src_frame, frame)   
                    print(first_remained_flag, common_RGB_perception)
                    if first_remained_flag == 0:
                        print("fail in the first filter part!  OUT!")
                    GM_Spherical_plane( vertexs_src, V, src_frame, frame,1, 1, first_remained_flag, common_RGB_perception, use_ground_normal=False) 
                    end_time = time.perf_counter()
                    time_ =  end_time - start_time
                    if show_:
                        plot_ax_parameter_GM_Spherical(only_show_once, ax, ax_query)      
                    print(time_,"s")
        #---------------------------------------------------------------------------external 'for'  loop end--------------------------------------------------------------------------------------#
        finally:
            bag.close()

    if dataset_is_semantic_kitti == 1:
        curr_path = os.path.dirname(os.path.abspath(__file__))
        sk_preprocess = SKPreprocess(f"{curr_path}/../config/sk_preprocess.yaml")
        global local_sem_feature_lis_src, local_sem_feature_class_lis_src
        global local_sem_feature_lis_query, local_sem_feature_class_lis_query
        
        if debug:
            src_frame = 4486
            _, centers, semantics, src_normal = sk_preprocess.get_clustered_semantic_arary(0, src_frame)
            vertexs_src = np.hstack([centers, semantics]).tolist()
            #为每个点提取局部图特征
            local_sem_feature_lis_src=[]; local_sem_feature_class_lis_src = []
            for vertex in vertexs_src:
                local_sem_feature_dis, local_sem_feature_class, local_sem_feature_num = extract_local_feature_of_vertex(vertex, vertexs_src)
                #归一化 与 加权合并
                local_sem_feature_dis_normalized = np.array(local_sem_feature_dis) / np.sum(local_sem_feature_dis)
                local_sem_feature_num_normalized = np.array(local_sem_feature_num) / np.sum(local_sem_feature_num)
                local_sem_feature_normalized = w1*local_sem_feature_dis_normalized + w2*local_sem_feature_num_normalized
                local_sem_feature_lis_src.append(local_sem_feature_normalized); local_sem_feature_class_lis_src.append(local_sem_feature_class)
                
            query_frame = 1902
            _, centers, semantics, query_normal = sk_preprocess.get_clustered_semantic_arary(0, query_frame)
            vertexs_query = np.hstack([centers, semantics]).tolist()
            #为每个点提取局部图特征
            local_sem_feature_lis_query = []; local_sem_feature_class_lis_query = []
            for vertex in vertexs_query:
                local_sem_feature_dis, local_sem_feature_class, local_sem_feature_num = extract_local_feature_of_vertex(vertex, vertexs_query)
                #归一化 与 加权合并
                local_sem_feature_dis_normalized = np.array(local_sem_feature_dis) / np.sum(local_sem_feature_dis)
                local_sem_feature_num_normalized = np.array(local_sem_feature_num) / np.sum(local_sem_feature_num)
                local_sem_feature_normalized = w1*local_sem_feature_dis_normalized + w2*local_sem_feature_num_normalized
                local_sem_feature_lis_query.append(local_sem_feature_normalized); local_sem_feature_class_lis_query.append(local_sem_feature_class)
                
            first_remained_flag, common_RGB_perception = GM_first( vertexs_src, vertexs_query, src_frame, query_frame)   
            if first_remained_flag == 0:
                print("fail in the first filter part!  OUT!")
            
            GM_Spherical_plane(vertexs_src, vertexs_query,
                                src_frame, query_frame,
                                src_normal, query_normal,
                                first_remained_flag, common_RGB_perception) 

        elif debug == 0:
            #保存所有帧的vertexs，局部图特征, 地面法向量
            frames_vertexs = []; frames_local_sem_feature_lis = []; frame_local_sem_feature_class_lis = []; frames_normal = []
            import glob
            frame_number = len(glob.glob(os.path.join("/media/zhong/JIA/Temp/kitti/05/sequences/05/velodyne/", "*.bin")))
            print(frame_number)
            for frame in range(0,frame_number):
                print(frame)
                _, centers, semantics, normal = sk_preprocess.get_clustered_semantic_arary(0, frame)
                vertexs = np.hstack([centers, semantics]).tolist()
                frames_vertexs.append(vertexs)
                frames_normal.append(normal)
                #为每个点提取局部图特征
                local_sem_feature_lis = []  ;  local_sem_feature_class_lis = []
                for vertex in vertexs:
                    local_sem_feature_dis, local_sem_feature_class, local_sem_feature_num = extract_local_feature_of_vertex(vertex, vertexs)
                    #归一化 与 加权合并
                    local_sem_feature_dis_normalized = np.array(local_sem_feature_dis) / np.sum(local_sem_feature_dis)
                    local_sem_feature_num_normalized = np.array(local_sem_feature_num) / np.sum(local_sem_feature_num)
                    local_sem_feature_normalized = w1*local_sem_feature_dis_normalized + w2*local_sem_feature_num_normalized
                    local_sem_feature_lis.append(local_sem_feature_normalized); local_sem_feature_class_lis.append(local_sem_feature_class)
                frames_local_sem_feature_lis.append(local_sem_feature_lis);  frame_local_sem_feature_class_lis.append(local_sem_feature_class_lis)
            
            #读取benchmark的frame_pair
            file_path = "/home/zhong/SSGM/pairs/pairs_kitti/neg_100/05.txt"  # 替换成你的文件路径
            with open(file_path, 'r') as file:
                lines = file.readlines()
            frame_pairs = []
            for line in lines:
                line = line.strip()  # 去除行尾的换行符
                frame1, frame2, label = line.split()  # 按空格分割每行的三个数
                print(frame1, frame2, label)
                frame1 = int(frame1.lstrip('0')) if frame1 != "000000" else 0  # 去除frame1前面的0，并转换为整数，"000000" 默认为0
                frame2 = int(frame2.lstrip('0')) if frame2 != "000000" else 0  # 去除frame2前面的0，并转换为整数，"000000" 默认为0
                label = int(label)  # 转换为整数
                frame_pairs.append((frame1, frame2, label))  # 将提取的数据作为元组添加到列表中

            
            for index in range(len(frame_pairs)):
                start_time = time.perf_counter()
                src_frame = frame_pairs[index][0]
                query_frame = frame_pairs[index][1]
                
                print(str(src_frame)+" "+str(query_frame)+" "+str(frame_pairs[index][2]))

                vertexs_src = frames_vertexs[src_frame];  V = frames_vertexs[query_frame]
                print("vertex src:", len(vertexs_src))
                print("vertex query:", len(V))
                #读取局部图特征
                local_sem_feature_lis_src = frames_local_sem_feature_lis[src_frame]  ;  local_sem_feature_class_lis_src = frame_local_sem_feature_class_lis[src_frame]
                local_sem_feature_lis_query = frames_local_sem_feature_lis[query_frame]  ;  local_sem_feature_class_lis_query = frame_local_sem_feature_class_lis[query_frame]
                
                src_normal = frames_normal[src_frame] ; query_normal = frames_normal[query_frame] 
                f_GM_information.write(str(src_frame)+" "+str(query_frame)+" "+str(frame_pairs[index][2])+"\n")
                
                first_remained_flag, common_RGB_perception = GM_first( vertexs_src, V, src_frame, query_frame)   
                if first_remained_flag == 0:
                    print("fail in the first filter part!  OUT!")
                
                GM_Spherical_plane(vertexs_src, V,
                                    src_frame, query_frame,
                                    src_normal, query_normal,
                                    first_remained_flag, common_RGB_perception) 
                
                end_time = time.perf_counter()
                time_ =  end_time - start_time
                print("sum time is: ",time_,"s")
                
                import pose_estimation
                # #1: 利用相似球坐标的顶点对 作为两个点云的correspondences，进行两片点云的位姿估计 
                # vertex_indexs_for_pose_estimation = pose_estimation.filter_vp_w_simi_sphe(consistent_list_row_col, similar_spherical_index)
                # if vertex_indexs_for_pose_estimation!=[]:
                #     vertexs_for_pose_estimation = []
                #     for index_value in vertex_indexs_for_pose_estimation:
                #         vertexs_for_pose_estimation.append(np.array([vertexs_src[vertex_pairs[index_value][0]][0:3], V[vertex_pairs[index_value][1]][0:3]]))
                #     pose_estimation.pose_transform(vertexs_for_pose_estimation)
                
                #2: 直接对两个坐标系做对齐，对齐过程中的变换矩阵 就是两片点云配准的变换矩阵
                transformation_matrix = pose_estimation.transform_two_coordinate_system(z_axis_src, y_axis_src, x_axis_src, origin_src, origin_2_src, z_axis_query, y_axis_query, x_axis_query, origin_query, origin_2_query)

if __name__ == "__main__":
    main()

f_GM_information.close()
