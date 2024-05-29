# -*- coding:UTF-8 -*-
import re

# precision_GM.append(1) ; recall_GM.append(0)                  

pos_neg = []

def subStrIndex(substr,str):
    result = []
    index = 0
    while str.find(substr,index,len(str)) != -1:
        temIndex = str.find(substr,index,len(str))
        result.append(temIndex)
        index = temIndex + 1
    return result

def get_score_from_file_GM_new():   
    scores = []
    scores_num = []
    filename = "../data/log/07_neg100_2024-05-28-May-05-1716892205ours.txt"
    # filename = "../data/log/rthetafai=0.5 2 2/07/07-neg-100.txt"

    less_20_list = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        i = 0
        min_pos = 1000
        position_1 = []
        numbers = []
        max_neg = 0
        while i < len(lines):
            if i + 3 < len(lines):
                
                
                # print(lines[i])
                p1 = float(lines[i + 1].strip().split()[1])
                p2 = float(lines[i + 2].strip().split()[1])
                p3 = float(lines[i + 3].strip().split()[1])
                number_1 = int(lines[i].strip().split()[0])
                number_2 = int(lines[i].strip().split()[1])
                #src_vertexs_num = len(frames_vertexs[number_1])
                #query_vertexs_num = len(frames_vertexs[number_2])
                # if src_vertexs_num <20 and query_vertexs_num < 20:
                #     less_20_list.append(number_1)
                #     i += 4
                #     continue
                    # print("<20: ",i/4)
                
                flag = int(lines[i].strip().split()[2])
                pos_neg.append(flag)
                
            if p1 < 0.5 or (p2 == 0 and p3 == 0):  
                tem_final_score = 0
                scores_num.append(0)
            else:            
                tem_final_score = p1  + p3  +p2
                number_1 = int(lines[i].strip().split()[0])
                number_2 = int(lines[i].strip().split()[1])
                #src_vertexs_num = len(frames_vertexs[number_1])
                #query_vertexs_num = len(frames_vertexs[number_2])
            
                scores_num.append(tem_final_score)
                
            if flag and  tem_final_score == 0:
                pos_number_1 = int(lines[i].strip().split()[0])
                pos_number_2 = int(lines[i].strip().split()[1])
                position_1.append(len(scores_num)) 
                numbers.append((pos_number_1,pos_number_2))
            if flag==0 and max_neg < tem_final_score:
                
                
                max_neg = tem_final_score
                position_2 = len(scores_num)
                neg_number_1 = int(lines[i].strip().split()[0])
                neg_number_2 = int(lines[i].strip().split()[1])
                #print(neg_number_1, neg_number_2, p1, p2 ,p3, len(frames_vertexs[neg_number_1]), len(frames_vertexs[neg_number_2]))
                
            i += 4
    # print("min(p1: ", min(p1_list))
    # import matplotlib.pyplot as plt
    # plt.plot([i for i in range(1,len(p1_list)+1)],p1_list)
    # plt.show()
    print("less_20_list", len(less_20_list))
    print(position_1)
    print(numbers)
    #print(max_neg, position_2 ,neg_number_1,neg_number_2, len(frames_vertexs[neg_number_1]), len(frames_vertexs[neg_number_2]))
    return scores_num


def R_P_calculate(frames_num, scores, score_threshold, true_positive, flag = "1"):
    FP_list = [];  TP_list = []
    TP=0 ; FP =0 ; TN=0 ; FN=0
    for i in range(frames_num):
        if scores[i] ==-1:  #不参与比较
            continue
        elif  scores[i] >=score_threshold:   #阳性P出现 1
            if true_positive[i] == 1:
                TP = TP+ 1
                TP_list.append(i)
            elif true_positive[i] == 0:
                FP = FP + 1
                FP_list.append(i)
        elif scores[i] < score_threshold:  #阴性N出现 0
            if true_positive[i] == 1:
                FN = FN+ 1
            elif true_positive[i] == 0:
                TN = TN + 1
    if TP +FN == 0:
        recall_once = 0
    else:
        recall_once = float(TP) / (TP +FN) 
    
    if TP +FP == 0:
        precision_once = 0
    else:
        precision_once =  float(TP) / (TP +FP) 
    
        
    return recall_once, precision_once

def R_P_area_calculate(precision_list, recall_list): 
    area = 0
    for i in range(len(recall_list)-1):
        area += float(precision_list[i] + precision_list[i+1]) * (recall_list[i+1] - recall_list[i]) /2
    print( area)
    return area
    
    
def calculate_F1_score_max(recall_list, precision_list):
    F1_score_Max = 0
    for i in range(len(recall_list)):
        if F1_score_Max < (2*float(recall_list[i])*precision_list[i]/(recall_list[i] + precision_list[i])):
            F1_score_Max = 2*float(recall_list[i])*precision_list[i]/(recall_list[i] + precision_list[i])
    print(F1_score_Max)
    return F1_score_Max

def calculate_EP(recall_list, precision_list):
    EP = 0
    
    for i in range(len(recall_list)):
        if precision_list[i] == 1:
            EP = float(precision_list[0] + recall_list[i])/2
    print(EP)
    return EP

def print_R1(recall_list, precision_list):
    R1 = 0
    
    for i in range(len(recall_list)):
        if precision_list[i] == 1:
            R1 = recall_list[i]
    print(R1)
    return R1


# score_GM = get_score_from_file_GM()
score_GM = get_score_from_file_GM_new()
# score_GM = get_score_from_file_GM_new_ours_1_29_self_adjust_new()

frames_num = len(score_GM)


recall_GM = [] ; precision_GM = []  
precision_GM.append(1) ; recall_GM.append(0)                
# for score_threshold in range(2000, -1, -1):
for score_threshold in range(500, -1, -1):
    # score_threshold = float(score_threshold)/100
    # score_threshold = float(score_threshold)/10
    
    recall_GM_once, precision_GM_once = R_P_calculate(frames_num, score_GM, score_threshold, pos_neg, 'ours')
    # print(recall_GM_once, precision_GM_once)
    if recall_GM_once != 0 or precision_GM_once != 0:
        recall_GM.append( recall_GM_once )
        precision_GM.append( precision_GM_once )
# print(recall_GM, precision_GM)


# print(recall_X_view_3, precision_X_view_3)
print("AP SSGM:")
R_P_area_calculate(precision_GM,recall_GM)


print("F1 max:       ")
calculate_F1_score_max(recall_GM, precision_GM)

print("EP:       ")
calculate_EP(recall_GM, precision_GM)

print("R1:      ")
print_R1(recall_GM, precision_GM)

import numpy as np
import matplotlib.pyplot as plt


line_w = 4
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=0.2, right=0.9, top=0.8, bottom=0.2)
plt.plot(recall_GM, precision_GM, color='tomato', label='SeGraM', lw=line_w)
plt.xlim([0, 1.05])  # 
plt.ylim([0, 1.05])
plt.rcParams.update({'font.size': 20})
plt.xlabel('Recall', fontdict={'weight': 'normal', 'size': 20})
plt.ylabel('Precision', fontdict={'weight': 'normal', 'size': 20})  # 
# plt.title("scannet02_00-with-photo-deletion")
plt.legend()
plt.grid()
plt.savefig('real-world.png',dpi=300)
plt.show()

