# Logist regression -> Naive
# Kmeans
# Simple neural net work.
# 
from collections import defaultdict
import pprint
import numpy as np


def read_score_imdb():
    with open('data/imdb.txt') as file:
        my_list = file.readlines()
        print(my_list)



def rec_dd():
    return defaultdict(rec_dd)

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

def get_ranking_in_video(probability_array):
    x_list=[]
    for sublist in probability_array:
        x_list.append(sublist[0])

    temp = argsort(x_list)
    ranks = np.empty(len(x_list), int)
    ranks[temp] = np.arange(len(x_list)) + 1
    # for i in range(0,len(x_list)):
    #     print(ranks[i],x_list[i])
    # print(ranks)
    # ranking_list
    return ranks

def count_frame_in_one_movie(one_vector, movie_name=''):
    count_frame=0

    for movie in one_vector:
        if movie_name!='' and movie==movie_name:
            for frame in one_vector[movie]:
                count_frame+=1

    return count_frame


def count_frame(one_vector, with_out_movie='', with_only_movie=''):
    count_frame=0

    for movie in one_vector:
        if with_only_movie!='' and with_out_movie!='':
            break
        elif with_out_movie!= '' and movie != with_out_movie:
            for frame in one_vector[movie]:
                count_frame+=1
        elif with_only_movie!= ''and movie ==with_only_movie:
            for frame in one_vector[movie]:
                count_frame+=1

    return count_frame

def get_score_ranking_from(path_string):
    result_dict = rec_dd()
    
    with open(path_string) as file:
        my_list = file.readlines()
        
        for each_line in my_list:
            # print(len(each_line))
            if each_line[len(each_line)-1].isspace():
                # print(each_line[:len(each_line)-1])
                word_list = each_line[:len(each_line)-1].split(',')
            else:
                # print(each_line)
                word_list = each_line.split(',')
            for i in range(0,len(word_list)):
                label_name=['is_interest','score','rank']
                if i==0:
                    temp_movie=word_list[i]
                elif i==1:
                    temp_frame=word_list[i]
                else:
                    result_dict[temp_movie][temp_frame][label_name[i-2]]=word_list[i]

    return result_dict


def getDictFrom(path_string):
    object_detection_class = ['aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']


    result_dict = rec_dd()
    
    with open(path_string) as file:
        my_list = file.readlines()
        for each_line in my_list:
            each_line = each_line.replace("\n", "")
            word_list = each_line.split(',')
            # print len(word_list)
            # for word in word_list:
            #     if len(word)>0 and '\n' not in word and '\r' not in word and ' ' not in word:
            #         print word
            print('word_list:',word_list)
            for i in range(0,len(word_list)):
                if i == 0 and word_list[i].startswith('video'):
                    temp_video_name = word_list[i]
                    # result_dict[temp_video_name]={}
                elif i == 1 and word_list[i].endswith('.jpg'):
                    temp_frame_name = word_list[i]
                elif 'nothing' in word_list[i]:
                    result_dict[temp_video_name][temp_frame_name] = 'nothing'
                elif word_list[i] in object_detection_class:
                    # result_dict[temp_video_name][temp_frame_name]={}
                    temp_class_name = word_list[i]
                    i += 1;
                    try:
                        temp_class_number = int(word_list[i])
                        i += 1;
                    except:
                        print('Error reading class number')
                    temp_class_list = []
                    for j in range(0,temp_class_number):
                        try:
                            temp_class_list.append(float(word_list[i+j]))
                        except:
                            print('Error reading class probability')
    
                    result_dict[temp_video_name][temp_frame_name][temp_class_name]=temp_class_list
                    # Next line
                    # print(result_dict[temp_video_name][temp_frame_name])
                    break;
    
    
    # pprint.pprint(result_dict)
    # print(result_dict)
    return result_dict
# print((result_dict))

def dict2vector(source_dict,object_detection_class):
    result_dict = rec_dd()
    for movie in source_dict:
        for frame in source_dict[movie]:
            result_dict[movie][frame]=np.zeros((len(object_detection_class)))
            # print(result_dict[movie][frame])
            if source_dict[movie][frame]!= 'nothing':
                # print(source_dict[movie][frame])
                for i in range(0,len(object_detection_class)):
                    class_name=object_detection_class[i]
                    # print('SHIT',source_dict[movie][frame][class_name],type(source_dict[movie][frame][class_name]))
                    if( len(source_dict[movie][frame][class_name])>0):
                        # print('FUCK',movie,frame,i,class_name,source_dict[movie][frame][class_name])
                        # List
                        # [1,0.999,0.98]
                        temp_list = source_dict[movie][frame][class_name]
                        # 
                        result_dict[movie][frame][i]+=np.sum(temp_list)

            # print('FUCKHERE',movie,frame,result_dict[movie][frame])
        # break

    return result_dict



    