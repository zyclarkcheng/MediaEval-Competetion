# Logist regression -> Naive
# Kmeans
# Simple neural net work.
# 
from parsing2vector import read_score_imdb, getDictFrom,dict2vector,get_score_ranking_from,count_frame,get_ranking_in_video
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

read_score_imdb()


def average_distance(vec1,vec2):
    temp_vec1=np.array(vec1)
    temp_vec2=np.array(vec2)
    # print(temp_vec1,type(temp_vec1))
    ave_dis=np.sum(np.abs(temp_vec1-temp_vec2))/len(vec1)
    return ave_dis
train_dict=getDictFrom("data/detect_v0_v77_old_backup.txt");
# test_dict=getDictFrom("data/testFeature_backup.txt");
# Score/Distance....
# print(test_dict['video_107'])

object_detection_class = ['aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']


train_vector=dict2vector(train_dict,object_detection_class)
# test_vector=dict2vector(test_dict,object_detection_class)

train_label=get_score_ranking_from('data/trainLabel.txt')
errorrate_list=[]
absolute_mean_list=[]

for i in range(1,78):
    with_out_movie_name='video_'+str(i)
    print('Using ',with_out_movie_name,' as a test set:')
    with_only_movie_name=with_out_movie_name
    train_frame_number=count_frame(train_vector,with_out_movie=with_out_movie_name)
    test_frame_number=count_frame(train_vector,with_only_movie=with_only_movie_name)
    
    # print(train_frame_number)
    # print(test_frame_number)
    
    X = np.zeros((train_frame_number,len(object_detection_class)))
    X_Test = np.zeros((test_frame_number,len(object_detection_class)))
    y_test = []
    y_test_ranking = []
    frame_name_list = []
    y = []
    i = 0
    j = 0
    for movie in train_vector:
        if movie!=with_out_movie_name:
            for frame in train_vector[movie]:
                X[i,:]=train_vector[movie][frame]
                y.append(train_label[movie][frame]['is_interest'])
                i += 1
        elif movie == with_out_movie_name:
            for frame in train_vector[movie]:
                # print('SHIT,FRAME:\t',frame)
                frame_name_list.append(frame)
                X_Test[j,:]=train_vector[movie][frame]
                y_test.append(train_label[movie][frame]['is_interest'])
                y_test_ranking.append(train_label[movie][frame]['rank'])
                j += 1
    
    
    # print(X.shape)
    
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                         hidden_layer_sizes=(5, 2), random_state=1)
    
    
    
    # X_train, X_Test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
    
    clf.fit(X, y)
    # clf.fit(X_train,y_train)
    
    y_predict= clf.predict(X_Test)
    errorrate=np.sum([y_predict!=y_test])/len(y_test)
    print('errorrate',errorrate)
    errorrate_list.append(errorrate)
    # for i in y_predict:
    #     print(i)
    prbability = clf.predict_proba(X_Test)
    # print(prbability,prbability.shape)
    ranks=get_ranking_in_video(prbability)
    
    # for i in range(0,test_frame_number):
    #     print(frame_name_list[i],ranks[i],y_test_ranking[i])
    
    # plt.figure
    # plt.plot(list(range(0,len(ranks))),ranks)
    
    # plt.plot(list(range(0,len(ranks))),y_test_ranking)
    ranks= list(map(int, ranks))
    y_test_ranking= list(map(int, y_test_ranking))
    ranks=[x for y, x in sorted(zip(y_test_ranking, ranks))]
    y_test_ranking=sorted(y_test_ranking)
    absolute_mean=average_distance(ranks,y_test_ranking)/len(ranks)
    print('average_distance',absolute_mean)
    absolute_mean_list.append(absolute_mean)
    # plt.plot(y_test_ranking,ranks,'o')
    # plt.ylabel('some numbers')
    # plt.show()
    # break

print(errorrate_list)
print('--')
print(absolute_mean_list)


# print(clf.predict(X_Test))





# test_frame_number=count_frame(test_vector)
# Z = np.zeros((test_frame_number,len(object_detection_class)))

# i = 0
# for movie in test_vector:
#     for frame in test_vector[movie]:
#         Z[i,:]=test_vector[movie][frame]

# print(clf.predict(Z))










    