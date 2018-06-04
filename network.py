# Logist regression -> Naive
# Kmeans
# Simple neural net work.
# 
from parsing2vector import getDictFrom,dict2vector,get_score_ranking_from,count_frame,get_ranking_in_video,count_frame_in_one_movie
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


def change_rank_to_label(rank,frame_number_in_video):
    # print(rank,type(rank))
    # print(frame_number_in_video,type(frame_number_in_video))
    rank=int(rank)
    print(rank,frame_number_in_video)
    if rank*1.0/frame_number_in_video>=0 and rank*1.0/frame_number_in_video<0.20:
        return 5
    elif rank*1.0/frame_number_in_video>=0.20 and rank*1.0/frame_number_in_video<0.40:
        return 4
    elif rank*1.0/frame_number_in_video>=0.40 and rank*1.0/frame_number_in_video<0.60:
        return 3
    elif rank*1.0/frame_number_in_video>=0.60 and rank*1.0/frame_number_in_video<0.80:
        return 2
    else:
        return 1

def change_rank_to_label3(rank,frame_number_in_video):
    # print(rank,type(rank))
    # print(frame_number_in_video,type(frame_number_in_video))
    rank=int(rank)
    print(rank,frame_number_in_video)
    if rank*1.0/frame_number_in_video>=0 and rank*1.0/frame_number_in_video<1.0/3:
        return 3
    elif rank*1.0/frame_number_in_video>=1.0/3 and rank*1.0/frame_number_in_video<2.0/3:
        return 2
    else:
        return 1


    # return 


train_dict=getDictFrom("data/detect_v0_v77.txt");
test_dict=getDictFrom("data/testFeature.txt");
# Score/Distance....
# print(test_dict['video_107'])

object_detection_class = ['aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']


train_vector=dict2vector(train_dict,object_detection_class)
test_vector=dict2vector(test_dict,object_detection_class)

train_label=get_score_ranking_from('data/trainLabel.txt')

with_out_movie_name='video_74'
with_only_movie_name=with_out_movie_name
train_frame_number=count_frame(train_vector,with_out_movie=with_out_movie_name)
test_frame_number=count_frame(train_vector,with_only_movie=with_only_movie_name)

print(train_frame_number)
print(test_frame_number)

X = np.zeros((train_frame_number,len(object_detection_class)))
X_Test = np.zeros((test_frame_number,len(object_detection_class)))
y_test = []
y_test_ranking = []
# y_test_score = []
frame_name_list = []
y = []
i = 0
j = 0
for movie in train_vector:
    frame_num_in_movie=count_frame_in_one_movie(train_vector,movie_name=movie)
    if movie!=with_out_movie_name:
        for frame in train_vector[movie]:
            X[i,:]=train_vector[movie][frame]
            temp_label=change_rank_to_label3(train_label[movie][frame]['rank'],frame_num_in_movie)
            y.append(temp_label)
            i += 1
    elif movie == with_out_movie_name:
        for frame in train_vector[movie]:
            # print('SHIT,FRAME:\t',frame)
            frame_name_list.append(frame)
            X_Test[j,:]=train_vector[movie][frame]
            temp_label=change_rank_to_label3(train_label[movie][frame]['rank'],frame_num_in_movie)
            y_test.append(temp_label)
            y_test_ranking.append(train_label[movie][frame]['rank'])
            j += 1


print(X.shape)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5, 2), random_state=1)



# X_train, X_Test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

clf.fit(X, y)
# clf.fit(X_train,y_train)

y_predict= clf.predict(X_Test)

print('Error rate',np.sum([y_predict!=y_test])/len(y_test))
for i in range(0,len(y_test)):
    print(y_test[i],y_predict[i])
prbability = clf.predict_proba(X_Test)
print(prbability,prbability.shape)
ranks=get_ranking_in_video(prbability)

for i in range(0,test_frame_number):
    print(frame_name_list[i],ranks[i],y_test_ranking[i])

# plt.figure
# plt.plot(list(range(0,len(ranks))),ranks)

# plt.plot(list(range(0,len(ranks))),y_test_ranking)

ranks= list(map(int, ranks))
y_test_ranking= list(map(int, y_test_ranking))
ranks=[x for y, x in sorted(zip(y_test_ranking, ranks))]
y_test_ranking=sorted(y_test_ranking)
plt.plot(y_test_ranking,ranks,'o')
plt.ylabel('some numbers')
plt.show()


# print(clf.predict(X_Test))





# test_frame_number=count_frame(test_vector)
# Z = np.zeros((test_frame_number,len(object_detection_class)))

# i = 0
# for movie in test_vector:
#     for frame in test_vector[movie]:
#         Z[i,:]=test_vector[movie][frame]

# print(clf.predict(Z))










    