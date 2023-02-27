import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/emails.csv') #Directly loading csv in numpy giving string to int typecast issue
#training_data=np.loadtxt('data/emails.csv', dtype="None")
training_data = np.zeros([5000,3001]);
output_set = np.zeros([1000,3002]);
test_set = np.zeros([1000,3001]);
training_set = np.zeros([4000,3001]);
Recall = np.zeros([5,1])
Precision = np.zeros([5,1])
Accuracy = np.zeros([5,1])
Prob = np.zeros([1000]);

a = np.array(df.iloc[:,1])
for i in range(3001):
    training_data[:,i] = np.array(df.iloc[:,i+1])
np.shape(training_data)
distance = np.zeros(1);
euclid_distance = np.zeros(1);
def knn(test_set, training_set, output_set,k):
    k_neighbours_dist = np.zeros([4000,2]); # col 0: distance , col 1: label
    TP=0;
    TN=0;
    FN=0;
    FP=0;
    for index,test_elem in enumerate(test_set):
        distance = 500000;
        output_set[index, :3001] = test_elem;
        for index_train,train_elem in enumerate(training_set):
            euclid_distance = np.linalg.norm(np.array(train_elem[:3000])-np.array(test_elem[:3000]));
            k_neighbours_dist[index_train,0] = euclid_distance;
            k_neighbours_dist[index_train,1] = train_elem[3000]; # label
           # if(euclid_distance < distance):
            #    distance=euclid_distance;
             #   output_set[index, 3000]  = train_elem[3000];
        #all distances are computed. Now sort it.
        k_neighbours_dist = k_neighbours_dist[k_neighbours_dist[:,0].argsort()] # sorted by distances - index 0
        label_1_count = np.sum(k_neighbours_dist[:k,1]) # count all label 1 count
        label_0_count = k - label_1_count;
        if(label_1_count > label_0_count):
            output_set[index,3001] = label_1_count; 
            output_set[index, 3000]  = 1;
        else:
            output_set[index, 3000]  = 0;

        if((output_set[index, 3000] == 1) and (test_elem[3000]==1)):
            #print(TP);
            TP=TP+1;
        if((output_set[index, 3000] == 0) and (test_elem[3000]==0)):
            TN=TN+1;
        if((output_set[index, 3000] == 0) and (test_elem[3000]==1)):
            FN=FN+1;
        if((output_set[index, 3000] == 1) and (test_elem[3000]==0)):
            FP=FP+1;
    print("TP =", TP );
    print("TN =",TN);
    print("FP =", FP);
    print("FN =", FN);
    Recall = TP/(TP+FN);
    Precision = (TP)/(TP+FP);
    Accuracy = (TP+TN)/(TP+TN+FP+FN);
    print("Recall = ",Recall)
    print("Precision = ",Precision)
    print("Accuracy = ",Accuracy)
    return [Recall, Precision, Accuracy];
    
def do_5_fold_knn(k):
    #1 st fold
    print("1st Fold")
    test_set = training_data[:1000,:];
    training_set = training_data[1000:5000, :];
    [Recall[0], Precision[0], Accuracy[0]] = knn(test_set,training_set, output_set,k);
    print("2nd Fold")
    #2 nd fold
    test_set = training_data[1000:2000,:];
    training_set = np.vstack((training_data[0:1000, :], training_data[2000:5000, :]));
    [Recall[1], Precision[1], Accuracy[1]] = knn(test_set,training_set, output_set,k);
    #3rd  fold
    print("3rd Fold")
    test_set = training_data[2000:3000,:];
    training_set = np.vstack((training_data[0:2000, :], training_data[3000:5000, :]));
    [Recall[2], Precision[2], Accuracy[2]] = knn(test_set,training_set, output_set,k);
    #4 st fold
    print("4th Fold")
    test_set = training_data[3000:4000,:];
    training_set = np.vstack((training_data[0:3000, :], training_data[4000:5000, :]));
    [Recall[3], Precision[3], Accuracy[3]] = knn(test_set,training_set,output_set,k);
    #5 st fold
    print("5th Fold")
    test_set = training_data[4000:5000,:];
    training_set = training_data[:4000,:];
    [Recall[4], Precision[4], Accuracy[4]] = knn(test_set,training_set, output_set,k);
    
    
#Q2_2
do_5_fold_knn(1)