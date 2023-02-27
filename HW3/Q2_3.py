import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Q2_3
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
learning_rate = 0.005;
epochs = 200;
threshold = 0.5
batch_size = 16;
loss = np.zeros([4000*epochs,1])
#SGD gradient is used

def logistic_regression(test_set, training_set, output_set):
    gradient = np.zeros([3000]);
    #f_x_thetha = np.zeros(1);
    thetha=np.zeros([3000]);
    TP=0;
    TN=0;
    FN=0;
    FP=0;
    count=0;
    for iteration in range(epochs):
        for index_train,train_elem in enumerate(training_set):
            f_x_thetha = 1/(1+np.exp(-1*(np.dot(thetha[:3000],train_elem[0:3000]))))
            if(count==batch_size):
                thetha = thetha - learning_rate*gradient/batch_size;
                gradient = train_elem[0:3000]*(f_x_thetha-train_elem[3000]);
                count = 0;
            else:
                gradient = np.add(np.array(gradient),np.array(train_elem[0:3000]*(f_x_thetha-train_elem[3000])));
                count = count + 1;
            #print(train_elem)
            if(f_x_thetha==0):
                log_f_x_thetha1 = 0;
            else:
                log_f_x_thetha1 = np.log(f_x_thetha);
            if(f_x_thetha==1):
                log_f_x_thetha2 = 0;
            else:
                log_f_x_thetha2 = np.log(1-f_x_thetha);                
            loss[iteration*4000+index_train] = -train_elem[3000]*log_f_x_thetha1-(1-train_elem[3000])*log_f_x_thetha2;
                
    for index, test_elem in enumerate(test_set):
        f_x_thetha = 1/(1+np.exp(-1*(np.dot(thetha[:3000],test_elem[0:3000]))))
        output_set[index,3001] = f_x_thetha;
        if(f_x_thetha >= threshold):
            output_set[index,3000] = 1;
        else:
            output_set[index,3000] = 0;
        if((output_set[index, 3000] == 1) and (test_elem[3000]==1)):
            #print(TP);
            TP=TP+1;
        if((output_set[index, 3000] == 0) and (test_elem[3000]==0)):
            TN=TN+1;
        if((output_set[index, 3000] == 0) and (test_elem[3000]==1)):
            FN=FN+1;
        if((output_set[index, 3000] == 1) and (test_elem[3000]==0)):
            FP=FP+1;
    print("TP = ",TP);
    print("TN =",TN);
    print("FP =", FP);
    print("FN =",FN);
    Recall = TP/(TP+FN);
    Precision = (TP)/(TP+FP);
    Accuracy = (TP+TN)/(TP+TN+FP+FN);
    print("Recall = ",Recall)
    print("Precision = ",Precision)
    print("Accuracy = ",Accuracy)
    return [Recall, Precision, Accuracy];


#Normalization
test_set = training_data[:1000,:];
training_set = training_data[1000:5000, :];
training_data_scaled = (training_data -np.mean(training_data,axis=0))/np.std(training_data,axis=0)
#test_set_scaled = (test_set -np.mean(training_set,axis=0))/np.std(training_set,axis=0)
training_data_scaled[:,3000] = training_data[:,3000] # undo label normalization 
#test_set_scaled[:,3000] =0;
#logistic_regression(test_set_scaled, training_set_scaled, output_set);
#training_set_scaled
def do_5_fold_logistic_regression():
    training_data_scaled = (training_data -np.mean(training_data,axis=0))/np.std(training_data,axis=0)
    training_data_scaled[:,3000] = training_data[:,3000] # undo label normalization 
    #1 st fold
    print("1st Fold")
    test_set = training_data_scaled[:1000,:];
    training_set = training_data_scaled[1000:5000, :];
    [Recall[0], Precision[0], Accuracy[0]] = logistic_regression(test_set,training_set, output_set);
    #2 nd fold
    print("2nd Fold")
    test_set = training_data_scaled[1000:2000,:];
    training_set = np.vstack((training_data_scaled[0:1000, :], training_data_scaled[2000:5000, :]));
    [Recall[1], Precision[1], Accuracy[1]] = logistic_regression(test_set,training_set, output_set);
    #3rd  fold
    print("3rd Fold")
    test_set = training_data_scaled[2000:3000,:];
    training_set = np.vstack((training_data_scaled[0:2000, :], training_data_scaled[3000:5000, :]));
    [Recall[2], Precision[2], Accuracy[2]] = logistic_regression(test_set,training_set, output_set);
    #4 st fold
    print("4th Fold")
    test_set = training_data_scaled[3000:4000,:];
    training_set = np.vstack((training_data_scaled[0:3000, :], training_data_scaled[4000:5000, :]));
    [Recall[3], Precision[3], Accuracy[3]] = logistic_regression(test_set,training_set,output_set);
    #5 st fold
    print("5th Fold")
    test_set = training_data_scaled[4000:5000,:];
    training_set = training_data_scaled[:4000,:];
    [Recall[4], Precision[4], Accuracy[4]] = logistic_regression(test_set,training_set, output_set);
do_5_fold_logistic_regression();