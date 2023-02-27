import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

 #Q2_5
    #1 st fold -KNN
test_set = training_data[:1000,:];
training_set = training_data[1000:5000, :];
[Recall[0], Precision[0], Accuracy[0]] = knn(test_set,training_set, output_set,5);
roc_data = np.vstack((test_set[:,3000].T, output_set[:,3000].T, output_set[:,3001].T)).T
roc_sorted1 = roc_data[roc_data[:,1].argsort()]
roc_indx_start = np.argmax(roc_sorted1[:,1])
roc_sorted2 = roc_sorted1[roc_indx_start:]
roc_sorted3 = roc_sorted2[(-roc_sorted2[:,2]).argsort()]
num_positives = np.shape(roc_sorted3)[0]
TP = np.sum(roc_sorted3[:,0]) 
FP = num_positives - TP;
TPRk = np.zeros(num_positives);
FPRk = np.zeros(num_positives);
TPR_count =0;
FPR_count =0;
for i in range(num_positives):
    if(roc_sorted3[i,0]==0):
        TPRk[i]=TPR_count/TP;
        FPRk[i]=FPR_count/FP;
        FPR_count = FPR_count +1;
    else:
        TPRk[i] = TPRk[i-1];
        FPRk[i] = FPRk[i-1];
        TPR_count = TPR_count + 1;
 #1 st fold- Logistic REgression
test_set = training_data[:1000,:];
training_set = training_data[1000:5000, :];
[Recall[0], Precision[0], Accuracy[0]] = logistic_regression(test_set,training_set, output_set);
roc_data = np.vstack((test_set[:,3000].T, output_set[:,3000].T, output_set[:,3001].T)).T
roc_sorted1 = roc_data[roc_data[:,1].argsort()]
roc_indx_start = np.argmax(roc_sorted1[:,1])
roc_sorted2 = roc_sorted1[roc_indx_start:]
roc_sorted3 = roc_sorted2[(-roc_sorted2[:,2]).argsort()]
num_positives = np.shape(roc_sorted3)[0]
print(num_positives)
TP = np.sum(roc_sorted3[:,0]) 
FP = num_positives - TP;
TPRl = np.zeros(num_positives);
FPRl = np.zeros(num_positives);
TPR_count =0;
FPR_count =0;
for i in range(num_positives):
    if(roc_sorted3[i,0]==0):
        TPRl[i]=TPR_count/TP;
        FPRl[i]=FPR_count/FP;
        FPR_count = FPR_count +1;
    else:
        TPRl[i] = TPRl[i-1];
        FPRl[i] = FPRl[i-1];
        TPR_count = TPR_count + 1;
plt.plot(FPRk,TPRk)
plt.plot(FPRl,TPRl)
#np.shape(roc_sorted3)
plt.xlabel('False Positive Rate(FPR)')
plt.ylabel('True Positive Rate(TPR)')
plt.title('ROC curve')
plt.legend(['KNN','Logistic Regression'])