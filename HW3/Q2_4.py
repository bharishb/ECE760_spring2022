import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Q2_4
Recall = np.zeros([5,1])
Precision = np.zeros([5,1])
Accuracy = np.zeros([5,1])
Avg_Accuracy = np.zeros([5,1]);
Avg_Precision = np.zeros([5,1]);
Avg_Recall = np.zeros([5,1]);
do_5_fold_knn(1);
Avg_Accuracy[0] = np.mean(Accuracy);
Avg_Precision[0] = np.mean(Precision);
Avg_Recall[0] = np.mean(Recall);
do_5_fold_knn(3);
Avg_Accuracy[1] = np.mean(Accuracy);
Avg_Precision[1] = np.mean(Precision);
Avg_Recall[1] = np.mean(Recall);
do_5_fold_knn(5);
Avg_Accuracy[2] = np.mean(Accuracy);
Avg_Precision[2] = np.mean(Precision);
Avg_Recall[2] = np.mean(Recall);
do_5_fold_knn(7);
Avg_Accuracy[3] = np.mean(Accuracy);
Avg_Precision[3] = np.mean(Precision);
Avg_Recall[3] = np.mean(Recall);
do_5_fold_knn(10);
Avg_Accuracy[4] = np.mean(Accuracy);
Avg_Precision[4] = np.mean(Precision);
Avg_Recall[4] = np.mean(Recall);
plt.plot(Avg_Accuracy)
plt.plot(Avg_Accuracy)
plt.xlabel('K 5-Fold')
plt.ylabel('Average Accuracy')
plt.title('Average accuracy vs K')