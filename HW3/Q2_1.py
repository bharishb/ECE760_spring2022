import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

training_data=np.loadtxt("data/D2z.txt", delimiter=' ', dtype=None)
np.shape(training_data)
x1_test = np.linspace(-2, 2, 41);
x2_test = np.linspace(-2, 2, 41);
pred_data = np.zeros([41*41,3]);
dist = 500000;
pred_label = -1;
t_dist = np.zeros(1);
a = np.zeros([1,2]);
b = np.zeros([1,2]);
for i in range(41):
    for j in range(41):
        dist = 500000;
        pred_data[41*i+j, 0:2] = [x1_test[i], x2_test[j]];
        for index,t_data in enumerate(training_data):
            a = [pred_data[41*i+j, 0],   pred_data[41*i+j, 1]]
            b = [t_data[0], t_data[1]]
            t_dist = np.linalg.norm(np.array(a)-np.array(b));
            if(t_dist < dist):
                dist = t_dist
                pred_label = t_data[2];
        pred_data[41*(i)+j, 2]= pred_label;
plt.scatter(pred_data[:,0], pred_data[:,1])
data_sorted_y = pred_data[pred_data[:,2].argsort()]
y1_indx_start = np.argmax(data_sorted_y[:,2])
plt.scatter(data_sorted_y[:y1_indx_start,0],data_sorted_y[:y1_indx_start,1])
plt.scatter(data_sorted_y[y1_indx_start:,0],data_sorted_y[y1_indx_start:,1])
data_sorted_y = training_data[training_data[:,2].argsort()]
y1_indx_start = np.argmax(data_sorted_y[:,2])
plt.scatter(data_sorted_y[:y1_indx_start,0],data_sorted_y[:y1_indx_start,1])
plt.scatter(data_sorted_y[y1_indx_start:,0],data_sorted_y[y1_indx_start:,1])
plt.xlabel('Fetaure x1')
plt.ylabel('Feature x2')
plt.title('Decision Boundary Plot \n Green, Saffron : Grid test data \n violet, red : Training data')