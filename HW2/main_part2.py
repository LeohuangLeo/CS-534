import numpy as np
import Read_data as rd
import Data_pred_op_ap as dp
import Read_test_data as rtd
import matplotlib.pyplot as plt
import csv

# Average Perceptron algorithm
def Average_perceptron(X, y, iters):
    w = np.random.rand(len(X[0]), 1)
    w_ave = np.random.rand(len(X[0]), 1)
    ttl_w = []
    j = 0
    s = 1
    while j < iters:
        for i, X_i in enumerate(X):
            X_i = X_i.reshape(1,-1)
            if X_i.dot(w)*y[i] <= 0:
                w = w + y[i]*(X_i.T)
            w_ave = (s*w_ave + w) / (s+1)
            s += 1
        ttl_w.append(w_ave)
        j += 1
    return ttl_w

def main():
    X_train, y_train = rd.Read_data('pa2_train.csv')
    X_val, y_val = rd.Read_data('pa2_valid.csv')
    w_list_ap = Average_perceptron(X_train, y_train, 15)
    acc_list_train_ap = dp.data_pred(X_train, y_train, w_list_ap)
    acc_list_val_ap = dp.data_pred(X_val, y_val, w_list_ap)
    for i in range(1, len(acc_list_train_ap) + 1):
        print('iterations={}, acc_train={}, acc_val={}'.format(i, acc_list_train_ap[i - 1], acc_list_val_ap[i - 1]))
    # Data visualization
    plt.plot(range(1, 16), acc_list_train_ap)
    plt.plot(range(1, 16), acc_list_val_ap)
    plt.legend(['training', 'validation'])
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.show()
    # Generate the prediction file oplabel.csv with the best iteration = 14
    X_test = rtd.Read_test_data('pa2_test_no_label.csv')
    y_test = list(np.sign(X_test.dot(w_list_ap[14])))
    csvFile = open("aplabel.csv", "w")
    writer = csv.writer(csvFile)
    writer.writerows(y_test)
    csvFile.close()

main()