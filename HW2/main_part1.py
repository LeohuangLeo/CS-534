import numpy as np
import Read_data as rd
import Data_pred_op_ap as dp
import Read_test_data as rtd
import matplotlib.pyplot as plt
import csv

# Online perceptron algorithm
def Online_perceptron(x, y, iters):
    w = np.zeros(x.shape[1]).reshape(-1,1)
    w_lis = []
    for i in range(iters):
        for i, X_i in enumerate(x):
            y_i = y[i]
            X_i = X_i.reshape(1,-1)
            if X_i.dot(w)*y_i <= 0:
                w = w + y_i*(X_i.T)
        w_lis.append(w)
    return w_lis

def main():
    X_train, y_train = rd.Read_data('pa2_train.csv')
    X_val, y_val = rd.Read_data('pa2_valid.csv')
    w_list_op = Online_perceptron(X_train, y_train, 15)
    acc_list_train_op = dp.data_pred(X_train, y_train, w_list_op)
    acc_list_val_op = dp.data_pred(X_val, y_val, w_list_op)
    for i in range(1,len(acc_list_train_op)+1):
        print('iterations={}, acc_train={}, acc_val={}'.format(i, acc_list_train_op[i-1], acc_list_val_op[i-1]))

    # Data visualization
    plt.plot(range(1, 16), acc_list_train_op)
    plt.plot(range(1, 16), acc_list_val_op)
    plt.legend(['training', 'validation'])
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.show()

    # Generate the prediction file oplabel.csv with the best iteration = 14
    X_test = rtd.Read_test_data('pa2_test_no_label.csv')
    y_test = list(np.sign(X_test.dot(w_list_op[14])))
    csvFile = open("oplabel.csv", "w")
    writer = csv.writer(csvFile)
    writer.writerows(y_test)
    csvFile.close()

main()