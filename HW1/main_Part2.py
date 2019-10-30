import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Gradient_descent as gd
import Read_norm_data as rnd

#Part 2: Experiments with different λ values
def main():
    X_train, y_train = rnd.read_norm_data('PA1_train.csv') #Read training data
    lr = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7] #Learning rate
    epochs = 10000 #iterations
    lamda = [1e-3, 1e-2, 1e-1, 1, 10, 100]
    #Gradient descent with regularization calculation
    #Plotting each learning rate and lamda
    #Showing sse of gradient descent during the traning
    final_w, ttl_J = gd.ttl_plt_w_r2(X_train, y_train, lr, epochs, lamda)
    print('-------------------sse in traning data-------------------')
    # Showing the final sse in traning data in every learning rate and lamda
    print(len(ttl_J))
    add = 0
    for i in range(len(lamda)):
        print('When lamda ={}'.format(lamda[i]))
        adder = 0
        for best in ttl_J[add]:
            print('--lr:{}--'.format(lr[adder]))
            print('final sse:{}'.format(best[-1]))
            adder+=1
        add+=1

    X_val, y_val = rnd.read_norm_data('PA1_dev.csv') #Read validation data
    print('-------------------sse in validation data-------------------')
    # Showing the final sse in validation data in every learning rate and lamda
    w_1 = final_w[:8]
    w_2 = final_w[8:16]
    w_3 = final_w[16:24]
    w_4 = final_w[24:32]
    w_5 = final_w[32:40]
    w_6 = final_w[40:]
    ttl_w = [w_1,w_2,w_3,w_4,w_5,w_6]
    count = 0
    count_2 = 0
    for i in ttl_w:
        print('When lamda:{}'.format(lamda[count]))
        count += 1
        count_2 = 0
        for j in i:
            print('--lr:{}--'.format(lr[count_2]))
            se = (X_val.dot(j)- y_val) **2
            print('final sse:{}'.format(se.sum()))
            count_2 += 1

main()