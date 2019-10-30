import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Gradient_descent as gd
import Read_norm_data as rnd

#Part 1: Explore different learning rate for batch gradient descent
def main():
    X_train, y_train = rnd.read_norm_data('PA1_train.csv') #Read training data
    lr = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7] #Learning rate
    epochs = 10000 #iterations
    #Gradient descent calculation
    #Plotting each learning rate
    #Showing sse of gradient descent during the traning
    final_w, ttl_J = gd.ttl_plt_wo_r2(X_train, y_train, lr, epochs)
    print('-------------------sse in traning data-------------------')
    # Showing the final sse in traning data in every learning rate
    adder = 0
    for best in ttl_J:
        print('--lr:{}--'.format(lr[adder]))
        print('final sse:{}'.format(best[-1]))
        adder+=1

    X_val, y_val = rnd.read_norm_data('PA1_dev.csv') #Read validation data
    print('-------------------sse in validation data-------------------')
    # Showing the final sse in validation data in every learning rate
    count = 0
    for j in final_w:
        print('--lr:{}--'.format(lr[count]))
        se = (X_val.dot(j)- y_val) **2
        print('final sse:{}'.format(se.sum()))
        count+=1

main()

