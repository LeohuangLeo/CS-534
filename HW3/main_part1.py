import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import sys
import Decision_tree as dt
import Data_prep as dp

#Part 1 : Decision Tree (DT)

def main():
    df = dp.Read_data('pa3_train.csv')
    df_val = dp.Read_data('pa3_val.csv')
    acc_train = []
    acc_val = []
    # depth from 1 to 8
    for i in range(1,9):
        tree = dt.decisiontree(df.values,i,0,df.shape[1])
        acc_t, ex = dt.calculate_Accuracy(df,tree)
        acc_v, ex = dt.calculate_Accuracy(df_val,tree)
        acc_train.append(acc_t)
        acc_val.append(acc_v)
        print('depth = {}: train accuracy = {}, validation accuracy = {}'.format(i, acc_t, acc_v))
    plt.plot(range(1,9), acc_train)
    plt.plot(range(1,9), acc_val)
    plt.legend(['training', 'validation'])
    plt.xlabel('Depth')
    plt.ylabel('accuracy')
    plt.title('Decision Tree')
    plt.show()

main()