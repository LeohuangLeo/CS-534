import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import Adaboost as ada
import Decision_tree as dt
import Data_prep as dp

# Part 3 : AdaBoost (Boosting)

def main():
    # depth = 1
    df = dp.Read_data("pa3_train.csv")
    df_val = dp.Read_data("pa3_val.csv")
    print('------------number of base classifiers, tree with depth of only 1------------')
    d = 1
    Acc_ada = []
    Acc_ada_v = []
    for l in [1, 2, 5, 10, 15]:
        stumps = ada.adaboost(df, l, d)
        pred = ada.adaboost_predictions(df, stumps)
        acc = ada.calculate_accuracy_ab(pred, df.values[:, -1])
        pred = ada.adaboost_predictions(df_val, stumps)
        acc_v = ada.calculate_accuracy_ab(pred, df_val.values[:, -1])
        Acc_ada.append(acc)
        Acc_ada_v.append(acc_v)
        print('l = {}: train accuracy = {}, validation accuracy = {}'.format(l, acc, acc_v))

    plt.plot([1, 2, 5, 10, 15], Acc_ada)
    plt.plot([1, 2, 5, 10, 15], Acc_ada_v)
    plt.legend(['training', 'validation'])
    plt.xlabel('n classifiers')
    plt.ylabel('accuracy')
    plt.title('number of base classifiers, tree with depth of 1 in adaboost')
    plt.show()
    #depth = 2
    print('------------number of base classifiers, tree with depth of 2------------')
    d = 2
    Acc_ada_ = []
    Acc_ada_v_ = []
    for l in [1, 2, 5, 10, 15]:
        stumps = ada.adaboost(df, l, d)
        pred = ada.adaboost_predictions(df, stumps)
        acc = ada.calculate_accuracy_ab(pred, df.values[:, -1])
        pred = ada.adaboost_predictions(df_val, stumps)
        acc_v = ada.calculate_accuracy_ab(pred, df_val.values[:, -1])
        Acc_ada_.append(acc)
        Acc_ada_v_.append(acc_v)
        print('l = {}: train accuracy = {}, validation accuracy = {}'.format(l, acc, acc_v))

    plt.plot([1, 2, 5, 10, 15], Acc_ada_)
    plt.plot([1, 2, 5, 10, 15], Acc_ada_v_)
    plt.legend(['training', 'validation'])
    plt.xlabel('n classifiers')
    plt.ylabel('accuracy')
    plt.title('number of base classifiers, tree with depth of 2 in adaboost')
    plt.show()
    # comparison between d=2, l=6 and d = 1, l = 15
    print('------------comparison between d=2, l=6 and d = 1, l = 15------------')
    stumps = ada.adaboost(df, 6, 2)
    pred = ada.adaboost_predictions(df, stumps)
    acc = ada.calculate_accuracy_ab(pred, df.values[:, -1])
    pred = ada.adaboost_predictions(df_val, stumps)
    acc_v = ada.calculate_accuracy_ab(pred, df_val.values[:, -1])
    stumps = ada.adaboost(df, 15, 1)
    pred = ada.adaboost_predictions(df, stumps)
    acc_ = ada.calculate_accuracy_ab(pred, df.values[:, -1])
    pred = ada.adaboost_predictions(df_val, stumps)
    acc_v_ = ada.calculate_accuracy_ab(pred, df_val.values[:, -1])
    print('d = 2, l = 6 : train accuracy = {}, validation accuracy = {}'.format(acc, acc_v))
    print('d = 1, l = 15: train accuracy = {}, validation accuracy = {}'.format(acc_, acc_v_))
main()