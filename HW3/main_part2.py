import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import sys
import Decision_tree as dt
import Data_prep as dp
import Random_forest as rd

# Part 2 : Random Forest (Bagging)

# when q = 0, we train forest in different number of trees
# when q = 1, we train forest in different number of features
# when q = 2, we make the comparison in different Number of seed
def Part2(q=0):
    df = dp.Read_data("pa3_train.csv")
    df_val = dp.Read_data("pa3_val.csv")
    if q == 0:
        m = 5
        d = 2
        n = [1, 2, 5, 10, 25]
        acc_train = []
        acc_val = []
        for n_i in n:
            forest = rd.random_forest_algorithm(df, n_i, m, d)
            predictions = rd.random_forest_predictions(df, forest)
            accuracy = rd.calculate_accuracy_rf(predictions, df.values[:, -1])
            acc_train.append(accuracy)
            predictions_ = rd.random_forest_predictions(df_val, forest)
            accuracy_ = rd.calculate_accuracy_rf(predictions_, df_val.values[:, -1])
            acc_val.append(accuracy_)
            print('n = {}: train accuracy = {}, validation accuracy = {}'.format(n_i, accuracy, accuracy_))

        return acc_train, acc_val

    elif q == 1:
        m = [1, 2, 5, 10, 25, 50]
        d = 2
        n = 15
        acc_train = []
        acc_val = []
        for m_i in m:
            forest = rd.random_forest_algorithm(df, n, m_i, d)
            predictions = rd.random_forest_predictions(df, forest)
            accuracy = rd.calculate_accuracy_rf(predictions, df.values[:, -1])
            acc_train.append(accuracy)
            predictions_ = rd.random_forest_predictions(df_val, forest)
            accuracy_ = rd.calculate_accuracy_rf(predictions_, df_val.values[:, -1])
            acc_val.append(accuracy_)
            print('m = {}: train accuracy = {}, validation accuracy = {}'.format(m_i, accuracy, accuracy_))

        return acc_train, acc_val

    # best result, run 10 trials with different random seeds
    # we set seed number from 0 to 9
    elif q == 2:
        acc = []
        acc_v = []
        for seed in range(10):
            random.seed(seed)
            forest = rd.random_forest_algorithm(df, 15, 50, 2)
            predictions = rd.random_forest_predictions(df, forest)
            accuracy = rd.calculate_accuracy_rf(predictions, df.values[:, -1])
            predictions_ = rd.random_forest_predictions(df_val, forest)
            accuracy_ = rd.calculate_accuracy_rf(predictions_, df_val.values[:, -1])
            acc.append(accuracy)
            acc_v.append(accuracy_)
            print('seed = {}: train accuracy = {}, validation accuracy = {}'.format(seed, accuracy, accuracy_))
        return acc, acc_v
def main():
    print('------------different number of trees in the forest------------')
    acc_train, acc_val = Part2(q=0)
    plt.plot([1, 2, 5, 10, 25], acc_train)
    plt.plot([1, 2, 5, 10, 25], acc_val)
    plt.legend(['training', 'validation'])
    plt.xlabel('n tree')
    plt.ylabel('accuracy')
    plt.title('Different number of trees in random forest')
    plt.show()
    print('------------different number of features for a tree------------')
    acc_train_, acc_val_ = Part2(q=1)
    plt.plot([1, 2, 5, 10, 25, 50], acc_train_)
    plt.plot([1, 2, 5, 10, 25, 50], acc_val_)
    plt.legend(['training', 'validation'])
    plt.xlabel('n features for a tree')
    plt.ylabel('accuracy')
    plt.title('Different number of features for a tree in random forest')
    plt.show()
    print('------------different seed in the performace of the best scenario------------')
    acc_best, acc_best_v = Part2(q=2)
    x = range(10)
    plt.plot(x, acc_best)
    plt.plot(x, acc_best_v)
    plt.legend(['training', 'validation'])
    plt.xlabel('No.seed')
    plt.ylabel('accuracy')
    plt.title('Different seed in the performace of the best scenario in random forest')
    plt.show()



main()