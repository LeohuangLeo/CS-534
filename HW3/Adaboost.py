import numpy as np
import pandas as pd
import math
import random
import sys
import Decision_tree as dt
import Random_forest as rf

# check the purity of data. If it return True, the tree doen't have to go more deep
def check_purity_ada(data):
    label_column = data[:, -2]
    unique_classes = np.unique(label_column)
    if len(unique_classes) == 1:
        return True
    else:
        return False

# calculate alpha
def cal_alpha(er_index, D):
    D = D / D.sum()
    epison = np.sum(D[er_index])
    alpha = 0.5 * np.log((1-epison)/epison)
    return alpha

# gini index in adaboost
def cal_weighted_gini(rootdata, ldata, rdata):
    root_label_column = rootdata[:, -2]
    root_weight = rootdata[:, -1]
    left_label_column = ldata[:, -2]
    left_weight = ldata[:, -1]
    right_label_column = rdata[:, -2]
    right_weight = rdata[:, -1]

    zero_cnt = 0
    one_cnt = 0
    for i in range(len(root_label_column)):
        if root_label_column[i] == 0:
            zero_cnt += root_weight[i]
        else:
            one_cnt += root_weight[i]
    _, root_counts = np.unique(root_label_column, return_counts=True)
    root_counts = [zero_cnt, one_cnt]
    root_counts = np.asarray(root_counts)

    zero_cnt = 0
    one_cnt = 0
    for i in range(len(left_label_column)):
        if left_label_column[i] == 0:
            zero_cnt += left_weight[i]
        else:
            one_cnt += left_weight[i]
    l_counts = [zero_cnt, one_cnt]
    l_counts = np.asarray(l_counts)

    zero_cnt = 0
    one_cnt = 0
    for i in range(len(right_label_column)):
        if right_label_column[i] == 0:
            zero_cnt += right_weight[i]
        else:
            one_cnt += right_weight[i]
    r_counts = [zero_cnt, one_cnt]
    r_counts = np.asarray(r_counts)

    if root_counts.sum() != 0:
        root_uncertainty = 1 - (root_counts.sum() / (len(rootdata))) ** 2 - (
                    (len(rootdata) - root_counts.sum()) / (len(rootdata))) ** 2
    else:
        root_uncertainty = 1

    if l_counts.sum() != 0:
        left_prob = l_counts / l_counts.sum()
        left_uncertainty = 1 - sum(left_prob * left_prob)
    else:
        left_uncertainty = 1

    if r_counts.sum() != 0:
        right_prob = r_counts / r_counts.sum()
        right_uncertainty = 1 - sum(right_prob * right_prob)

    else:
        right_uncertainty = 1

    pl = (l_counts.sum() / (l_counts.sum() + r_counts.sum()))
    pr = (r_counts.sum() / (l_counts.sum() + r_counts.sum()))

    benefit = root_uncertainty - pl * left_uncertainty - pr * right_uncertainty
    return benefit

def get_best_split(n_columns, data, random_subspace):
    columns_indices = list(range(n_columns-1))
    if random_subspace and random_subspace <= len(columns_indices):
        columns_indices = random.sample(population=columns_indices, k=random_subspace)
    max_benefit = float('-inf')
    for i in columns_indices:
        l, r = split_data(data,i)
        if data.shape[1] == 118:
            benefit = cal_weighted_gini(data,l,r)
        else:
            benefit = cal_weighted_gini(data,l,r)
        if benefit > max_benefit:
            max_benefit = benefit
            best_feature = i
    return max_benefit, best_feature

# adaboost algorithm
def adaboost(df,L,d):
    stumps = []
    D = np.zeros(df.shape[0])
    D += 1/df.shape[0]
    D = np.reshape(D,(4874,1))
    data = np.concatenate((df.values, D), axis = 1)
    stump = dt.decisiontree(data,d,0,data.shape[1])
    stumps.append(stump)
    for i in range(L-1):
        acc, e_example = dt.calculate_Accuracy(df,stumps[-1])
        err = 1 - acc
        alpha = cal_alpha(e_example, D)
        for i in range(df.shape[0]):
            if i in e_example:
                D[i] = D[i] * np.exp(alpha)
            else:
                D[i] = D[i] * np.exp(-alpha)
        D = D / D.sum()
        data = np.delete(data,-1,axis = 1)
        data = np.concatenate((data, D), axis = 1)
        stumps.append(dt.decisiontree(data,d,0,data.shape[1]))
    return stumps

# predict example
def adaboost_predictions(df_val, stumps):
    df_predictions = {}
    for i in range(len(stumps)):
        column_name = 'Tree {}'.format(i)
        predictions = rf.decision_tree_predictions(df_val, stumps[i])
        df_predictions[column_name] = predictions

    df_predictions = pd.DataFrame(df_predictions)
    adaboostpred = df_predictions.mode(axis=1)[0]

    return adaboostpred

# get accuracy of train and validation data
def calculate_accuracy_ab(predictions, labels):
    predictions_correct = predictions == labels
    accuracy = predictions_correct.mean()
    return accuracy



