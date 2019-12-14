import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import sys
import Data_prep as dp
import Adaboost as ada

# check the purity of data. If it return True, the tree doen't have to go more deep
def check_purity(data):
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)
    if len(unique_classes) == 1:
        return True
    else:
        return False

# classify
def classify_data(data):
    if data.shape[1] == 118:
        label_column = data[:, -1]
    else:
        label_column = data[:, -2]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    return classification

# gini index
def cal_gini(rootdata, ldata, rdata):
    root_label_column = rootdata[:, -1]
    left_label_column = ldata[:, -1]
    right_label_column = rdata[:, -1]
    _, root_counts = np.unique(root_label_column, return_counts=True)
    _, l_counts = np.unique(left_label_column, return_counts=True)
    _, r_counts = np.unique(right_label_column, return_counts=True)

    if root_counts.sum() != 0:
        root_prob = root_counts / root_counts.sum()
        root_uncertainty = 1 - sum(root_prob * root_prob)
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

    pl = l_counts.sum() / (l_counts.sum() + r_counts.sum())
    pr = r_counts.sum() / (l_counts.sum() + r_counts.sum())

    benefit = root_uncertainty - pl * left_uncertainty - pr * right_uncertainty
    return benefit

# use gini to find the best feature
def get_best_split(n_columns, data, random_subspace):
    columns_indices = list(range(n_columns-1))
    if random_subspace and random_subspace <= len(columns_indices):
        try:
            random.seed(seed)
            columns_indices = random.sample(population=columns_indices, k=random_subspace)
        except:
            columns_indices = random.sample(population=columns_indices, k=random_subspace)
    max_benefit = float('-inf')
    for i in columns_indices:
        l, r = dp.split_data(data,i)
        if data.shape[1] == 118:
            benefit = cal_gini(data,l,r)
        else:
            benefit = ada.cal_weighted_gini(data,l,r)
        if benefit > max_benefit:
            max_benefit = benefit
            best_feature = i
    return max_benefit, best_feature

# decision tree
def decisiontree(data, mxdepth, depth, n_columns, random_subspace=None):
    d = depth
    if data.shape[1] == 118:
        purity = check_purity(data)
    else:
        purity = ada.check_purity_ada(data)
    if purity or d > mxdepth:
        classification = classify_data(data)
        return classification
    else:
        if data.shape[1] == 119:
            benefit, col = get_best_split(n_columns - 1, data, random_subspace)
        else:
            benefit, col = get_best_split(n_columns, data, random_subspace)
        d_left, d_right = dp.split_data(data, col)

        question = "col: {} =0".format(col)
        sub_tree = {question: []}
        true_ans = decisiontree(d_left, mxdepth, d + 1, n_columns)
        false_ans = decisiontree(d_right, mxdepth, d + 1, n_columns)

        sub_tree[question].append(true_ans)
        sub_tree[question].append(false_ans)
        return sub_tree

# classification
def classify_example(example, tree):
    question = list(tree.keys())[0]
    _, col, n = question.split (" ")
    col = int(col)
    if example[col] == 0:
        answer = tree[question][0]
    else:
        answer = tree[question][1]
    if not isinstance(answer, dict):
        return answer
    else:
        residual_tree = answer
        return classify_example(example,residual_tree)

# get accuracy of train and validation data
def calculate_Accuracy(df, tree):
    cnt = 0
    e_example = []
    for i in range(df.shape[0]-1):
        example = df.iloc[i]
        res = classify_example(example, tree)
        if res == example[df.shape[1]-1]:
            cnt += 1
        else:
            e_example.append(i)
    acc = cnt / (df.shape[0]-1)
    return acc, e_example


