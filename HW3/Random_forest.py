import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import sys
import Decision_tree as dt

# bootstrapping
def bootstrapping(train_df, n_bootstrap):
    bootstrap_indices = np.random.randint(low=0, high=len(train_df), size=n_bootstrap)
    df_bootstrapped = train_df.iloc[bootstrap_indices]
    return df_bootstrapped

# random forest
def random_forest_algorithm(train_df, n_trees, n_features, dt_max_depth):
    forest = []
    for i in range(n_trees):
        tree = dt.decisiontree(train_df.values, dt_max_depth, 0, train_df.shape[1], n_features)
        forest.append(tree)
    return forest

# predict for one example
def decision_tree_predictions(test_df, tree):
    predictions = test_df.apply(dt.classify_example, args=(tree,), axis=1)
    return predictions

# predict for all examples
def random_forest_predictions(df_val, forest):
    df_predictions = {}
    for i in range(len(forest)):
        column_name = 'Tree {}'.format(i)
        predictions = decision_tree_predictions(df_val, forest[i])
        df_predictions[column_name] = predictions
    df_predictions = pd.DataFrame(df_predictions)
    random_forest_predictions = df_predictions.mode(axis=1)[0]
    return random_forest_predictions

# get accuracy of train and validation data
def calculate_accuracy_rf(predictions, labels):
    predictions_correct = predictions == labels
    accuracy = predictions_correct.mean()
    return accuracy

