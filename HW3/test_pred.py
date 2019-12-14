import numpy as np
import pandas as pd
import math
import random
import csv
import Data_prep as dp
import Decision_tree as dt

# Since all the method can get good result in both training and validation accuracy
# We decide to use decision to predicet the test file

#predict test file
def calculate_Accuracy_pred(df, tree):
    cnt = 0
    e_example = []
    pred = []
    for i in range(df.shape[0]-1):
        example = df.iloc[i]
        res = dt.classify_example(example, tree)
        pred.append(res)
        if res == example[df.shape[1]-1]:
            cnt += 1
        else:
            e_example.append(i)
    acc = cnt / (df.shape[0]-1)
    return acc, e_example, pred

def main():
    df = dp.Read_data("pa3_train.csv")
    df_test = dp.Read_data("pa3_test.csv")
    tree = dt.decisiontree(df.values,5,0,df.shape[1])
    acc_t, ex, pred = calculate_Accuracy_pred(df_test,tree)
    len(pred)
    outfile = open('pa3_test_pred.csv','w')
    out = csv.writer(outfile)
    out.writerows(map(lambda x: [x], pred))
    outfile.close()

main()