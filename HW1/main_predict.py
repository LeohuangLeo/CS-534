import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Gradient_descent as gd
import Read_norm_data as rnd
import csv

#Predict test data
def main():
    X_train, y_train = rnd.read_norm_data('PA1_train.csv')
    w, j_history, w_history = gd.gradient_descent_r2(X_train, y_train, 1e-5, 10000, 0.1)

    X_test = rnd.read_norm_data('PA1_test.csv')
    y_test = list(X_test.dot(w))

    csvFile = open("predict.csv", "w")
    writer = csv.writer(csvFile)
    writer.writerows(y_test)
    csvFile.close()

main()