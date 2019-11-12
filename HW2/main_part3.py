import numpy as np
import Read_data as rd
import Data_pred_op_ap as dp
import Read_test_data as rtd
import matplotlib.pyplot as plt
import csv

# Kernel (polynomial) Perceptron algorithm

# Create Kernel matrix
def kernel_func(X, y, p):
    return np.power((1 + np.matmul(X, y.T)), p)
# Create u to help us evaluate accuracy of the data
def pred_kernel(gram_matrix, alpha, y_train):
    K = gram_matrix
    return np.sign(np.matmul(K, np.multiply(alpha, y_train)))
# test accuracy of training  and validation data
def test_acc_kernel(pre, y):
    sum_diff = 0
    for i in range(len(pre)):
        if pre[i] != y[i]:
            sum_diff += 1
    return 1 - sum_diff / len(y)
# Data visualization
def plot_acc(acc_train, acc_val, title):
    iters = range(1, len(acc_train)+1)
    plt.plot(iters, acc_train)
    plt.plot(iters, acc_val)
    plt.legend(['training', 'validation'])
    plt.xlabel("iterations")
    plt.ylabel("accuracy")
    plt.title("accuracy vs. iterations (" + title + ")")
    plt.show()
# training data with Kernel polynomial Perceptron
def kernel_perceptron(X_train, y_train, X_val, y_val, p=3, iters=15):
    acc_train, acc_val = [], []
    N = len(X_train)
    alpha = np.zeros(N)

    K_train = kernel_func(X_train, X_train, p)
    K_val = kernel_func(X_val, X_train, p)
    for it in range(iters):
        for i in range(N):
            u = np.sign(np.dot(K_train[i], np.multiply(alpha, y_train)))
            if y_train[i] * u <= 0:
                alpha[i] += 1

        pred = pred_kernel(K_train, alpha, y_train)
        acc_train.append(test_acc_kernel(pred, y_train))

        pred = pred_kernel(K_val, alpha, y_train)
        acc_val.append(test_acc_kernel(pred, y_val))
        print('iterations={}, acc_train={}, acc_val={}'.format(it+1, acc_train[it], acc_val[it]))
    return alpha, acc_train, acc_val
# Predict y_test and write into target file
def pred_test(filename, pred_K):
    csvFile = open(filename, 'w')
    writer = csv.writer(csvFile)
    writer.writerows(pred_K)
    csvFile.close()

def main():
    X_train, y_train = rd.Read_data('pa2_train.csv')
    X_val, y_val = rd.Read_data('pa2_valid.csv')
    X_test = rtd.Read_test_data('pa2_test_no_label.csv')
    p = [1, 2, 3, 4, 5]
    max_acc_val = []
    for i in p:
        a_kp, acc_t_kp, acc_v_kp = kernel_perceptron(X_train, y_train.ravel(), X_val, y_val.ravel(), p=i, iters=15)
        plot_acc(acc_t_kp, acc_v_kp, "Kernel Perceptron")
        K_test = kernel_func(X_test, X_train, i)
        max_acc_val.append(max(acc_v_kp))
    plt.plot(p, max_acc_val)
    plt.xlabel('polynomial number')
    plt.ylabel('validation accuracy')
    plt.title('best validation accuracies versus p')
    plt.show()
    # Generate the prediction file kplabel.csv with best p=3
    a_best, acc_train_best, acc_val_best = kernel_perceptron(X_train, y_train.ravel(), X_val, y_val.ravel(), p=3,
                                                             iters=15)
    K_test = kernel_func(X_test, X_train, 3)
    pred_K = pred_kernel(K_test, a_best, y_train.ravel())
    pred_test('kplabel.csv', pred_K.reshape(-1, 1))

main()