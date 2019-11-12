import numpy as np
# Data prediction for online perceptron and average perceptron
def data_pred(X, y, W):
    acc_list = []
    for w in W:
        pred_correct = 0
        for i in range(y.shape[0]):
            if y[i] == np.sign(X[i,:].dot(w)):
                pred_correct += 1
        acc = pred_correct / y.shape[0]
        acc_list.append(acc)
    return acc_list