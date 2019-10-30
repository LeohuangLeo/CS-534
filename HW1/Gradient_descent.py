import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Gradient descent calculation
def gradient_descent_r2(X, y, learning_rate, epochs, lamda):

    def get_sse(X, w, y, lamda):
        y_hat = X.dot(w)
        se = (y_hat - y) ** 2
        re = lamda * (w ** 2)
        sse = se.sum() + re.sum()

        return sse

    def get_gradients_r2(X, w, y, lamda):
        y_hat = X.dot(w)
        y_reshaped = y.reshape(-1, 1)
        grads = 2 * (X.T.dot(y_hat - y_reshaped)) + 2 * lamda * w

        return grads

    n = X.shape[1]
    w = np.random.rand(n).reshape(-1, 1)
    w_history = w
    j_history = np.array([get_sse(X, w, y, lamda)])
    n_prints = epochs // 10

    for i in range(epochs):
        grads = get_gradients_r2(X, w, y, lamda)
        w -= learning_rate * grads
        sse = get_sse(X, w, y, lamda)
        w_history = np.hstack([w_history, w])
        j_history = np.append(j_history, sse)

        if i % n_prints == 0:
            print("w: {}; sse: {}".format(w.ravel(), sse))
        if sse > 10e300:
            break;
        if abs(sse) <= 0.5:
            break

    return w, j_history, w_history

def LinearRegression_lr_r2(X_train, y_train, lr, epochs, lamda=0):
    W = []
    J_history = []
    W_history = []
    if lamda == 0:
        for i in lr:
            print('------------Learning Rate:{} without regularization------------'.format(i))
            w, j_history, w_history = gradient_descent_r2(X_train, y_train, i, epochs, lamda)
            W.append(w)
            J_history.append(j_history)
            W_history.append(w_history)
    else:
        for j in lamda:
            for i in lr:
                print('------------Learning Rate:{} with regalarization:{}------------'.format(i,j))
                w, j_history, w_history = gradient_descent_r2(X_train, y_train, i, epochs, j)
                W.append(w)
                J_history.append(j_history)
                W_history.append(w_history)

    return W, J_history, W_history

def lr_plt(data):
    for i in data:
        plt.plot(range(i.size), i)

def ttl_plt_wo_r2(X_train, y_train, lr, epochs):
    W, J_history, W_history = LinearRegression_lr_r2(X_train, y_train, lr, epochs)
    lr_plt(J_history[:1])
    plt.legend(['1'])
    plt.xlabel('iterations')
    plt.ylabel('sse')
    plt.show()
    lr_plt(J_history[1:2])
    plt.legend(['1e-1'])
    plt.xlabel('iterations')
    plt.ylabel('sse')
    plt.show()
    lr_plt(J_history[2:3])
    plt.legend(['1e-2'])
    plt.xlabel('iterations')
    plt.ylabel('sse')
    plt.show()
    lr_plt(J_history[3:4])
    plt.legend(['1e-3'])
    plt.xlabel('iterations')
    plt.ylabel('sse')
    plt.show()
    lr_plt(J_history[4:5])
    plt.legend(['1e-4'])
    plt.xlabel('iterations')
    plt.ylabel('sse')
    plt.show()
    lr_plt(J_history[5:6])
    plt.legend(['1e-5'])
    plt.xlabel('iterations')
    plt.ylabel('sse')
    plt.show()
    lr_plt(J_history[6:7])
    plt.legend(['1e-6'])
    plt.xlabel('iterations')
    plt.ylabel('sse')
    plt.show()
    lr_plt(J_history[7:8])
    plt.legend(['1e-7'])
    plt.xlabel('iterations')
    plt.ylabel('sse')
    plt.show()

    return W, J_history

def ttl_plt_w_r2(X_train, y_train, lr, epochs, lamda):
    W_r2, J_history_r2, W_history_r2 = LinearRegression_lr_r2(X_train, y_train, lr, epochs, lamda)
    J_1 = J_history_r2[:8]
    J_2 = J_history_r2[8:16]
    J_3 = J_history_r2[16:24]
    J_4 = J_history_r2[24:32]
    J_5 = J_history_r2[32:40]
    J_6 = J_history_r2[40:]
    ttl_J = [J_1, J_2, J_3, J_4, J_5, J_6]
    counter = 0
    for J in ttl_J:
        lr_plt(J[:1])
        plt.xlabel('iterations')
        plt.ylabel('sse')
        plt.title('lamda={}, lr={}'.format(lamda[counter],lr[0]))
        plt.show()
        lr_plt(J_history_r2[1:2])
        plt.xlabel('iterations')
        plt.ylabel('sse')
        plt.title('lamda={}, lr={}'.format(lamda[counter],lr[1]))
        plt.show()
        lr_plt(J_history_r2[2:3])
        plt.xlabel('iterations')
        plt.ylabel('sse')
        plt.title('lamda={}, lr={}'.format(lamda[counter],lr[2]))
        plt.show()
        lr_plt(J_history_r2[3:4])
        plt.xlabel('iterations')
        plt.ylabel('sse')
        plt.title('lamda={}, lr={}'.format(lamda[counter],lr[3]))
        plt.show()
        lr_plt(J_history_r2[4:5])
        plt.xlabel('iterations')
        plt.ylabel('sse')
        plt.title('lamda={}, lr={}'.format(lamda[counter],lr[4]))
        plt.show()
        lr_plt(J_history_r2[5:6])
        plt.xlabel('iterations')
        plt.ylabel('sse')
        plt.title('lamda={}, lr={}'.format(lamda[counter],lr[5]))
        plt.show()
        lr_plt(J_history_r2[6:7])
        plt.xlabel('iterations')
        plt.ylabel('sse')
        plt.title('lamda={}, lr={}'.format(lamda[counter],lr[6]))
        plt.show()
        lr_plt(J_history_r2[7:8])
        plt.xlabel('iterations')
        plt.ylabel('sse')
        plt.title('lamda={}, lr={}'.format(lamda[counter],lr[7]))
        plt.show()
        counter += 1

    return W_r2, ttl_J