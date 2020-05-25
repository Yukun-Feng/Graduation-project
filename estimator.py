# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 22:26:29 2020

@author: 29066
"""

import matplotlib.pyplot as plt
import numpy as np
import math
from cvxopt import matrix, solvers
from copy import deepcopy


def loading_data(path):
    
    data = []
    
    with open(path, 'r') as f:
        raw_data = f.readlines()
        
        for tem_data in raw_data:
            tem_data = tem_data.strip()
            tem_data = tem_data.split()
            for item in tem_data:
                data.append(float(item))
    
    X_data = []
    Y_data = []
    X_test = []
    Y_test = []
    
    for i in range(12, 444):
        Y_data.append(data[i])
        X_tem = []
        for j in range(0,12):
            X_tem.append(data[i-j-1])
        X_data.append(X_tem)
    
    X_data = np.array(X_data)
    Y_data = np.array(Y_data)
    
    for i in range(444, 468):
        Y_test.append(data[i])
        X_tem = []
        for j in range(0,12):
            X_tem.append(data[i-j-1])
        X_test.append(X_tem)
    
    Y_test = np.array(Y_test)
        
    return data, X_data, Y_data, X_test, Y_test


def MMA(x_training, y_training, X_test, M, n):

    x_training = np.mat(x_training)
    y_training = np.mat(y_training).T

    x_training = np.mat(x_training)
    y_training = np.mat(y_training)
    xx = np.matmul(x_training.T, x_training)
    xy = np.matmul(x_training.T, y_training)
    
    theta_hat = np.matrix(np.zeros(shape = (M,M)))
    theta = np.mat(np.zeros(shape = (M, 1)))
    sigma_hat = 0.0
    
    
    for j in range(M):
        theta_hat[0:j+1, j] = np.matmul(np.linalg.inv(xx[0:j+1, 0:j+1]), xy[0:j+1])

    e_hat = y_training - np.matmul(x_training[:,0:M], theta_hat[0:M, -1])
    sigma_hat = (np.matmul(e_hat.T, e_hat)/(n-M)).getA()[0][0]
        
    error = y_training - np.matmul(x_training[:,0:M], theta_hat)
    P = matrix(2 * np.matmul(error.T, error))
    q = matrix(2 * np.mat(range(1, M+1)).T * sigma_hat)
    G = matrix(-np.eye(M))
    h = matrix(np.matrix(np.zeros(M)).T)
    A = matrix(np.mat(np.ones(M)))
    b = matrix([1.0])
    result = solvers.qp(P, q, G, h, A, b)
    w = list(result['x'])
    for j in range(M):
        theta += w[j] * theta_hat[:, j]
    
    len_x = len(X_test)
    
    Y_estimator = []
    
    for i in range(len_x):
        e = np.matmul(X_test[i][0:M], theta[0:M]).getA()[0][0]
        Y_estimator.append(e)
        for j in range(i+1, M):
            X_test[j][j-i-1] = e
    
    return Y_estimator

def JMA(x_training, y_training, X_test, M, n):
    
    x_training = np.mat(x_training)
    y_training = np.mat(y_training).T
    
    index = list(range(n))
    
    xx = np.matmul(x_training.T, x_training)
    xy = np.matmul(x_training.T, y_training)
    
    theta_hat = np.matrix(np.zeros(shape = (M,M)))
    theta = np.mat(np.zeros(shape = (M, 1)))
    
    for j in range(M):
        theta_hat[0:j+1, j] = np.matmul(np.linalg.inv(xx[0:j+1, 0:j+1]), xy[0:j+1])
    
    error = np.mat(np.zeros((n,M)))
    
    for m in range(M):
        for i in range(n):
            tem_index = deepcopy(index)
            del(tem_index[i])
            X_i = x_training[tem_index, 0:m+1]
            Y_i = y_training[tem_index]
            XX_i = np.matmul(X_i.T, X_i)
            XY_i = np.matmul(X_i.T, Y_i)
            theta_i = np.matmul(np.linalg.inv(XX_i), XY_i)
            error[i,m] = y_training[i] - np.matmul(x_training[i,0:m+1], theta_i)
            
    P = matrix(2 * np.matmul(error.T, error) / n)
    q = matrix(np.mat(np.zeros(M)).T)
    G = matrix(-np.eye(M))
    h = matrix(np.matrix(np.zeros(M)).T)
    A = matrix(np.mat(np.ones(M)))
    b = matrix([1.0])
    result = solvers.qp(P, q, G, h, A, b)
    w = list(result['x'])
    for j in range(M):
        theta += w[j] * theta_hat[:, j]
    
    Y_estimator = []
    len_x = len(X_test)
    
    for i in range(len_x):
        e = np.matmul(X_test[i][0:M], theta).getA()[0][0]
        Y_estimator.append(e)
        for j in range(i+1, M):
            X_test[j][j-i-1] = e
    
    return Y_estimator

def AIC(x_training, y_training, X_test, M, n):
    
    x_training = np.mat(x_training)
    y_training = np.mat(y_training).T
    
    xx = np.matmul(x_training.T, x_training)
    xy = np.matmul(x_training.T, y_training)
    
    aic_min = float("inf")
    theta_hat = np.matrix(np.zeros(shape = (1, M))).T
    aic_m = 0.0
    error_m = 0.0
    sigma_m = 0.0
    
    theta = np.matrix(np.zeros(shape = (1, M))).T
    num_p = 0
    
    for j in range(M):
        theta_hat[0:j+1, 0] = np.matmul(np.linalg.inv(xx[0:j+1, 0:j+1]), xy[0:j+1])
        error_m = y_training - np.matmul(x_training[:, 0:j+1], theta_hat[0:j+1, 0])
        sigma_m = np.matmul(error_m.T, error_m)/(n - j-1)
        aic_m = n * math.log(sigma_m) + 2 * (j+1)
        if aic_m < aic_min:
            aic_min = aic_m
            num_p  = j+1
            theta[0: num_p, 0] = theta_hat[0: num_p, 0]
            
    len_x = len(X_test)
    X_test = np.mat(X_test)
    
    Y_estimator = []
    
    for i in range(len_x):
        e = np.matmul(X_test[i, 0:num_p], theta[0:num_p, 0]).getA()[0][0]
        Y_estimator.append(e)
        for j in range(i+1, M):
            X_test[j, j-i-1] = e
    
    return Y_estimator


def BIC(x_training, y_training, X_test, M, n):
    
    x_training = np.mat(x_training)
    y_training = np.mat(y_training).T    
    
    xx = np.matmul(x_training.T, x_training)
    xy = np.matmul(x_training.T, y_training)
    
    bic_min = float("inf")
    theta_hat = np.matrix(np.zeros(shape = (1, M))).T
    bic_m = 0.0
    error_m = 0.0
    sigma_m = 0.0
    
    theta = np.matrix(np.zeros(shape = (1, M))).T
    num_p = 0
    
    for j in range(M):
        theta_hat[0:j+1,0] = np.matmul(np.linalg.inv(xx[0:j+1, 0:j+1]), xy[0:j+1, 0])
        error_m = y_training - np.matmul(x_training[:, 0:j+1], theta_hat[0:j+1, 0])
        sigma_m = np.matmul(error_m.T, error_m)/(n-j-1)
        bic_m = n * math.log(sigma_m) + math.log(n) * (j+1)
        
        if bic_min > bic_m:
            bic_min = bic_m
            num_p = j+1
            theta[0: num_p, 0] = theta_hat[0: num_p, 0]

    len_x = len(X_test)
    X_test = np.mat(X_test)

    Y_estimator = []
    
    for i in range(len_x):
        e = np.matmul(X_test[i, 0:num_p], theta[0:num_p, 0]).getA()[0][0]
        Y_estimator.append(e)
        for j in range(i+1, M):
            X_test[j, j-i-1] = e
    
    return Y_estimator

def S_AIC(x_training, y_training, X_test, M, n):
    
    x_training = np.mat(x_training)
    y_training = np.mat(y_training).T    
    
    xx = np.matmul(x_training.T, x_training)
    xy = np.matmul(x_training.T, y_training)
    
    aic_all = np.zeros(M)
    aic_m = 0.0
    error_m = 0.0
    sigma_m = 0.0
    theta_hat = np.mat(np.zeros(shape = (M, M)))
    theta = np.mat(np.zeros(shape = (M, 1)))
    
    for j in range(M):
        theta_hat[0:j+1, j] = np.matmul(np.linalg.inv(xx[0:j+1, 0:j+1]), xy[0:j+1, 0])
        error_m = y_training - np.matmul(x_training[:, 0:j+1], theta_hat[0:j+1, j])
        sigma_m = np.matmul(error_m.T, error_m)/(n - j-1)
        aic_m = n*math.log(sigma_m) + 2*(j+1)
        aic_all[j] = math.exp(-1 * aic_m / 2)
    
    for j in range(M):
        theta += (aic_all[j]/sum(aic_all)) * theta_hat[:,j]
        
    Y_estimator = []
    len_x = len(X_test)
    
    for i in range(len_x):
        e = np.matmul(X_test[i][0:M], theta).getA()[0][0]
        Y_estimator.append(e)
        for j in range(i+1, M):
            X_test[j][j-i-1] = e
    
    return Y_estimator

def S_BIC(x_training, y_training, X_test, M, n):
    
    x_training = np.mat(x_training)
    y_training = np.mat(y_training).T     
    
    xx = np.matmul(x_training.T, x_training)
    xy = np.matmul(x_training.T, y_training)
    
    bic_all = np.zeros(M)
    bic_m = 0.0
    error_m = 0.0
    sigma_m = 0.0
    theta_hat = np.mat(np.zeros(shape = (M,M)))
    theta = np.mat(np.zeros(shape = (M, 1)))
    
    for j in range(M):
        theta_hat[0:j+1, j] = np.matmul(np.linalg.inv(xx[0:j+1,0:j+1]), xy[0:j+1, 0])
        error_m = y_training - np.matmul(x_training[:, 0:j+1], theta_hat[0:j+1, j])
        sigma_m = np.matmul(error_m.T, error_m)/(n-j-1)
        bic_m = n*math.log(sigma_m) + math.log(n) * (j+1)
        bic_all[j] = math.exp(-1 * bic_m / 2)
        
    for j in range(M):
        theta += (bic_all[j])/sum(bic_all) * theta_hat[:, j]
    
    Y_estimator = []
    len_x = len(X_test)
    
    for i in range(len_x):
        e = np.matmul(X_test[i][0:M], theta).getA()[0][0]
        Y_estimator.append(e)
        for j in range(i+1, M):
            X_test[j][j-i-1] = e
    
    return Y_estimator





def main():
    path = r"C:\Users\29066\Desktop\毕业设计\实验模拟\fitting_experiment\data.txt"
    data, X_data, Y_data, X_test, Y_test = loading_data(path)
    
    M = 7
    n = 432
    
    MMA_estimator = MMA(X_data, Y_data, X_test, M, n)
    JMA_estimator = JMA(X_data, Y_data, X_test, M, n)
    AIC_estimator = AIC(X_data, Y_data, X_test, M, n)
    BIC_estimator = BIC(X_data, Y_data, X_test, M, n)
    S_AIC_estimator = S_AIC(X_data, Y_data, X_test, M, n)
    S_BIC_estimator = S_BIC(X_data, Y_data, X_test, M, n)
    
    solvers.options['show_progress'] = False
    
    
    plt.figure(figsize = (10, 10))
    plt.plot(data[468-24:], marker = 'o')
    plt.plot(MMA_estimator, color = 'r', label = 'MMA', linestyle = '--', marker = '^')
    plt.plot(JMA_estimator, color = 'm', label = 'JMA', linestyle = '--', marker = 'v')
    plt.plot(AIC_estimator, color = 'y', label = "AIC", linestyle = '-.', marker = '3')
    plt.plot(BIC_estimator, color = 'g', label = "BIC", linestyle = '-.', marker = '+')
    plt.plot(S_AIC_estimator, color = 'b', label = "S_AIC", linestyle = ':', marker = 's')
    plt.plot(S_BIC_estimator, color = 'k', label = "S_BIC", linestyle = ':', marker = 'p')
    
    plt.legend(prop={'size': 10})
    plt.show()
    
    MMA_stand = []
    JMA_stand = []
    AIC_stand = []
    BIC_stand = []
    S_AIC_stand = []
    S_BIC_stand = []
    
    
    for i in range(24):
         MMA_stand.append(MMA_estimator[i]/Y_test[i])
         JMA_stand.append(JMA_estimator[i]/Y_test[i])
         AIC_stand.append(AIC_estimator[i]/Y_test[i])
         BIC_stand.append(BIC_estimator[i]/Y_test[i])
         S_AIC_stand.append(S_AIC_estimator[i]/Y_test[i])
         S_BIC_stand.append(S_BIC_estimator[i]/Y_test[i])
    
    plt.figure(figsize = (10, 10))
    plt.plot(data[468-24:], marker = 'o')
    plt.plot(MMA_stand, color = 'r', label = 'MMA', linestyle = '--', marker = '^')
    plt.plot(JMA_stand, color = 'm', label = 'JMA', linestyle = '--', marker = 'v')
    plt.plot(AIC_stand, color = 'y', label = "AIC", linestyle = '-.', marker = '3')
    plt.plot(BIC_stand, color = 'g', label = "BIC", linestyle = '-.', marker = '+')
    plt.plot(S_AIC_stand, color = 'b', label = "S_AIC", linestyle = ':', marker = 's')
    plt.plot(S_BIC_stand, color = 'k', label = "S_BIC", linestyle = ':', marker = 'p')
    
    plt.legend(prop={'size': 10})
    plt.show()


if __name__ == "__main__":
    main()