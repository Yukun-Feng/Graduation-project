import numpy as np
import math
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from copy import deepcopy
from scipy.interpolate import make_interp_spline


def gen_data(alpha, c, n, dim, test_per):
    
    x_training = np.matrix(np.random.normal(size = (n, dim)))
    x_testing = np.matrix(np.random.normal(size = (int(n * test_per), dim)))
    
    x_training[:,0] = 1.0
    x_testing[:,0] = 1.0
    
    theta = np.matrix(np.zeros((dim,1)))
    
    for j in range(dim):
        theta[j,0] = c * math.sqrt(2*alpha) * pow(j+1, -alpha-1/2)
        
    y_training = np.matmul(x_training, theta) + np.random.normal(size = (n, 1))
    y_testing = np.matmul(x_testing, theta) + np.random.normal(size = (int(n * test_per), 1))
    
    return x_training, y_training, x_testing, y_testing



#def OLS(x_training, y_training, x_testing, y_testing, M, n):
#    xx = np.matmul(x_training.T, x_training)
#    xy = np.matmul(x_training.T, y_training)
#    theta_hat = np.mat(np.zeros((M,1)))
#    error_m = 0.0
#    theta = np.mat(np.zeros((M, 1)))
#    best_M = 0
#    best_error = float("inf")
#    
#    exp_error = 0.0
#    
#    for j in range(M):
#        theta_hat[0:j+1, 0] = np.matmul(np.linalg.inv(xx[0:j+1, 0:j+1]), xy[0:j+1])
#        error_m = y_training - np.matmul(x_training[:, 0:j+1], theta_hat[0:j+1, 0])
#        error_m = np.matmul(error_m.T, error_m)
#        print(error_m, "    ", best_error)
#        if error_m < best_error:
#            best_error = error_m
#            theta[0:j+1, 0] = theta_hat[0:j+1, 0]
#            best_M = j+1
#    
#    exp_error = y_testing - np.matmul(x_testing[:, 0:best_M], theta[0:best_M,0])
#    exp_error = np.matmul(exp_error.T, exp_error)/(int(n/5))
#    
#    return exp_error.getA()[0][0]
    
    


def MMA(x_training, y_training, x_testing, y_testing, M, n):
    """
    

    Parameters
    ----------
    x_training : TYPE np.matrix
        DESCRIPTION. the n*p matrix of the training data
    y_training : TYPE np.matrix
        DESCRIPTION. the n*1 matrix of the training value
    x_testing : TYPE np.matrix
        DESCRIPTION. the (n/5)*p matrix of the testing data
    y_testing : TYPE np.matrix
        DESCRIPTION. the (n/5)*1 amtrix of the testing value
    M : TYPE int
        DESCRIPTION. the number of candidate models
    n : TYPE int
        DESCRIPTION. the number of sample size

    Returns
    -------
    TYPE float
        DESCRIPTION.

    """
    xx = np.matmul(x_training.T, x_training)
    xy = np.matmul(x_training.T, y_training)
    
    theta_hat = np.matrix(np.zeros(shape = (M,M)))
    theta = np.mat(np.zeros(shape = (M, 1)))
    sigma_hat = 0.0
    
    exp_error = 0.0
    mean_exp_error = 0.0
    
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
    
    exp_error = y_testing - np.matmul(x_testing[:,0:M], theta)
    mean_exp_error = np.matmul(exp_error.T, exp_error)/(int(n/5))
    
    return mean_exp_error.getA()[0][0]
    
        

def AIC(x_training, y_training, x_testing, y_testing, M, n):
    xx = np.matmul(x_training.T, x_training)
    xy = np.matmul(x_training.T, y_training)
    
    aic_min = float("inf")
    theta_hat = np.matrix(np.zeros(shape = (1, M))).T
    aic_m = 0.0
    error_m = 0.0
    sigma_m = 0.0
    
    theta = np.matrix(np.zeros(shape = (1, M))).T
    num_p = 0
    exp_error = 0.0
    mean_exp_error = 0.0
    
    for j in range(M):
        theta_hat[0:j+1, 0] = np.matmul(np.linalg.inv(xx[0:j+1, 0:j+1]), xy[0:j+1])
        error_m = y_training - np.matmul(x_training[:, 0:j+1], theta_hat[0:j+1, 0])
        sigma_m = np.matmul(error_m.T, error_m)/(n - j-1)
        aic_m = n * math.log(sigma_m) + 2 * (j+1)
        if aic_m < aic_min:
            aic_min = aic_m
            num_p  = j+1
            theta[0: num_p, 0] = theta_hat[0: num_p, 0]
    
    exp_error = y_testing - np.matmul(x_testing[:, 0:num_p], theta[0: num_p, 0])
    mean_exp_error = np.matmul(exp_error.T, exp_error)/(int(n/5))
    
    return mean_exp_error.getA()[0][0]



def BIC(x_training, y_training, x_testing, y_testing, M, n):
    xx = np.matmul(x_training.T, x_training)
    xy = np.matmul(x_training.T, y_training)
    
    bic_min = float("inf")
    theta_hat = np.matrix(np.zeros(shape = (1, M))).T
    bic_m = 0.0
    error_m = 0.0
    sigma_m = 0.0
    
    theta = np.matrix(np.zeros(shape = (1, M))).T
    num_p = 0
    exp_error = 0.0
    mean_exp_error = 0.0
    
    for j in range(M):
        theta_hat[0:j+1,0] = np.matmul(np.linalg.inv(xx[0:j+1, 0:j+1]), xy[0:j+1, 0])
        error_m = y_training - np.matmul(x_training[:, 0:j+1], theta_hat[0:j+1, 0])
        sigma_m = np.matmul(error_m.T, error_m)/(n-j-1)
        bic_m = n * math.log(sigma_m) + math.log(n) * (j+1)
        
        if bic_min > bic_m:
            bic_min = bic_m
            num_p = j+1
            theta[0: num_p, 0] = theta_hat[0: num_p, 0]
    
    exp_error = y_testing - np.matmul(x_testing[:,0:num_p], theta[0: num_p, 0])
    mean_exp_error = np.matmul(exp_error.T, exp_error)/(int(n/5))
    
    return mean_exp_error.getA()[0][0]


def S_AIC(x_training, y_training, x_testing, y_testing, M, n):
    xx = np.matmul(x_training.T, x_training)
    xy = np.matmul(x_training.T, y_training)
    
    aic_all = np.zeros(M)
    aic_m = 0.0
    error_m = 0.0
    sigma_m = 0.0
    theta_hat = np.mat(np.zeros(shape = (M, M)))
    theta = np.mat(np.zeros(shape = (M, 1)))
    
    exp_error = 0.0
    mean_exp_error = 0.0
    
    for j in range(M):
        theta_hat[0:j+1, j] = np.matmul(np.linalg.inv(xx[0:j+1, 0:j+1]), xy[0:j+1, 0])
        error_m = y_training - np.matmul(x_training[:, 0:j+1], theta_hat[0:j+1, j])
        sigma_m = np.matmul(error_m.T, error_m)/(n - j-1)
        aic_m = n*math.log(sigma_m) + 2*(j+1)
        aic_all[j] = math.exp(-1 * aic_m / 2)
    
    for j in range(M):
        theta += (aic_all[j]/sum(aic_all)) * theta_hat[:,j]
    
    exp_error = y_testing - np.matmul(x_testing[:, 0:M], theta)
    mean_exp_error = np.matmul(exp_error.T, exp_error)/(int(n/5))
    
    return mean_exp_error.getA()[0][0]
    
    

def S_BIC(x_training, y_training, x_testing, y_testing, M, n):
    xx = np.matmul(x_training.T, x_training)
    xy = np.matmul(x_training.T, y_training)
    
    bic_all = np.zeros(M)
    bic_m = 0.0
    error_m = 0.0
    sigma_m = 0.0
    theta_hat = np.mat(np.zeros(shape = (M,M)))
    theta = np.mat(np.zeros(shape = (M, 1)))
    
    exp_error = 0.0
    mean_exp_error = 0.0
    
    for j in range(M):
        theta_hat[0:j+1, j] = np.matmul(np.linalg.inv(xx[0:j+1,0:j+1]), xy[0:j+1, 0])
        error_m = y_training - np.matmul(x_training[:, 0:j+1], theta_hat[0:j+1, j])
        sigma_m = np.matmul(error_m.T, error_m)/(n-j-1)
        bic_m = n*math.log(sigma_m) + math.log(n) * (j+1)
        bic_all[j] = math.exp(-1 * bic_m / 2)
        
    for j in range(M):
        theta += (bic_all[j])/sum(bic_all) * theta_hat[:, j]
        
    exp_error = y_testing - np.matmul(x_testing[:, 0:M], theta)
    mean_exp_error = np.matmul(exp_error.T, exp_error)/(int(n/5))
    
    return mean_exp_error.getA()[0][0]


def JMA(x_training, y_training, x_testing, y_testing, M, n):
    
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
    
    exp_error = y_testing - np.matmul(x_testing[:,0:M], theta)
    mean_exp_error = np.matmul(exp_error.T, exp_error)/(int(n/5))
    
    return mean_exp_error.getA()[0][0]




def main():
    alpha = 0.5
    n = 50
    dim = 100
    test_per = 0.2
    M = round(3 * pow(n, 1/3))
    repeat_n = 1000        #实验重复次数
    
    R_2 = np.linspace(0.1, 0.9, 10)
    c_list = [math.sqrt(R/(1-R)) for R in R_2]
    
    MMA_error_list = np.zeros(len(c_list))
    AIC_error_list = np.zeros(len(c_list))
    BIC_error_list = np.zeros(len(c_list))
    S_AIC_error_list = np.zeros(len(c_list))
    S_BIC_error_list = np.zeros(len(c_list))
    JMA_error_list = np.zeros(len(c_list))
    
    solvers.options['show_progress'] = False
    
    for i, c in enumerate(c_list):
        print(i)
        MMA_error = 0.0
        AIC_error = 0.0
        BIC_error = 0.0
        S_AIC_error = 0.0
        S_BIC_error = 0.0
        JMA_error = 0.0
        for j in range(repeat_n):
            x_training, y_training, x_testing, y_testing = gen_data(alpha, c, n, dim, test_per)
            MMA_error += MMA(x_training, y_training, x_testing, y_testing, M, n)
            AIC_error += AIC(x_training, y_training, x_testing, y_testing, M, n)
            BIC_error += BIC(x_training, y_training, x_testing, y_testing, M, n)
            S_AIC_error += S_AIC(x_training, y_training, x_testing, y_testing, M, n)
            S_BIC_error += S_BIC(x_training, y_training, x_testing, y_testing, M, n)
            JMA_error += JMA(x_training, y_training, x_testing, y_testing, M, n)
        
        MMA_error_list[i] = MMA_error / repeat_n
        AIC_error_list[i] = AIC_error / repeat_n
        BIC_error_list[i] = BIC_error / repeat_n
        S_AIC_error_list[i] = S_AIC_error / repeat_n
        S_BIC_error_list[i] = S_BIC_error / repeat_n
        JMA_error_list[i] = JMA_error / repeat_n
        
        
    
    R_2_smooth = np.linspace(min(R_2), max(R_2), 300)
    MMA_smooth = make_interp_spline(R_2, MMA_error_list)(R_2_smooth)
    AIC_smooth = make_interp_spline(R_2, AIC_error_list)(R_2_smooth)
    BIC_smooth = make_interp_spline(R_2, BIC_error_list)(R_2_smooth)
    S_AIC_smooth = make_interp_spline(R_2, S_AIC_error_list)(R_2_smooth)
    S_BIC_smooth = make_interp_spline(R_2, S_BIC_error_list)(R_2_smooth)
    JMA_smooth = make_interp_spline(R_2, JMA_error_list)(R_2_smooth)
    
    plt.figure(figsize = (10, 10))
    plt.title("n={},  alpha={}".format(n, alpha))
    plt.plot(R_2_smooth, MMA_smooth, color = 'r', label = 'MMA')
    plt.plot(R_2_smooth, AIC_smooth, color = 'y', label = "AIC")
    plt.plot(R_2_smooth, BIC_smooth, color = 'g', label = "BIC")
    plt.plot(R_2_smooth, S_AIC_smooth, color = 'b', label = "S_AIC")
    plt.plot(R_2_smooth, S_BIC_smooth, color = 'k', label = "S_BIC")
    plt.plot(R_2_smooth, JMA_smooth,label = "JMA")
    plt.legend(prop={'size': 10})
    plt.xlim((0.1, 0.9))
    plt.ylim((1.0, 1.8))
    plt.xlabel(r'$R^2$')
    plt.ylabel("Risk")
    plt.show()
    
    
    
    



if __name__ == "__main__":
    main()
    