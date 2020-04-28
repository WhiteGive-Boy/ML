# -*- coding: utf-8 -*-
import numpy as np


def linearRegression(alpha=0.01, num_iters=400,lamada=1):
    data = np.loadtxt("data.txt", delimiter=",", dtype=np.float64)  # 读取数据
    X = data[:, 0:-1]  # X对应0到倒数第2列
    y = data[:, -1]  # y对应最后一列
    row = len(y)  # 总的数据条数
    col = data.shape[1]  # data的列数
    print("col:",col)
    X, mu, sigma = featureNormaliza(X)  # 归一化
    X = np.hstack((np.ones((row, 1)), X))  # 在X前加一列1
    print(u"\n执行梯度下降算法....\n")
    theta = np.zeros((col, 1))
    y = y.reshape(-1, 1)  # 将行向量转化为列
    theta, J_history = gradientDescent(X, y, theta, alpha, num_iters,lamada)
    theta2=np.zeros((col, 1))
    mul=np.dot(X.transpose(), X)
    normal=np.identity(X.shape[1])
    normal=lamada*normal
    mul=np.linalg.inv(mul+normal)
    mul2=np.dot(np.transpose(X), y)
    theta2=np.dot(mul,mul2)
    return mu, sigma, theta,theta2  # 返回均值mu,标准差sigma,和学习的结果theta


# 归一化feature
def featureNormaliza(X):
    X_norm = np.array(X)  # 将X转化为numpy数组对象，才可以进行矩阵的运算
    # 定义所需变量
    mean = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))

    mean = np.mean(X_norm, 0)  # 求每一列的平均值（0指定为列，1代表行）
    sigma = np.std(X_norm, 0)  # 求每一列的标准差
    for i in range(X.shape[1]):  # 遍历列
        X_norm[:, i] = (X_norm[:, i] - mean[i]) / sigma[i]  # 归一化

    return X_norm, mean, sigma




# 梯度下降算法
def gradientDescent(X, y, theta, alpha, num_iters,lamada=1):
    m = len(y)
    n = len(theta)
    temp = np.zeros((n, 1)) # 暂存每次迭代计算的theta，转化为矩阵形式
    #temp = np.zeros((n, num_iters))
    J_history = np.zeros((num_iters, 1))  # 记录每次迭代计算的代价值
    for i in range(num_iters):  # 遍历迭代次数
        h = np.dot(X, theta)  # 计算内积，matrix可以直接乘
        normal=lamada*np.identity(X.shape[1])
        temp = theta - ((alpha) * (np.dot(np.transpose(X), h - y)+np.dot(normal,theta)))  # 梯度的计算
        theta = temp
        J_history[i] = computerCost(X, y, theta)  # 调用计算代价函数
        print('.', end=' ')
    return theta, J_history


# 计算代价函数
def computerCost(X, y, theta):
    m = len(y)
    J = 0

    J = np.dot((np.transpose(np.dot(X ,theta) - y)) ,(np.dot(X ,theta) - y) )/ (2 * m)  # 计算代价J
    return J




# 测试学习效果（预测）
def predict(mu, sigma, theta):
    result = 0
    # 注意归一化
    predict = np.array([2132,4])
    norm_predict = (predict - mu) / sigma
    final_predict = np.hstack((np.ones((1)), norm_predict))
    result = np.dot(final_predict, theta)  # 预测结果

    return result


if __name__ == "__main__":
    mu, sigma, theta, theta2 = linearRegression(0.01, 400,lamada=1)
    # print u"\n计算的theta值为：\n",theta
    # print u"\n预测结果为：%f"%predict(mu, sigma, theta)
    print("\ntheta1:", theta)
    print("\ntheta2:", theta2)
    result = predict(mu, sigma, theta)
    print("\nresult of theta1:", result)
    result = predict(mu, sigma, theta2)
    print("\nresult of theta2:", result)


    mu, sigma, theta, theta2 = linearRegression(0.01, 400,lamada=1.3)
    # print u"\n计算的theta值为：\n",theta
    # print u"\n预测结果为：%f"%predict(mu, sigma, theta)
    print("\ntheta1:", theta)
    print("\ntheta2:", theta2)
    result = predict(mu, sigma, theta)
    print("\nresult of theta1:", result)
    result = predict(mu, sigma, theta2)
    print("\nresult of theta2:", result)

    mu, sigma, theta, theta2 = linearRegression(0.01, 400,lamada=2)
    # print u"\n计算的theta值为：\n",theta
    # print u"\n预测结果为：%f"%predict(mu, sigma, theta)
    print("\ntheta1:", theta)
    print("\ntheta2:", theta2)
    result = predict(mu, sigma, theta)
    print("\nresult of theta1:", result)
    result = predict(mu, sigma, theta2)
    print("\nresult of theta2:", result)