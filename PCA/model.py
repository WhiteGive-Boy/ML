import numpy as np
import matplotlib.pyplot as plt
x=np.array([2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1])
y=np.array([2.4,0.7,2.9,2.2,3,2.7,1.6,1.1,1.6,0.9])
plt.plot(x,y,'o')
plt.title("originaldata")#画原始数据
plt.xlabel("x")
plt.ylabel("y")
plt.show()
mean_x=np.mean(x)
mean_y=np.mean(y)
scaled_x=x-mean_x
scaled_y=y-mean_y
data=np.mat([[scaled_x[i],scaled_y[i]] for i in range(len(scaled_x))]) #数据集
print("data:",data)
plt.plot(scaled_x,scaled_y,'o',c='g') #画归一化后的数据
plt.title("scaleddata")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
plt.plot(scaled_x,scaled_y,'o',c='g')#画归一化后的数据 与后面放一起对比
plt.title("scaleddata")
plt.xlabel("x")
plt.ylabel("y")
cov=np.dot(np.transpose(data),data) #求协方差矩阵



eig_val, eig_vec = np.linalg.eig(cov) #求特征向量
print("eig_val:",eig_val)
print("eig_vec:",eig_vec)

xmin ,xmax = scaled_x.min(), scaled_x.max()
ymin, ymax = scaled_y.min(), scaled_y.max()
dx = (xmax - xmin) * 0.2
dy = (ymax - ymin) * 0.2
plt.xlim(xmin - dx, xmax + dx)
plt.ylim(ymin - dy, ymax + dy)
plt.plot([eig_vec[:,0][0],0],[eig_vec[:,0][1],0],c='b')#画特征值1的向量所在直线
plt.plot([eig_vec[:,1][0],0],[eig_vec[:,1][1],0],c='b')#画特征值2的向量所在直线





eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
eig_pairs.sort(reverse=True)


import math
maxeig=eig_pairs[0][1]

PCAresult=np.dot(data,maxeig)
sinx=(maxeig[1])/(math.sqrt(np.dot(maxeig.T,maxeig)))#将pca的结果变换至当前坐标系便于观察

cosx=(maxeig[0])/(math.sqrt(np.dot(maxeig.T,maxeig)))

plt.plot(PCAresult*cosx,PCAresult*sinx,'o',c="r") #画映射结果
plt.show()