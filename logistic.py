
# coding: utf-8

# 使用逻辑回归对德国信用卡欺诈数据分类

# 导入相关包
import pandas as pd
import numpy as np
import os

# 定位到数据文件所在目录
os.chdir("E:/数据分析/数据集")

# 读取数据
data = pd.read_csv("credit-a.csv",header=None)

# 查看是否有缺失值
data.isnull().sum()


# 查看数据描述性统计
data.describe()


#查看数据行列总数
data.shape


# 查看前10行数据
data.head(20)


# 提取特征值
feature = data[data.columns[:-1]]
feature.head()


#提取目标值，将-1替换为0
target = data[15].replace(-1,0)
target.head()

# 划分训练集合测试集
from sklearn.model_selection import train_test_split
feature_train,feature_test,target_train,target_test = train_test_split(feature, target,test_size=0.3)

# 初始化模型
from sklearn.linear_model.logistic import LogisticRegression
logistic_model = LogisticRegression()


# 训练模型
logistic_model.fit(feature_train,target_train)


# 预测
logistic_model.predict(feature_test)


#查看模型准确率
logistic_model.score(feature_test,target_test)





