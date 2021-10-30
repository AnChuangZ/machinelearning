import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def Import():
    data=pd.read_csv('pima-indians-diabetes.data.csv',names=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome'])
    
    #处理Glucose
    data_mean=data['Glucose'].mean()
    data.loc[data['Glucose']==0.0,'Glucose']=data_mean

    #处理BloodPressure
    data_mean1=data['BloodPressure'].mean()
    data.loc[data['BloodPressure']==0,'BloodPressure']=data_mean1

    #处理SkinThickness
    data_mean2=data['SkinThickness'].mean()
    data.loc[data['SkinThickness']==0,'SkinThickness']=data_mean2

    #处理Insulin
    data_mean3=data['Insulin'].mean()
    data.loc[data['Insulin']==0,'Insulin']=data_mean3
    
    #处理BMI
    pan_1 = data[data['BMI']<18.5].index.to_list()
    pan_2 = data[(data['BMI']>=18.5)&(data['BMI']<24)].index.to_list()
    pan_3 = data[(data['BMI']>=24)&(data['BMI']<28)].index.to_list()
    pan_4 = data[data['BMI']>=28].index.to_list()

    data.loc[pan_1,'BMI'] = 0.0
    data.loc[pan_2,'BMI'] = 0.25
    data.loc[pan_3,'BMI'] = 0.75
    data.loc[pan_4,'BMI'] = 1.0

    #处理DiabetesPedigreeFunction
    data_mean4=data['DiabetesPedigreeFunction'].mean()
    data.loc[data['DiabetesPedigreeFunction']==0.0,'DiabetesPedigreeFunction']=data_mean4

    return data.iloc[:,0:8],data.iloc[:,8]

#归一化
def Normalization(data):
    transfer=MinMaxScaler(feature_range=[0,1])
    data_new=transfer.fit_transform(data)
    return data_new

#标准化
def Standardization(data):
    transfer=StandardScaler()
    data_new=transfer.fit_transform(data)
    return data_new

#低方差特征过滤
def LowVarianceFiltering(data):
    transfer=VarianceThreshold()
    data_new=transfer.fit_transform(data)
    #print(data_new,data_new.shape)
    return data_new

#相关系数
def CorrelationCoefficient(data):
    feature=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
    for i in range(len(feature)):
        for j in range(i+1,len(feature)):
            print("特征%s与特征%s之间的相关性为%f"%(feature[i], feature[j], pearsonr(data[feature[i]], data[feature[j]])[0]))

#决策树
def DecisionTree(data,target):
    x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.15)

    estimator=DecisionTreeClassifier(criterion="gini")
    estimator.fit(x_train,y_train.astype('int'))

    score=estimator.score(x_test,y_test.astype('int'))
    print("决策树准确率为：",score)

#随机森林
def RandomForest(data,target):
    x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.15)

    estimator=RandomForestClassifier()
    estimator.fit(x_train,y_train.astype('int'))

    score=estimator.score(x_test,y_test.astype('int'))
    print("随机森林准确率为：",score)


if __name__=="__main__":
    data_init,target=Import()       ##对某些特征下的0值做了处理，用特征均值取代，保证较多的可用数据
    #print(data)
    #print(target)
    print("数据归一化：")
    data=Normalization(data_init)  ##在只进行归一化后，通过决策树得到的准确率变化不大，都在0.69~0.72的范围内，使用随机森林准确率在0.75~0.77
    DecisionTree(data,target) 
    RandomForest(data,target)

    print("\n")

    print("数据标准化：")
    data=Standardization(data_init) ##在只进行标准化后，通过决策树得到的准确率在0.71~0.75之间，使用随机森林准确率为0.71~0.78
    DecisionTree(data,target) 
    RandomForest(data,target)

    print("\n")

    print("低方差特征过滤：")   ##在进行低方差过滤，阈值设为5的情况下，特征减少到7个，决策树的准确率在0.7左右，而随机森林的准确率达到了0.8；在阈值为200的情况下，特征减少到6个，决策树的准确率下降到0.64~0.69，而随机森林在0.75~0.84之间
    data=LowVarianceFiltering(data_init)
    DecisionTree(data,target) 
    RandomForest(data,target)

    print("\n")

    #计算两两特征之间的相关系数
    CorrelationCoefficient(data_init)