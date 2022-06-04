import pandas as pd
import numpy as np
from sklearn import preprocessing
import random
import matplotlib.pyplot as plt 
data = pd.read_csv('depression.csv', header=0)
data = data.dropna()

data.drop('Whichyear', axis=1, inplace=True)
data.drop('Gender', axis=1, inplace=True)
data.drop('Yourlocation', axis=1, inplace=True)
data.drop('happywithlivingplace', axis=1, inplace=True)
data.drop('donerecreationalactivitytoday', axis=1, inplace=True)
data.drop('Age', axis=1, inplace=True)
data.drop('Relationshipstatus', axis=1, inplace=True)
data.drop('Understandingwithfamily', axis=1, inplace=True)
data.drop('feelingpressureinyourstudy', axis=1, inplace=True)
data.drop('supportsyouyouracademiclife', axis=1, inplace=True)
data.drop('usedanysocialmedia', axis=1, inplace=True)
data.drop('satisfiedwithmeal', axis=1, inplace=True)
data.drop('feelingSick/healt issues', axis=1, inplace=True)

dummies = pd.get_dummies(data.feelingrightnow)
merged = pd.concat([data,dummies],axis='columns')
final = merged.drop(['feelingrightnow'], axis='columns')
dummies = pd.get_dummies(final.Areyouhappyinancialy)
merged = pd.concat([final,dummies],axis='columns')
final2 = merged.drop(['Areyouhappyinancialy'], axis='columns')
dummies = pd.get_dummies(final2.satisfiedwithacademicresult)
merged = pd.concat([final2,dummies],axis='columns')
final3 = merged.drop(['satisfiedwithacademicresult'], axis='columns')
dummies = pd.get_dummies(final3.haveinferioritycomplex)
merged = pd.concat([final3,dummies],axis='columns')
final4 = merged.drop(['haveinferioritycomplex'], axis='columns')
df = final4.loc[:,~final4.columns.duplicated()]

X = df.values[:,0:4]

m=X.shape[0] 
n=X.shape[1]

n_iter=100
K=5
Centroids=np.array([]).reshape(n,0) 
for i in range(K):
    rand=random.randint(0,m-1)
    Centroids=np.c_[Centroids,X[rand]]

Output={}
EuclidianDistance=np.array([]).reshape(m,0)
for k in range(K):
       tempDist=np.sum((X-Centroids[:,k])**2,axis=1)
       EuclidianDistance=np.c_[EuclidianDistance,tempDist]
C=np.argmin(EuclidianDistance,axis=1)+1

Y={}
for k in range(K):
    Y[k+1]=np.array([]).reshape(4,0)
for i in range(m):
    Y[C[i]]=np.c_[Y[C[i]],X[i]]
     
for k in range(K):
    Y[k+1]=Y[k+1].T
    
for k in range(K):
     Centroids[:,k]=np.mean(Y[k+1],axis=0)

for i in range(n_iter):
      EuclidianDistance=np.array([]).reshape(m,0)
      for k in range(K):
          tempDist=np.sum((X-Centroids[:,k])**2,axis=1)
          EuclidianDistance=np.c_[EuclidianDistance,tempDist]
      C=np.argmin(EuclidianDistance,axis=1)+1
     
      Y={}
      for k in range(K):
          Y[k+1]=np.array([]).reshape(4,0)
      for i in range(m):
          Y[C[i]]=np.c_[Y[C[i]],X[i]]
     
      for k in range(K):
          Y[k+1]=Y[k+1].T
    
      for k in range(K):
          Centroids[:,k]=np.mean(Y[k+1],axis=0)
      Output=Y


color=['red','blue','green','cyan','magenta']
labels=['cluster1','cluster2','cluster3','cluster4','cluster5']
for k in range(K):
    plt.scatter(Output[k+1][:,0],Output[k+1][:,1],c=color[k],label=labels[k])
plt.scatter(Centroids[0,:],Centroids[1,:],s=300,c='yellow',label='Centroids')
plt.legend()
plt.show()