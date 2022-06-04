import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

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
X = df.values[:,1:11]
y = df.values[:,0:1]

def feature_normalize(X):
  mu = np.mean(X, axis = 0)  
  sigma = np.std(X, axis= 0, ddof = 1)  
  X_norm = (X - mu)/sigma
  return X_norm, mu, sigma

X, mu, sigma = feature_normalize(X)

X = np.hstack((np.ones((m,1)), X))
def compute_cost(X, y, theta):
  predictions = X.dot(theta)
  #predictions=predictions.reshape(231,11)
  errors = np.subtract(predictions, y)
  sqrErrors = np.square(errors)
  transPose = errors.T
  J = 1/(2 * m) * (transPose.dot(errors))
  #print("This is  ",J)  
  return J

def gradient_descent(X, y, theta, alpha, iterations):
  
  cost_history = np.zeros(iterations)

  for i in range(iterations):
    predictions = X.dot(theta)
    errors = np.subtract(predictions, y)
    sum_delta = (alpha / m) * X.transpose().dot(errors);
    theta = theta - sum_delta;
    cost_history[i] = compute_cost(X, y, theta)    

  return theta, cost_history

theta = np.zeros((11,1))
iterations = 400;
alpha = 0.15;

theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)