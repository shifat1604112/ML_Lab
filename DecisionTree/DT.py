import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn import metrics


df = pd.read_csv("depression.csv")
Class_Status = df["feelingrightnow"]
df.drop(["feelingrightnow","ExpressFeeling","Whichyear"],axis=1,inplace=True)
x_train, x_test, y_train, y_test = train_test_split(df, Class_Status, test_size= 0.4, random_state=1)

class DecisionTree:
    class Question:
        def __init__(self, column,value):
            self.column = column
            self.value = value
            
    class Node:
        def __init__(self,question,trueNode,falseNode,leafNode,prediction):
            self.question = question
            self.leafNode = leafNode
            self.trueNode = trueNode
            self.falseNode = falseNode
            self.prediction = prediction

    def fit(self,x_train,y_train):
        data = x_train
        data["label"] = y_train
        gain, question = self.find_feature(data)
        leafNode=False
        predictions=None
        trueNode=None
        falseNode=None
        if gain==0:
            leafNode = True
            predictions = self.classCount(data)
        else:
            trueBranch,falseBranch = self.branchTree(question,data)
            trueNode = self.train(trueBranch)
            falseNode = self.train(falseBranch)
        self.rootNode = self.Node(question,trueNode,falseNode,leafNode,predictions)
        
    def train(self,data):
        gain, question=self.find_feature(data)
        leafNode=False
        predictions=None
        trueNode=None
        falseNode=None
        if gain==0:
            leafNode = True
            predictions = self.classCount(data)
        else:
            trueBranch,falseBranch=self.branchTree(question,data)
            trueNode = self.train(trueBranch)
            falseNode = self.train(falseBranch)
        return self.Node(question,trueNode,falseNode,leafNode,predictions)
    
    def classCount(self,data):
        probability = data.groupby("label")["label"].count().to_dict()
        for key in probability.keys():
            probability[key] = (probability[key]/len(data))
        return probability
    
    def gini(self,data):
        counts = self.classCount(data)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / float(len(data))
            impurity -= prob_of_lbl**2
        return impurity
    
    def info_gain(self,left, right, Impurity):
        p = float(len(left)) / (len(left) + len(right))
        q = float(len(right)) / (len(left) + len(right))
        return Impurity - p * self.gini(left) - q * self.gini(right)
    
    def find_feature(self,data):
        gain = 0
        question = None
        current_uncertainty = self.gini(data)
        for col in data.drop("label",axis=1):
            values = data[col].unique()
            for val in values:
                q = self.Question(col,val)
                truenode,falsenode = self.branchTree(q,data)
                if len(truenode)==0 or len(falsenode)==0:
                    continue
                g = self.info_gain(truenode, falsenode, current_uncertainty)
                if g >= gain:
                    gain, question = g, q
        return gain,question
    
    def branchTree(self,question,data):
        truenode = data[data[question.column]==question.value]
        falsenode = data[data[question.column]!=question.value]
        return truenode,falsenode
        
    def predict(self,data,probability=False):
        if isinstance(data,pd.Series):
            data=data.to_frame().T
        result=[]
        for row in data.iterrows():
            row=row[1]
            node=self.rootNode
            while not node.leafNode:
                if row[node.question.column]==node.question.value:
                    node=node.trueNode
                else:
                    node=node.falseNode
            if probability:
                result.append(node.prediction)
            else:
                result.append(max(node.prediction, key=node.prediction.get))
        return result

model = DecisionTree()
model.fit(x_train,y_train)
 
y_pred = model.predict(x_train)
y_pred = model.predict(x_test)

print(metrics.classification_report(y_pred,y_test))