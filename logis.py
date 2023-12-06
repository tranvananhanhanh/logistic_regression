import numpy as np
from sklearn import datasets
import math
from sklearn.model_selection import train_test_split


data=datasets.load_breast_cancer()
X=data['data']
y=data['target']
print(X.shape)
print(y.shape)

X_tr,X_test,y_tr,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print(X_test.shape)
X_tr=X_tr.reshape(30,455)
X_test=X_test.reshape(30,114)

print(X_tr.shape)
print(y_tr.shape)




def sigmod(z):
   return 1/(1+math.e**(-z))
class logistic:
   def __init__(self,lerning_rate,iterations) -> None:
      self.learning_rate=lerning_rate
      self.iterations=iterations
      self.weights=None
      self.bias=None

   def fit(self,X,y):
      m=X.shape[1]
      n=X.shape[0]

      self.weights=np.zeros((n,))
      self.bias=0
      cost_list=[]

      for i in range(self.iterations):
         Z=np.dot(self.weights.T,X)+self.bias
         a=sigmod(Z)

         cost=-(1/m)*np.sum(y*np.log(a)+(1-y)*np.log(1-a))

         dW=(1/m)*np.dot(a-y,X.T)
         dB=(1/m)*np.sum(a-y)
         W = self.weights - self.learning_rate*dW.T
         B = self.bias - self.learning_rate*dB

         cost_list.append(cost)

         if(i%(self.iterations/10)==0):
            return W, B,cost_list
   
   def predict(self,X):
      Z=np.dot(self.weights.T,X)+self.bias
      a=sigmod(Z)
      return np.where(a>=0.5,1,0)
   
log=logistic(0.0005,100000)
W,B,cost_list=log.fit(X_tr,y_tr)
print(W,B)
y_pre=log.predict(X_test)
k=0
if not np.array_equal(y_pre, y_test):
   k += 1
err=k/len(y_test)
print(err)





def recall(y_true, y_pred):
    TP = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    FN = np.sum(np.logical_and(y_true == 1, y_pred == 0))
    recall = TP / (TP + FN)
    return recall

