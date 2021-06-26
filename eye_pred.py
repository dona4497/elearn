import pandas as pd 
import numpy as np
import statistics 
from sklearn.model_selection import train_test_split

df1=pd.read_csv('reading.csv')
df2=pd.read_csv('watching.csv')

df1=pd.concat([df1,df2]).sample(frac=2,replace=True).reset_index(drop=True)

df1['eye']=[i.split(' ') for i in df1.eye]
df1['eye']=[list(map(int,i)) for i in df1.eye]

df1['sum']=[sum(i) for i in df1.eye]
df1['psd']=[statistics.pstdev(i)  for i in df1.eye]
df1['sd']=[statistics.stdev(i)  for i in df1.eye]
df1['var']=[statistics.variance(i)  for i in df1.eye]
df1['pvar']=[statistics.pvariance(i)  for i in df1.eye]


print(df1['psd'].head())
#print(df1['sd'].head())

X=df1[['sum', 'psd', 'sd', 'var','pvar']] # Features
y=df1['label']  # Labels
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1) # Split dataset into training set and test set

#x， y Is the original data set

from sklearn.ensemble import RandomForestClassifier
import pickle
#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

clf.fit(X_train,y_train)#Train the model using the training sets

# save the model to disk
filename = 'eye_model.sav'
pickle.dump(clf, open(filename, 'wb'))

y_pred=clf.predict(X_test) # prediction on test set

from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))




