from sklearn.ensemble import RandomForestRegressor 
import pandas as pd
import statistics
import pickle
df1=pd.read_csv('cognitive.csv')
print(df1.eye.head())

df1['eye']=[i.split(' ') for i in df1.eye]
df1['eye']=[list(map(int,i)) for i in df1.eye]

df1['e_sum']=[sum(i) for i in df1.eye]
df1['e_psd']=[statistics.pstdev(i)  for i in df1.eye]
df1['e_sd']=[statistics.stdev(i)  for i in df1.eye]
df1['e_var']=[statistics.variance(i)  for i in df1.eye]
df1['e_pvar']=[statistics.pvariance(i)  for i in df1.eye]

#print(df1.m_x.head())
df1['m_x']=[i.split(' ') for i in df1.m_x]
df1['m_x']=[list(map(int,i)) for i in df1.m_x]

df1['mx_sum']=[sum(i) for i in df1.m_x]
df1['mx_psd']=[statistics.pstdev(i)  for i in df1.m_x]
df1['mx_sd']=[statistics.stdev(i) if len(i)>2 else i[0]  for i in df1.m_x]
df1['mx_var']=[statistics.variance(i) if len(i)>2 else i[0]  for i in df1.m_x]
df1['mx_pvar']=[statistics.pvariance(i) if len(i)>2 else i[0]  for i in df1.m_x]

df1['m_y']=[i.split(' ') for i in df1.m_y]
df1['m_y']=[list(map(int,i)) for i in df1.m_y]

df1['my_sum']=[sum(i) for i in df1.m_y]
df1['my_psd']=[statistics.pstdev(i) if len(i)>2 else i[0]  for i in df1.m_y]
df1['my_sd']=[statistics.stdev(i) if len(i)>2 else i[0]  for i in df1.m_y]
df1['my_var']=[statistics.variance(i) if len(i)>2 else i[0]  for i in df1.m_y]
df1['my_pvar']=[statistics.pvariance(i) if len(i)>2 else i[0]  for i in df1.m_y]

df1['emot']=[i.split(' ') for i in df1.emot]
df1['emot']=[list(map(int,i)) for i in df1.emot]
#print(df1.emot.head())

df1['em_sum']=[sum(i) for i in df1.emot]
df1['em_psd']=[statistics.pstdev(i) if len(i)>2 else i[0]  for i in df1.emot]
df1['em_sd']=[statistics.stdev(i) if len(i)>2 else i[0]  for i in df1.emot]
df1['em_var']=[statistics.variance(i) if len(i)>2 else i[0]  for i in df1.emot]
df1['em_pvar']=[statistics.pvariance(i) if len(i)>2 else i[0]  for i in df1.emot]

df1['amplitudes']=[i.split(' ') for i in df1.amplitudes]
df1['amplitudes']=[list(map(float,i)) for i in df1.amplitudes]

df1['a_sum']=[sum(i) for i in df1.amplitudes]
df1['a_psd']=[statistics.pstdev(i) if len(i)>2 else i[0]  for i in df1.amplitudes]
df1['a_sd']=[statistics.stdev(i) if len(i)>2 else i[0]  for i in df1.amplitudes]
df1['a_var']=[statistics.variance(i) if len(i)>2 else i[0]  for i in df1.amplitudes]
df1['a_pvar']=[statistics.pvariance(i) if len(i)>2 else i[0]  for i in df1.amplitudes]
un=df1.columns[0]
x=df1.drop(['eye','m_x','m_y','amplitudes','emot','c_state',un], axis = 1) 
print(x.head())
y=df1['c_state']

print(x.columns)
#print(x,y)
 # create regressor object 
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) 
  
# fit the regressor with x and y data 
regressor.fit(x, y)   

filename = 'cognitive_model.sav'
pickle.dump(regressor, open(filename, 'wb'))

print(regressor.predict(x[0:5]))