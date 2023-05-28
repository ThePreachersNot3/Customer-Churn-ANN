import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()

df = pd.read_csv('DeepLearning\\churn.csv')
df2 = df.drop(['customerID'], axis=1)
df2['TotalCharges'] = df2['TotalCharges'].apply(lambda x: np.NaN if x == ' ' else float(x))
df3 = df2.interpolate()
#print(df3.info())
loyal = df3[df3['Churn']=='No'].tenure
opps = df3[df3['Churn']=='Yes'].tenure


plt.hist([loyal,opps], color=['green','red'], label=['churn=no','churn=yes'])
plt.ylabel('no of customers')
plt.xlabel('churn or nah')
plt.legend()
#plt.show()
clean = ['Churn','PaperlessBilling','PhoneService','Dependents','Partner','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
for i in df3[clean]:
    df3[i].replace(['No phone service', 'No internet service'],'No', inplace=True)

for j in df3[clean]:
    df3[j].replace({'No':0,'Yes':1}, inplace=True)
df3['gender'].replace({'Female':0,'Male':1}, inplace=True)
from sklearn.preprocessing import OneHotEncoder
df4 = pd.get_dummies(data=df3, columns=['InternetService','Contract','PaymentMethod'])
'''def all_col():
    for column in df3:
        #if df3[column].dtypes =='object':
        print(f'{column}: {df4[column].unique()}')
all_col()
'''

from sklearn.preprocessing import MinMaxScaler
scalin = ['tenure','MonthlyCharges','TotalCharges']
mms = MinMaxScaler()

scaled = mms.fit_transform(df4[scalin])
df4['tenure'] = scaled[:,0]
df4['MonthlyCharges'] = scaled[:,1]
df4['TotalCharges'] = scaled[:,2]
#reducing the larger sample to fit the smaller one
'''
zero_count, one_count = df4['Churn'].value_counts()

zero_churn = df4[df4['Churn']==0]
one_churn = df4[df4['Churn']==1]
new_churn = zero_churn.sample(one_count)

df5 = pd.concat([new_churn, one_churn], axis=0)
print(df5.shape)'''

from sklearn.model_selection import train_test_split
X = df4.drop(['Churn'],axis=1)
y = df4['Churn']

#using smote

from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority')
Xsm, ysm = smote.fit_resample(X,y)
#print(Xsm.shape)

X_train, X_test, y_train, y_test = train_test_split(Xsm,ysm, test_size=0.2, random_state=15, stratify=ysm)

model = keras.Sequential([
    keras.layers.Dense(20, input_shape=(26,), activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(5, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.fit(X_train,y_train, epochs=200)

y_pred = model.predict(X_test)



#using machine learning
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred2 =lr.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
print(accuracy_score(y_test, y_pred2))
print(classification_report(y_test, y_pred2))
cf = confusion_matrix(y_test, y_pred2)
sns.heatmap(cf, annot=True)
#plt.show()






