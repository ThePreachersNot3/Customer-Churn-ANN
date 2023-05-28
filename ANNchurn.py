#importing libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

#read the dataset in
df = pd.read_csv('DeepLearning\\churn.csv')
#drop the customerID since it wouldnt be of help to our model
df2 = df.drop(['customerID'], axis=1)
#print(df2.info())
#iterating through the totalcharges column to change the dtype after discovering it was an object dtype
df2['TotalCharges'] = df2['TotalCharges'].apply(lambda x: np.NaN if x == ' ' else float(x))

#this method converts into float or int then changes errors into nan
#df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
#df.interpolate(inplace=True)

'''for index, value in df['TotalCharges'].items():
    try:
        float_value = float(value)
        df.at[index, 'TotalCharges'] = float_value
    except:
        df.at[index, 'TotalCharges'] = None

df.dropna(inplace=True)
df['TotalCharges'] = df['TotalCharges'].astype(float)'''

#interpolate is a way of fillna by solving for the NaN values with the known values
#i feel it is a lot better than filling with mean
df3 = df2.interpolate()
#print(df3.info())
#this below is just to check the customers who stayed and those who left
loyal = df3[df3['Churn']=='No'].tenure
opps = df3[df3['Churn']=='Yes'].tenure


#visualization of the number of customers who left and those who stayed by tenure
plt.hist([loyal,opps], color=['green','red'], label=['churn=no','churn=yes'])
plt.ylabel('no of customers')
plt.xlabel('churn or nah')
plt.legend()
#plt.show()
#trying to turn categoricals into 0 and 1s
#so i listed all the column names and looped through
clean = ['Churn','PaperlessBilling','PhoneService','Dependents','Partner','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
for i in df3[clean]:
    #since the two below still means No, it was replac3d
    df3[i].replace(['No phone service', 'No internet service'],'No', inplace=True)

for j in df3[clean]:
    df3[j].replace({'No':0,'Yes':1}, inplace=True)
df3['gender'].replace({'Female':0,'Male':1}, inplace=True)

df4 = pd.get_dummies(data=df3, columns=['InternetService','Contract','PaymentMethod'])
'''def all_col():
    for column in df3:
        #if df3[column].dtypes =='object':
        print(f'{column}: {df4[column].unique()}')
all_col()
'''
#scaled all the numerical columns
scalin = ['tenure','MonthlyCharges','TotalCharges']
mms = MinMaxScaler()

scaled = mms.fit_transform(df4[scalin])
df4['tenure'] = scaled[:,0]
df4['MonthlyCharges'] = scaled[:,1]
df4['TotalCharges'] = scaled[:,2]
#reducing the larger sample to fit the smaller one, a method of balancing dataset
'''
zero_count, one_count = df4['Churn'].value_counts()

zero_churn = df4[df4['Churn']==0]
one_churn = df4[df4['Churn']==1]
new_churn = zero_churn.sample(one_count)

df5 = pd.concat([new_churn, one_churn], axis=0)
print(df5.shape)'''

X = df4.drop(['Churn'],axis=1)
y = df4['Churn']

#using smote, another method of balancing dataset

smote = SMOTE(sampling_strategy='minority')
Xsm, ysm = smote.fit_resample(X,y)
#print(Xsm.shape)

#splitting my data into train and test set
X_train, X_test, y_train, y_test = train_test_split(Xsm,ysm, test_size=0.2, random_state=15, stratify=ysm)

#now creating my ANN model
model = keras.Sequential([
    #hidden layers with using the activation relu
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
#the epoch was this much so as to get an acceptable model with good prediction
model.fit(X_train,y_train, epochs=200)

y_pred = model.predict(X_test)



#using machine learning

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred2 =lr.predict(X_test)

#using the sklearn metrics to find out how well our ML model performed
print(accuracy_score(y_test, y_pred2))
print(classification_report(y_test, y_pred2))
cf = confusion_matrix(y_test, y_pred2)
sns.heatmap(cf, annot=True)
#plt.show()






