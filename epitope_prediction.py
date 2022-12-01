# for numerical computing
import numpy as np




# for dataframes
import pandas as pd



# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")


import matplotlib.pyplot as plt

import pickle


# to split train and test set
from sklearn.model_selection import train_test_split



# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


from sklearn.metrics import accuracy_score




df=pd.read_csv('input_bcell.csv')
print(df)
print(df.shape)
print(df.columns)
print(df.head())
print(df.describe())
print(df.corr())

df = df.drop_duplicates()
print( df.shape )
print(df.isnull().sum())
df=df.dropna()
print(df.isnull().sum())


#converting string value to numerical(int)
dist = (df['parent_protein_id'])
distset = set(dist)
dd = list(distset)
dictOfWords1 = { dd[i] : i for i in range(0, len(dd) )}
df['parent_protein_id'] = df['parent_protein_id'].map(dictOfWords1)


#creating pk1 file

with open('parent_protein_id.pkl', 'wb') as handle:
        pickle.dump(dictOfWords1, handle, protocol = pickle.HIGHEST_PROTOCOL)




#converting string value to numerical(int)
dist = (df['protein_seq'])
distset = set(dist)
dd = list(distset)
dictOfWords2 = { dd[i] : i for i in range(0, len(dd) )}
df['protein_seq'] = df['protein_seq'].map(dictOfWords2)


#creating pk1 file

with open('protein_seq.pkl', 'wb') as handle:
        pickle.dump(dictOfWords2, handle, protocol = pickle.HIGHEST_PROTOCOL)






#converting string value to numerical(int)
dist = (df['peptide_seq'])
distset = set(dist)
dd = list(distset)
dictOfWords3 = { dd[i] : i for i in range(0, len(dd) )}
df['peptide_seq'] = df['peptide_seq'].map(dictOfWords3)


#creating pk1 file

with open('peptide_seq.pkl', 'wb') as handle:
        pickle.dump(dictOfWords3, handle, protocol = pickle.HIGHEST_PROTOCOL)

        

df=df.drop(['parent_protein_id','protein_seq','peptide_seq'], axis=1)


y = df.target

# Create separate object for input features
X = df.drop('target', axis=1)




# Split X and y into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, random_state=0)



# Print number of observations in X_train, X_test, y_train, and y_test
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)



model1= LogisticRegression()
model2=RandomForestClassifier(n_estimators=500)
model3= KNeighborsClassifier(n_neighbors=5)
model4= SVC()
model5=GaussianNB()
model6=DecisionTreeClassifier()



#training
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)
model4.fit(X_train, y_train)
model5.fit(X_train, y_train)
model6.fit(X_train, y_train)

## Predict Test set results
y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)
y_pred3 = model3.predict(X_test)
y_pred4 = model4.predict(X_test)
y_pred5 = model5.predict(X_test)
y_pred6 = model6.predict(X_test)

df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred2})
print(df1)

y_pred1 = np.array(y_pred1).reshape(-1,1)
y_pred2 = np.array(y_pred2).reshape(-1,1)
y_pred3 = np.array(y_pred3).reshape(-1,1)
y_pred4 = np.array(y_pred4).reshape(-1,1)
y_pred5 = np.array(y_pred5).reshape(-1,1)
y_pred6 = np.array(y_pred6).reshape(-1,1)

acc1 = accuracy_score(y_test, y_pred1)  ## get the accuracy on testing data
print("Accurcay of LogisticRegression is {:.2f}%".format(acc1*100))
acc2 = accuracy_score(y_test, y_pred2)  ## get the accuracy on testing data
print("Accurcay of RandomForestClassifier is {:.2f}%".format(acc2*100))
acc3 = accuracy_score(y_test, y_pred3) ## get the accuracy on testing data
print("Accurcay of KNeighborsClassifier is {:.2f}%".format(acc3*100))
acc4 = accuracy_score(y_test, y_pred4)  ## get the accuracy on testing data
print("Accurcay of SVC is {:.2f}%".format(acc4*100))
acc5= accuracy_score(y_test, y_pred5)  ## get the accuracy on testing data
print("Accurcay of GaussianNB is {:.2f}%".format(acc5*100))
acc6= accuracy_score(y_test, y_pred6)  ## get the accuracy on testing data
print("Accurcay of DecisionTreeClassifier is {:.2f}%".format(acc6*100))


#from sklearn.externals import joblib 
import joblib
# Save the model as a pickle in a file 
joblib.dump(model2, 'final_pickle_model.pkl') 

  
# Load the model from the file 
final_model = joblib.load('final_pickle_model.pkl')

pred=final_model.predict(X_test)

acc = accuracy_score(y_test, pred) ## get the accuracy on testing data
print("Accuracy of Final Model is {:.2f}%".format(acc*100))

left = [1, 2, 3, 4, 5, 6] 
height = [73.19, 85.70, 73.76, 72.68, 72.12, 80.57]
tick_label = ['LR', 'RF', 'KNN', 'SVC', 'GNB', 'DT']
  
plt.bar(left, height, tick_label = tick_label,
        width = 0.8, color = ['red', 'green', 'salmon', 'lightblue', 'blue', 'pink'])
plt.xlabel('Algorithms')
plt.ylabel('Accuracy Percentage%')
plt.title('Accuracy Graph')
plt.show()


df=pd.read_csv('input_covid.csv')
df=df.drop(['parent_protein_id','protein_seq','peptide_seq'], axis=1)

pred=final_model.predict(df)
df['result']=pred
print(df)

results=df.result.value_counts()
print(results)


df["result"].value_counts().plot(kind="bar", color=["salmon","lightblue"])
plt.xlabel("1 =can be used for vaccine , 0 = can not be used for vaccine")
plt.title("Covid-19 Epitope Prediction")
plt.show()
