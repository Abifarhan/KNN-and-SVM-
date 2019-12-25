#Importing Libraries
#basics and Visualization
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

#ML libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#metrics
from statistics import mean
from sklearn.metrics import accuracy_score as score
from sklearn.metrics import explained_variance_score as evs


#Ignore Warning
import warnings as wrn
wrn.filterwarnings('ignore')

# read data
url = "abalone1.csv"
dataAbalone = pd.read_csv(url)
dataAbalone.head()
# print(dataAbalone.head())

#Visualization after doing label encoding
dataAbalone['sex'] = LabelEncoder().fit_transform(dataAbalone['sex'].tolist())
#pairplot
sns.pairplot(data=dataAbalone)

#Heatmap
num_feat = dataAbalone.select_dtypes(include=np.number).columns
plt.figure(figsize= (15, 15))
sns.heatmap(dataAbalone.corr())
# plt.show()
# print(dataset.info())

#Dividing X and y
y = dataAbalone[['sex']]
X = dataAbalone.drop(['sex'], axis = 1)

# print(y.head())
# print(X.head())

# for detail
# print(dataAbalone.info())

# for training data
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 30)

# for check empty data
# print(dataAbalone.isnull().sum)

# feature scaler
scaler = StandardScaler()
scaler.fit(train_X)
x_train = scaler.transform(train_X)
x_test = scaler.transform(test_X)

# make prediction
classification = KNeighborsClassifier(n_neighbors=3)
classification.fit(train_X, train_y)
y_prediction = classification.predict(test_X)
# print(y_prediction)

# confusion matrix
from sklearn.metrics import confusion_matrix
confusMatrik = confusion_matrix(test_y, y_prediction)
# print(confusMatrik)

# accuracy
accuration = classification_report(test_y,y_prediction)
# print(accuration)


#Classification and prediction
#KNN

clf = KNeighborsClassifier(n_neighbors=35)
clf.fit(train_X, train_y)
pred = clf.predict(test_X)
print('Nilai akurasi KNN = ',score(pred, test_y)*100)


#Classification and prediction
#SVM

# for svm linear
clf = SVC()
clf.fit(train_X, train_y)
pred = clf.predict(test_X)
print('Nilai akurasi SVM dalam svm linear= ',score(pred, test_y)*100)


# for kernel non linear
cld = SVC(C=2, kernel= 'rbf', gamma='scale')
cld.fit(train_X, train_y)

prediction = cld.predict(test_X)
print("Nilai akurasi dalam SVM kernel non linear:")
print(round(accuracy_score(test_y, prediction)*100,2))
