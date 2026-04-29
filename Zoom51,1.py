import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv(r"Zooms/ML and AI/titanic50.csv")
print(data.head())

# Preprocessing

print(data.isnull().sum()) # how many null data in each column

print("Median of Age Column %.2f"% (data["Age"].median(skipna=True)))
print("Percentage of Missing Records in Carbon %.2f"% ((data["Cabin"].isnull().sum()/data.shape[0])*100))
print("Most Common Boarding Point %s" %data["Embarked"].value_counts().idxmax())

data["Age"] = data["Age"].fillna(data["Age"].median(skipna=True))
data["Fare"] = data["Fare"].fillna(data["Fare"].median(skipna=True))
data["Sex"] = data["Sex"].fillna(data["Sex"].mode()[0])
data["Pclass"] = data["Pclass"].fillna(data["Pclass"].mode()[0])
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].value_counts().idxmax())
data.drop("Cabin", axis=1, inplace=True)
data = data.dropna(subset=["Survived"]) # Drop all rows of a survived column with Nan value.
print(data.isnull().sum())

data.drop("PassengerId", axis=1, inplace=True)
data.drop("Name", axis=1, inplace=True)
data.drop("Ticket", axis=1, inplace=True)
data["TravelAlone"] = np.where((data["SibSp"]+data["Parch"]) > 0, 0, 1) # if ... > 0, then it's 0 for false and 1 for true.
data.drop("SibSp", axis=1, inplace=True)
data.drop("Parch", axis=1, inplace=True)

print(data.head())

label_encoder = preprocessing.LabelEncoder()
data["Sex"] = label_encoder.fit_transform(data["Sex"]) # Change sex into 0 and 1s
data["Embarked"] = label_encoder.fit_transform(data["Embarked"])
print(data.head())

# Classification

x = data[["Pclass","Sex","Age","Fare","Embarked","TravelAlone"]]
y = data["Survived"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)

lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)
y_pred = lr_model.predict(x_test)
print(y_pred)

matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(matrix, annot=True,fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(classification_report(y_test, y_pred))
