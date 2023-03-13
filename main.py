import pandas as pd
import numpy as np
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

file = 'pumpkin-seeds.csv'
df = pd.read_csv(file)

#print(df.head())
#print(df.isna())

# string class names to integers
new_col = []
unique_class_name = df.Class.unique()
for element in df.Class:
    new_col.append(np.where(unique_class_name == element)[0][0])

df.Class = new_col
print(df.head())
# descriptive stats
print(df.shape)
print(df.dtypes)
#print(df.describe())

# data visualisation
df.hist()
scatter_matrix(df)
#pyplot.show()

# class distribution
print(df.groupby('Class').size())

# data split
array = df.values
X = array[:, 0:12]
Y = array[:, 12]

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y)

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('RF', RandomForestClassifier()))

for model in models:
    model[1].fit(X_train, Y_train)
    print(f'accuracy for {model[0]}: {accuracy_score(model[1].predict(X_validation), Y_validation)}')







