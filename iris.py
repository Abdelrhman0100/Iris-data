#required libraries
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


#read the dataset
data  = pd.read_csv('iris.csv')
print(data.head())
data.columns = ['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm', 'Species']
print('\n\nColumn names\n\n')
print(data.columns)    

#label encode the target variable
encoder = LabelEncoder()
data.Species = encoder.fit_transform(data.Species)

print(data.head())

#train_test_split
train, test = train_test_split(data, test_size=0.2, random_state=0 )

#split the target from the predictors
x_train = train.drop(['Species'], axis=1 )
y_train = train['Species']

x_test = test.drop(['Species'], axis=1)
y_test = test['Species']


#create the object of the model
lg = LogisticRegression()
#fit the training data
lg.fit(x_train, y_train)
#predict our test data
pred = lg.predict(x_test)

#get back our species names 
print('Predicted Values on Test Data',encoder.inverse_transform(pred))

#score
print('\\ accuracy score in test data : \n\n')
print(accuracy_score(y_test, pred))
