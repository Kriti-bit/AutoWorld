from sklearn import linear_model
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('cleaned_automobile_data.csv')

y = df['price']  # dependent variable
X = df[['horsepower']]  # independent variable
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, y)

lm = linear_model.LinearRegression()
lm.fit(X, y)  # fitting the model
# save the model
#pickle.dump(clf, open('model_price_from_horsepower.pkl', 'wb'))

# print(lm.predict([[61]]))  # format of input
print(clf.predict([[61]]))
print(f'score: {clf.score(X, y)}')
