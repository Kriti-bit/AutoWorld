from sklearn import linear_model
import pandas as pd
import pickle

df = pd.read_csv('cleaned_automobile_data.csv')

y = df['price']  # dependent variable
X = df[['horsepower']]  # independent variable

lm = linear_model.LinearRegression()
lm.fit(X, y)  # fitting the model
# save the model
pickle.dump(lm, open('model_price_from_horsepower.pkl', 'wb'))

print(lm.predict([[61]]))  # format of input
print(f'score: {lm.score(X, y)}')
