from sklearn import linear_model
import pandas as pd
import pickle

df = pd.read_csv('cleaned_automobile_data.csv')

y = df['curb-weight']  # dependent variable
# independent variable
X = df[['length', 'width', 'height', 'engine-size', 'horsepower']]

lm = linear_model.LinearRegression()
lm.fit(X, y)  # fitting the model
# save the model
pickle.dump(lm, open('model_weight_from_lbh_enginesize_HP.pkl', 'wb'))

print(lm.predict([[15, 15, 15, 15, 15]]))  # format of input
print(f'score: {lm.score(X, y)}')
