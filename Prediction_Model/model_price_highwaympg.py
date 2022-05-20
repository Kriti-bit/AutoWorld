from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import pandas as pd
import pickle
import numpy as np

df = pd.read_csv('cleaned_automobile_data.csv')

# y = df['price']  # dependent variable
# X = df[['highway-mpg']]  # independent variable

# lm = linear_model.LinearRegression()
# lm.fit(X, y)  # fitting the model
# # save the model
# pickle.dump(lm, open('model_price_from_highwaympg.pkl', 'wb'))

# print(lm.predict([[61]]))  # format of input
# print(f'score: {lm.score(X, y)}')

# # x = df['highway-mpg']
# # y = df['price']
# # f = np.polyfit(x, y, 11)
# # p = np.poly1d(f)

# # pickle.dump(p, open('model_price_from_highwaympg.pkl', 'wb'))


x = np.array(df['highway-mpg'])
y = df['price']
poly = PolynomialFeatures(degree=11, include_bias=False)
poly_features = poly.fit_transform(x.reshape(-1, 1))
lm = linear_model.LinearRegression()
lm.fit(poly_features, y)
# print(f'score: {lm.score(x, y)}')
pickle.dump(lm, open('model_price_from_highwaympg.pkl', 'wb'))
