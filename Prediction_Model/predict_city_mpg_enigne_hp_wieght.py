from sklearn import linear_model
import pandas as pd
import pickle

df = pd.read_csv('cleaned_automobile_data.csv')

y = df['city-mpg']  # dependent variable
X = df[['curb-weight', 'engine-size', 'horsepower']]  # independent variable

lm = linear_model.LinearRegression()
lm.fit(X, y)  # fitting the model
# save the model
pickle.dump(lm, open('predict_city_mpg_enigne_hp_wieght.pkl', 'wb'))

print(lm.predict([[15, 15, 61]]))  # format of input
print(f'score: {lm.score(X, y)}')
