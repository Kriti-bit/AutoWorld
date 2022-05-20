# https://github.com/ifrankandrade/ml-web-app/blob/main/deploy-lr-project/app.py
# https://towardsdatascience.com/how-to-easily-build-your-first-machine-learning-web-app-in-python-c3d6c0f0a01c#e0d6
from turtle import width
from flask import Flask, render_template, request, url_for
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pickle

app = Flask(__name__)
model_HP_EngineSize = pickle.load(
    open('./Prediction_Model/model_price_from_engine_size_horsepower.pkl', 'rb'))
model_HP = pickle.load(
    open('./Prediction_Model/model_price_from_horsepower.pkl', 'rb'))
model_highwaympg = pickle.load(
    open('./Prediction_Model/model_price_from_highwaympg.pkl', 'rb'))
model_weight_from_lbh = pickle.load(
    open('./Prediction_Model/model_weight_from_lbh.pkl', 'rb'))


@app.route("/")
def hello():
    return render_template('index.html')


@app.route("/analysis")
def analysis():
    return render_template('automobile-data-cleaning.html')


@app.route("/tryPredict_HP_EngineSize")
def tryPredict():
    return render_template('tryPredict_HP_EngineSize.html')


@app.route("/predict",  methods=['POST'])
def predict():
    engine_size = request.form['engine-size']
    horsepower = request.form['horsepower']
    prediction = model_HP_EngineSize.predict([[engine_size, horsepower]])
    output = round(prediction[0], 2)
    return render_template('predictions_HP_EngineSize.html', prediction_text=f'For an engine size = {engine_size} and horsepower = {horsepower} price can be estimated as ${output}')


@app.route("/tryPredict_HP")
def tryPredict_HP():
    return render_template('tryPredict_HP.html')


@app.route("/predict_HP",  methods=['POST'])
def predict_HP():
    horsepower = request.form['horsepower']
    prediction = model_HP.predict([[horsepower]])
    output = round(prediction[0], 2)
    return render_template('predictions_HP.html', prediction_text=f'For a horsepower = {horsepower} price can be estimated as ${output}')


@app.route("/tryPredict_highwaympg")
def tryPredict_highwaympg():
    return render_template('tryPredict_highwaympg.html')


@app.route("/predict_highwaympg",  methods=['POST'])
def predict_highwaympg():
    highway_mpg = request.form['highway-mpg']
    highway_mpg = np.array(highway_mpg)
    poly = PolynomialFeatures(degree=11, include_bias=False)
    poly_features = poly.fit_transform(highway_mpg.reshape(-1, 1))
    prediction = model_highwaympg.predict(poly_features)
    output = round(prediction[0], 2)
    return render_template('predictions_highwaympg.html', prediction_text=f'For a Highway MPG = {highway_mpg} price can be estimated as ${output}')


@app.route("/tryPredict_weight_from_lbh")
def tryPredict_weight_from_lbh():
    return render_template('tryPredict_weight_from_lbh.html')


@app.route("/predict_weight_from_lbh",  methods=['POST'])
def predict_weight_from_lbh():
    height = request.form['height']
    width = request.form['width']
    length = request.form['length']
    prediction = model_weight_from_lbh.predict([[height, width, length]])
    output = round(prediction[0], 2)
    return render_template('predictions_weight_from_lbh.html', prediction_text=f'For height = {height}, length = {length} and width = {width} price can be estimated as ${output}')


if __name__ == "__main__":
    app.run()
