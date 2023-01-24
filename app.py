from flask import Flask, render_template, request, url_for
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pickle

app = Flask(__name__, template_folder='templates')
model_HP_EngineSize = pickle.load(
    open('./Prediction_Model/model_price_from_engine_size_horsepower.pkl', 'rb'))
model_HP = pickle.load(
    open('./Prediction_Model/model_price_from_horsepower.pkl', 'rb'))
model_highwaympg = pickle.load(
    open('./Prediction_Model/model_price_from_highwaympg.pkl', 'rb'))
model_weight_from_lbh = pickle.load(
    open('./Prediction_Model/model_weight_from_lbh.pkl', 'rb'))
model_weight_cityMPG_engineSize = pickle.load(
    open('./Prediction_Model/model_weight_cityMPG_engineSize.pkl', 'rb'))
model_weight_from_lbh_enginesize_HP = pickle.load(
    open('./Prediction_Model/model_weight_from_lbh_enginesize_HP.pkl', 'rb'))
model_city_mpg = pickle.load(
    open('./Prediction_Model/model_city_mpg.pkl', 'rb'))

model_city_mpg_enigne_hp_wieght = pickle.load(
    open('./Prediction_Model/predict_city_mpg_enigne_hp_wieght.pkl', 'rb'))


@app.route("/")
def hello():
    return render_template('index.html')


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


@app.route("/tryPredict_weight_cityMPG_engineSize")
def tryPredict_weight_cityMPG_engineSize():
    return render_template('tryPredict_cityMPG_engineSize.html')


@app.route("/predict_weight_cityMPG_engineSize",  methods=['POST'])
def predict_weight_cityMPG_engineSize():
    engine_size = request.form['engine-size']
    citympg = request.form['city-mpg']
    prediction = model_HP_EngineSize.predict([[engine_size, citympg]])
    output = round(prediction[0], 2)
    return render_template('predictions_cityMPG_engineSize.html', prediction_text=f'For an engine size = {engine_size} and city-mpg = {citympg} price can be estimated as ${output}')


@app.route("/tryPredict_weight_from_lbh_enginesize_HP")
def tryPredict_weight_from_lbh_enginesize_HP():
    return render_template('tryPredict_weight_from_lbh_enginesize_HP.html')


@app.route("/predict_weight_from_lbh_enginesize_HP",  methods=['POST'])
def predict_weight_from_lbh_enginesize_HP():
    height = request.form['height']
    width = request.form['width']
    length = request.form['length']
    engine_size = request.form['engine-size']
    horsepower = request.form['horsepower']
    prediction = model_weight_from_lbh_enginesize_HP.predict(
        [[height, width, length, engine_size, horsepower]])
    output = round(prediction[0], 2)
    return render_template('predictions_weight_from_lbh_enginesize_HP.html', prediction_text=f'For height = {height}, length = {length}, width = {width}, engine size = {engine_size} and horsepower = {horsepower}  price can be estimated as ${output}')


@app.route("/tryPredict_city_mpg")
def tryPredict_city_mpg():
    return render_template('tryPredict_city_mpg.html')


@app.route("/predict_city_mpg",  methods=['POST'])
def predict_city_mpg():
    horsepower = request.form['horsepower']
    prediction = model_city_mpg.predict([[horsepower]])
    output = round(prediction[0], 2)
    return render_template('predictions_city_mpg.html', prediction_text=f'For a horsepower = {horsepower} City MPG can be estimated as {output}')


@app.route("/tryPredict_city_mpg_enigne_hp_wieght")
def tryPredict_city_mpg_enigne_hp_wieght():
    return render_template('tryPredict_city_mpg_enigne_hp_wieght.html')


@app.route("/predict_city_mpg_enigne_hp_wieght",  methods=['POST'])
def predict_city_mpg_enigne_hp_wieght():
    weight = request.form['weight']
    engine_size = request.form['engine-size']
    horsepower = request.form['horsepower']
    prediction = model_city_mpg_enigne_hp_wieght.predict(
        [[weight, engine_size, horsepower]])
    output = round(prediction[0], 2)
    return render_template('predict_city_mpg_enigne_hp_wieght.html', prediction_text=f'For weight = {weight}, engine size = {engine_size} and horsepower = {horsepower}  City MPG can be estimated as {output}')


if __name__ == "__main__":
    app.run()
