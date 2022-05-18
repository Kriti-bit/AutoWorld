# https://github.com/ifrankandrade/ml-web-app/blob/main/deploy-lr-project/app.py
# https://towardsdatascience.com/how-to-easily-build-your-first-machine-learning-web-app-in-python-c3d6c0f0a01c#e0d6
from flask import Flask, render_template, request, url_for
import pickle

app = Flask(__name__)
model_HP_EngineSize = pickle.load(
    open('./Prediction_Model/model_price_from_engine_size_horsepower.pkl', 'rb'))
model_HP = pickle.load(
    open('./Prediction_Model/model_price_from_horsepower.pkl', 'rb'))
model_highwaympg = pickle.load(
    open('./Prediction_Model/model_price_from_highwaympg.pkl', 'rb'))


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
    return render_template('predictions_HP_EngineSize.html', prediction_text=f'For an engine size = {engine_size} and horsepower = {horsepower} price can be estimated as ${output}K')


@app.route("/tryPredict_HP")
def tryPredict_HP():
    return render_template('tryPredict_HP.html')


@app.route("/predict_HP",  methods=['POST'])
def predict_HP():
    horsepower = request.form['horsepower']
    prediction = model_HP.predict([[horsepower]])
    output = round(prediction[0], 2)
    return render_template('predictions_HP.html', prediction_text=f'For a horsepower = {horsepower} price can be estimated as ${output}K')


@app.route("/tryPredict_highwaympg")
def tryPredict_highwaympg():
    return render_template('tryPredict_highwaympg.html')


@app.route("/predict_highwaympg",  methods=['POST'])
def predict_highwaympg():
    highway_mpg = request.form['highway-mpg']
    prediction = model_highwaympg.predict([[highway_mpg]])
    output = round(prediction[0], 2)
    return render_template('predictions_highwaympg.html', prediction_text=f'For a Highway MPG = {highway_mpg} price can be estimated as ${output}K')


if __name__ == "__main__":
    app.run()
