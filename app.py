# https://github.com/ifrankandrade/ml-web-app/blob/main/deploy-lr-project/app.py
# https://towardsdatascience.com/how-to-easily-build-your-first-machine-learning-web-app-in-python-c3d6c0f0a01c#e0d6
from flask import Flask, render_template, request, url_for
import pickle

app = Flask(__name__)
model = pickle.load(
    open('./Prediction_Model/model_price_from_engine_size_horsepower.pkl', 'rb'))


@app.route("/")
def hello():
    return render_template('index.html')


@app.route("/analysis")
def hello1():
    return render_template('index.html')


@app.route("/tryPredict")
def tryPredict():
    return render_template('tryPredict_HP_EngineSize.html')


@app.route("/predict",  methods=['POST'])
def predict():
    engine_size = request.form['engine-size']
    horsepower = request.form['horsepower']
    prediction = model.predict([[engine_size, horsepower]])
    output = round(prediction[0], 2)
    return render_template('predictions_HP_EngineSize.html', prediction_text=f'For an engine size = {engine_size} and horsepower= {horsepower} price can be estimated as ${output}K')


if __name__ == "__main__":
    app.run()
