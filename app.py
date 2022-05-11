# https://github.com/ifrankandrade/ml-web-app/blob/main/deploy-lr-project/app.py
# https://towardsdatascience.com/how-to-easily-build-your-first-machine-learning-web-app-in-python-c3d6c0f0a01c#e0d6
from flask import Flask, render_template, request, url_for
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def hello():
    return render_template('index.html')


@app.route("/analysis")
def hello1():
    return render_template('index.html')


@app.route("/tryPredict")
def tryPredict():
    return render_template('tryPredict.html')


@app.route("/predict",  methods=['POST'])
def predict():
    doors = request.form['num-of-doors']
    horsepower = request.form['horsepower']
    prediction = model.predict([[doors, horsepower]])
    output = round(prediction[0], 2)
    return render_template('predictions.html', prediction_text=f'Doors= {doors} and horsepower= {horsepower} means ${output}K')


if __name__ == "__main__":
    app.run()
