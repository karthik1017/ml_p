from flask import Flask, request, render_template
import numpy as np 
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

model = pickle.load(open('models/model.pkl','rb'))

app = Flask(__name__)

@app.route('/')

def hello():
    return render_template('index.html')

@app.route('/predict')

def predict():

    open_stock = float(request.args.get('open'))
    high_stock = float(request.args.get('high'))
    low_stock = float(request.args.get('low'))
    volume = float(request.args.get('vol'))
    change_p = float(request.args.get('change'))

    feature_array = np.array([[open_stock,high_stock,low_stock,volume,change_p]])
    print(feature_array)

    scaler = StandardScaler()
    feature_array = scaler.fit_transform(feature_array)

    predicted_value = float(model.predict(feature_array))

    return render_template('predict.html',predicted_value = predicted_value)

if __name__ == '__main__':
    app.run()



