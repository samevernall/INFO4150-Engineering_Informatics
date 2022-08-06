from flask import Flask, request, jsonify, render_template,url_for, redirect
import sqlite3
import json
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")


@app.route('/model', methods=['GET','POST'])
def model():

    password_input = request.form.get('password')
    password_input = str(password_input)

    if password_input != 'password':
        return redirect(url_for('home'))
    # if password_input != 'password':

    process_db= sqlite3.connect('process_values.db')
    cursor1 = process_db.cursor()

    db_data = cursor1.execute("SELECT * FROM sensor_data").fetchall()
    sensor_df = pd.DataFrame(db_data,columns= ['sensor1','sensor2'])

    sensor1 = sensor_df['sensor1']
    sensor2 = sensor_df['sensor2']

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    
    poly2 = PolynomialFeatures(degree = 2, include_bias = False)
    poly3 = PolynomialFeatures(degree = 3, include_bias = False)

    features_2order = poly2.fit_transform(sensor1.values.reshape(-1,1))
    features_3order = poly3.fit_transform(sensor1.values.reshape(-1,1))

    order2_df = pd.DataFrame(features_2order)
    order3_df = pd.DataFrame(features_3order)

    linreg_poly2 = LinearRegression().fit(order2_df, sensor2)
    linreg_poly3 = LinearRegression().fit(order3_df, sensor2)

    pickle.dump(linreg_poly2, open('linreg_poly2','wb'))
    pickle.dump(linreg_poly3, open('linreg_poly3','wb'))

    from sklearn.metrics import mean_squared_error
    from math import sqrt

    y_pred_poly2 = linreg_poly2.predict(features_2order)
    y_pred_poly3 = linreg_poly3.predict(features_3order)

    mse_2 = mean_squared_error(sensor2, y_pred_poly2)
    mse_3 = mean_squared_error(sensor2, y_pred_poly3)

    rmse_2 = sqrt(mse_2)
    rmse_3 = sqrt(mse_3)


    return render_template("train_model.html", rmse_3 = rmse_3)


@app.route('/predict', methods=['GET','POST'])
def predict():

    sensor1_input = request.form.get('sensor1val')
    input = float(sensor1_input)
    array_input = np.array([input])
    array_input = np.reshape(array_input, (-1,1))

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    
    poly2 = PolynomialFeatures(degree = 2, include_bias = False)
    poly3 = PolynomialFeatures(degree = 3, include_bias = False)

    poly2_sens1 = poly2.fit_transform(array_input)
    poly3_sens1 = poly3.fit_transform(array_input)

    poly2_df_input = pd.DataFrame(poly2_sens1)
    poly3_df_input = pd.DataFrame(poly3_sens1)
    
    model_poly2 = pickle.load(open('linreg_poly2', 'rb'))
    model_poly3 = pickle.load(open('linreg_poly3', 'rb'))

    result_poly2 = model_poly2.predict(poly2_df_input)
    result_poly3 = model_poly3.predict(poly3_df_input)

    data = result_poly3 

    #result sensor1_input, result_poly3 need to be saved

    current_db= sqlite3.connect('process_values.db')
    fresh_cursor = current_db.cursor()

    fresh_cursor.execute('INSERT INTO sensor_data (sensor1, sensor2) VALUES (?, ?)',
            (sensor1_input , result_poly3[0]))
    current_db.commit()
    current_db.close()


    return render_template("predict.html", data=data)
    
app.run(debug=True)