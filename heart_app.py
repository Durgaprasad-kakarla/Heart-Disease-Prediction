from flask import Flask, request, render_template
import sklearn
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("heart_rf.pkl", "rb"))


@app.route("/")
def home():
    return render_template("Heart_Home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Age
        age = int(request.form['age'])
        # Sex
        sex = request.form['sex']
        # Chest Pain Type
        cp = request.form['cp']
        # Resting Blood Pressure
        trestbps = int(request.form['trestbps'])
        # Serum Cholestoral
        chol = int(request.form['chol'])
        # Fasting Blood Pressure
        fbs=int(request.form['fbs'])
        #Resting electrocardiographic results
        restecg=request.form['restecg']
        #Thalach
        thalach=int(request.form['thalach'])
        #Exercised Induced Angina
        exang=request.form['exang']
        #Old Peak
        oldpeak=(request.form['oldpeak'])
        #slope of the peak excercies
        slope=request.form['slope']
        #ca
        ca=int(request.form['ca'])
        #thal
        thal=request.form['thal']


        prediction=model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
       exang, oldpeak, slope, ca, thal]])

        return render_template('Heart_Home.html',prediction=prediction)

if __name__=="__main__":
    app.run(debug=True)