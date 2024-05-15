from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('LinearRegressionModel.pkl','rb'))
car=pd.read_csv('Cleandata.csv')

@app.route('/',methods=['GET','POST'])
def index():
    companies=sorted(car['company'].unique())
    car_models=sorted(car['name'].unique())
    year=sorted(car['year'].unique(),reverse=True)
    fuel_type=car['fuel_type'].unique()

    companies.insert(0,'Select Company')
    return render_template('index.html',companies=companies, car_models=car_models, years=year,fuel_types=fuel_type)


@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    name = request.form.get('car_models')
    company = request.form.get('company')
    year=int(request.form.get('year'))
    kms_driven =int(request.form.get('kilo_driven'))
    fuel_type = request.form.get('fuel_type')
    

    print(name,company,year,fuel_type,kms_driven)
    prediction = model.predict(pd.DataFrame([[name,company,year,kms_driven,fuel_type]],columns=['name', 'company','year','kms_driven','fuel_type']))
    print(prediction)

    return str(np.round(prediction[0],2))




if __name__=='__main__':
    app.run()