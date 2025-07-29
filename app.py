from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle
from sklearn.preprocessing import StandardScaler


#importing model
model = pickle.load(open('model.pkl', 'rb'))

#import Standard Scaler
sc = pickle.load(open('scaler.pkl', 'rb')) 

#creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    N = int(request.form['Nitrogen'])
    P = int(request.form['Phosporus'])
    K = int(request.form['Pottasium'])
    Temperture = float(request.form['Temperture'])
    Humidity = float(request.form['Humidity'])
    PH = float(request.form['PH'])  # change to float if decimal PH is possible
    Rainfall = float(request.form['Rainfall'])

    # Convert input into array
    feature_list = [N, P, K, Temperture, Humidity, PH, Rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Scale using the same scalers used during training
    single_pred = sc.transform(single_pred)

    # Make prediction
    prediction = model.predict(single_pred)

    # Crop label dictionary
    crop_dict = {
        1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut',
        6: 'papaya', 7: 'orange', 8: 'apple', 9: 'muskmelon', 10: 'watermelon',
        11: 'grapes', 12: 'mango', 13: 'banana', 14: 'pomegranate', 15: 'lentil',
        16: 'blackgram', 17: 'mungbean', 18: 'mothbeans', 19: 'pigeonpeas',
        20: 'kidneybeans', 21: 'chickpea', 22: 'coffee'
    }

    # Return crop name
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = f"{crop} is the best crop to be cultivated right there"
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data"

    return render_template('index.html', result=result)


#python main
if __name__ == "__main__":
    app.run(debug = True)