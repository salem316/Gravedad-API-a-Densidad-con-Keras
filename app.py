import numpy as np
from flask import Flask, request, render_template, url_for
from keras.models import load_model


app = Flask(__name__)
model = load_model("model.h5")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods = ['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    return render_template('home.html', prediction_text=" Densidad específica: {} ".format(prediction[0][0]))

if __name__ == '__main__':
    app.run(debug=True)