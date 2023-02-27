from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model/loan_pred_pkl', 'rb'))

@app.route('/home')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    print(int_features)
    final_features = [np.array(int_features)]
    print(final_features)
    prediction = model.predict(final_features)
    print(prediction)
    if prediction[0] == 0:
        output = "Not Eligible for loan"
    elif prediction[0] == 1:
        output = "Eligible for loan"

    return render_template('index.html', prediction_text=output)



if __name__ == "__main__":
    app.run(debug=True)