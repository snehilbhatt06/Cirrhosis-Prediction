from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
from sklearn.impute import SimpleImputer

app = Flask(__name__)
CORS(app)
# Load models and encoders
with open("scaler1.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("cirrhosis_model1.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    try:
        # Encode categorical fields
        sex = encoders['Sex'].transform([data['sex']])[0]
        ascites = encoders['Ascites'].transform([data['ascites']])[0]
        hepatomegaly = encoders['Hepatomegaly'].transform([data['hepatomegaly']])[0]
        spiders = encoders['Spiders'].transform([data['spiders']])[0]
        edema = encoders['Edema'].transform([data['edema']])[0]
        drug = encoders['Drug'].transform([data['drug']])[0]

        features = np.array([[data['age'], data['bilirubin'], data['cholesterol'],
                              data['albumin'], data['copper'], data['alk_phos'],
                              data['sgot'], data['tryglicerides'], data['platelets'],
                              data['prothrombin'], sex, ascites, hepatomegaly,
                              spiders, edema, drug]])

        features_imputed = SimpleImputer(strategy='mean').fit_transform(features)
        features_scaled = scaler.transform(features_imputed)
        prediction = model.predict(features_scaled)

        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
