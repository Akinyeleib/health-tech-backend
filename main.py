from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained models
try:
    cancer_clf = joblib.load('cancer_model.pkl')
    emergency_clf = joblib.load('emergency_model.pkl')
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    exit(1)

# Define the features in the same order as the training data
features = ['age', 'sex', 'family_history', 'fatigue', 'weight_loss', 'pain', 'fever',
            'night_sweats', 'bleeding', 'lumps', 'cough', 'bowel_bladder_changes',
            'pain_severity', 'weight_loss_amount', 'bleeding_severity', 'vital_sign_abnormalities']

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        data = {
            'age': float(request.form['age']),
            'sex': float(request.form['sex']),
            'family_history': float(request.form['family_history']),
            'fatigue': float(request.form['fatigue']),
            'weight_loss': float(request.form['weight_loss']),
            'pain': float(request.form['pain']),
            'fever': float(request.form['fever']),
            'night_sweats': float(request.form['night_sweats']),
            'bleeding': float(request.form['bleeding']),
            'lumps': float(request.form['lumps']),
            'cough': float(request.form['cough']),
            'bowel_bladder_changes': float(request.form['bowel_bladder_changes']),
            'pain_severity': float(request.form['pain_severity']),
            'weight_loss_amount': float(request.form['weight_loss_amount']),
            'bleeding_severity': float(request.form['bleeding_severity']),
            'vital_sign_abnormalities': float(request.form['vital_sign_abnormalities'])
        }
        
        # Convert to numpy array for prediction
        sample = np.array([[data[feature] for feature in features]])
        
        # Make predictions
        cancer_prob = cancer_clf.predict_proba(sample)[0][1]  # Probability of cancer
        emergency_pred = emergency_clf.predict(sample)[0]     # Binary emergency prediction
        
        # Return JSON response
        return jsonify({
            'cancer_risk': f"{(cancer_prob * 100):.2f}%",
            'emergency_status': 'EMERGENCY: Seek immediate medical care' if emergency_pred else 'No emergency detected'
        })
    
    # Render the template for GET request
    # return render_template('index.html')

    return jsonify({
        'success': True,
        'message': 'Server working!'
    })

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
