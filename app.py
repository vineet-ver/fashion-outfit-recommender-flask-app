from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the fashion recommendation system
MODEL_PATH = 'fashion_recommender_system.pkl'
DATA_PATH = 'AI_PERSONAL_STYLIST_DATA.csv'

# Global variables to store models and data
models_data = None
df = None

def load_models():
    """Load the trained models and data"""
    global models_data, df
    try:
        if os.path.exists(MODEL_PATH):
            models_data = joblib.load(MODEL_PATH)
            print("Models loaded successfully!")
        else:
            print(f"Model file {MODEL_PATH} not found!")
            
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            print("Data loaded successfully!")
        else:
            print(f"Data file {DATA_PATH} not found!")
            
    except Exception as e:
        print(f"Error loading models: {e}")

def get_fashion_recommendations(skin_tone, hair_color, eye_color, body_shape, body_proportion):
    """
    Get comprehensive fashion recommendations based on user characteristics
    """
    global models_data, df
    
    if models_data is None or df is None:
        return {'error': 'Models not loaded properly'}
    
    try:
        # Encode input features
        encoded_features = []
        user_input = [skin_tone, hair_color, eye_color, body_shape, body_proportion]
        
        for i, feature in enumerate(models_data['input_features']):
            if user_input[i] in models_data['label_encoders'][feature].classes_:
                encoded_val = models_data['label_encoders'][feature].transform([user_input[i]])[0]
                encoded_features.append(encoded_val)
            else:
                # Use the first class as default for unknown values
                encoded_features.append(0)
        
        # Make predictions
        X_pred = np.array(encoded_features).reshape(1, -1)
        
        clothing_pred = models_data['clothing_model'].predict(X_pred)[0]
        color_pred = models_data['color_model'].predict(X_pred)[0]
        
        # Get prediction probabilities for confidence scores
        clothing_proba = models_data['clothing_model'].predict_proba(X_pred)[0]
        color_proba = models_data['color_model'].predict_proba(X_pred)[0]
        
        # Decode predictions
        clothing_rec = models_data['target_encoders']["Do's to Wear"].inverse_transform([clothing_pred])[0]
        color_rec = models_data['target_encoders']['Wear Colors'].inverse_transform([color_pred])[0]
        
        # Get confidence scores
        clothing_confidence = max(clothing_proba)
        color_confidence = max(color_proba)
        
        # Find similar profiles in the dataset to get avoid recommendations
        user_matches = df[(df['Skin Tone'] == skin_tone) & 
                         (df['Hair Color'] == hair_color) & 
                         (df['Body Shape'] == body_shape)]
        
        avoid_items = []
        avoid_colors = []
        if not user_matches.empty:
            avoid_items = user_matches["Don'ts to Wear"].mode().tolist()
            avoid_colors = user_matches['Avoid Colors'].mode().tolist()
        
        return {
            'success': True,
            'user_profile': {
                'skin_tone': skin_tone,
                'hair_color': hair_color,
                'eye_color': eye_color,
                'body_shape': body_shape,
                'body_proportion': body_proportion
            },
            'recommendations': {
                'clothing': clothing_rec,
                'colors': color_rec,
                'clothing_confidence': round(clothing_confidence * 100, 1),
                'color_confidence': round(color_confidence * 100, 1)
            },
            'avoid': {
                'clothing': avoid_items[0] if avoid_items else "No specific items to avoid",
                'colors': avoid_colors[0] if avoid_colors else "No specific colors to avoid"
            }
        }
    except Exception as e:
        return {'success': False, 'error': f"Recommendation error: {str(e)}"}

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/get_options')
def get_options():
    global df
    if df is None:
        return jsonify({'error': 'Data not loaded'})

    options = {
        'skin_tones': sorted(df['Skin Tone'].unique().tolist()),
        'hair_colors': sorted(df['Hair Color'].unique().tolist()),
        'eye_colors': sorted(df['Eye Color'].unique().tolist()),
        'body_shapes': sorted(df['Body Shape'].unique().tolist()),
        'body_proportions': sorted(df['Body Proportion'].unique().tolist())
    }
    return jsonify(options)


@app.route('/recommend', methods=['POST'])
def recommend():
    """Generate fashion recommendations"""
    try:
        data = request.get_json()
        
        # Extract user inputs
        skin_tone = data.get('skin_tone')
        hair_color = data.get('hair_color')
        eye_color = data.get('eye_color')
        body_shape = data.get('body_shape')
        body_proportion = data.get('body_proportion')
        
        # Validate inputs
        if not all([skin_tone, hair_color, eye_color, body_shape, body_proportion]):
            return jsonify({'success': False, 'error': 'All fields are required'})
        
        # Get recommendations
        result = get_fashion_recommendations(skin_tone, hair_color, eye_color, body_shape, body_proportion)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': models_data is not None,
        'data_loaded': df is not None
    })

if __name__ == '__main__':
    print("Starting Fashion Recommendation System...")
    load_models()
    
    # Check if models are loaded
    if models_data is None or df is None:
        print("WARNING: Models or data not loaded properly!")
        print("Make sure 'fashion_recommender_system.pkl' and 'AI_PERSONAL_STYLIST_DATA.csv' are in the same directory")
    else:
        print("All models and data loaded successfully!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
