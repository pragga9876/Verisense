from flask import Flask, request, jsonify, render_template
import joblib
import json
from statistics import mean

app = Flask(__name__)

# Load your trained model
model = joblib.load('verisense_model.pkl')

# Load feature columns
with open('feature_columns.json', 'r') as f:
    feature_columns = json.load(f)

def extract_features(text):
    """Extract the same 7 features used in training"""
    text = str(text)
    
    features = {}
    
    # 1. Word count
    features['word_count'] = len(text.split())
    
    # 2. Exclamation count
    features['exclamation_count'] = text.count('!')
    
    # 3. Question count
    features['question_count'] = text.count('?')
    
    # 4. Capital letter ratio
    caps = sum(1 for c in text if c.isupper())
    features['capital_ratio'] = caps / len(text) if len(text) > 0 else 0
    
    # 5. Average word length
    words = text.split()
    if len(words) > 0:
        features['avg_word_length'] = sum(len(w) for w in words) / len(words)
    else:
        features['avg_word_length'] = 0
    
    # 6. Simple subjectivity (opinion words)
    opinion_words = ['think', 'believe', 'feel', 'should', 'would', 'could', 'maybe', 'perhaps', 'likely', 'probably']
    text_lower = text.lower()
    opinion_count = sum(1 for word in opinion_words if word in text_lower)
    features['subjectivity'] = min(opinion_count / 10, 1.0)
    
    # 7. Simple polarity (positive/negative words)
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'love', 'happy', 'perfect']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'angry', 'dangerous', 'fear']
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    total = pos_count + neg_count
    features['polarity'] = (pos_count - neg_count) / total if total > 0 else 0
    
    # Return features in the same order as training
    return [features[col] for col in feature_columns]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'})
    
    # Extract features
    features = extract_features(text)
    
    # Predict
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0]
    
    is_real = bool(prediction == 1)
    confidence = max(probability) * 100
    
    return jsonify({
        'is_real': is_real,
        'authenticity_score': round(confidence, 2),
        'verdict': 'REAL ✅' if is_real else 'FAKE ❌',
        'color': 'green' if is_real else 'red'
    })

if __name__ == '__main__':
    app.run(debug=True)