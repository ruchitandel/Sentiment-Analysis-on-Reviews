from flask import Flask, request, jsonify
from nltk.stem import PorterStemmer

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400  # Return an error status

    text = data['text']
    
    # Stemming to match different word forms
    ps = PorterStemmer()
    words = text.lower().split()
    stemmed_words = [ps.stem(word) for word in words]

    # Log the stemming process
    for word, stemmed_word in zip(words, stemmed_words):
        print(f"Original: {word}, Stemmed: {stemmed_word}")

    # Expanded keyword lists
    positive_keywords = ['love', 'great', 'excel', 'good', 'fantast', 'amaz', 'happi', 
                         'wonder', 'delight', 'joy', 'satisfi', 'perfect', 'recommend', 'best']
    negative_keywords = ['bad', 'terribl', 'aw', 'hate', 'poor', 'disappoint', 'sad', 
                         'horribl', 'worst', 'unhappi', 'disgust', 'fail', 'wast', 
                         'bor', 'awful', 'wors', 'boring', 'mediocr', 'unpleas', 'dissatisfi']
    
    # Handle common phrases
    if "not bad" in text or "not terrible" in text:
        return jsonify({"prediction": "Positive"})

    # Sentiment scoring
    positive_score = sum([1 for word in stemmed_words if word in positive_keywords])
    negative_score = sum([-1 for word in stemmed_words if word in negative_keywords])

    total_score = positive_score + negative_score
    
    # Handle prediction result
    if total_score < 0:
        prediction = "Negative"
    elif total_score > 0:
        prediction = "Positive"
    else:
        prediction = "Neutral"

    # Log the final prediction and score
    print(f"Total score: {total_score}, Predicted sentiment: {prediction}")

    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
