from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Load the sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    confidence = None
    review = ""

    if request.method == 'POST':
        review = request.form['review']
        result = sentiment_analyzer(review)[0]
        sentiment = result['label'].capitalize()
        confidence = round(result['score'] * 100, 2)

    return render_template('index.html', sentiment=sentiment, confidence=confidence, review=review)

if __name__ == '__main__':
    app.run(debug=True)
