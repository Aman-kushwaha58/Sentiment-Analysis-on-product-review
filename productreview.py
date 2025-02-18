!pip install transformers
from transformers import pipeline

# Use a pre-trained model that can recognize positive, neutral, and negative sentiments
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Get input from user

review = input("Enter a product review: ")

# Perform sentiment analysis
result = sentiment_analyzer(review)

# Display the result
print(result)
