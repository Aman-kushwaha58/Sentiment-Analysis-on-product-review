import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import pipeline
import torch
from google.colab import files

# Upload the file
uploaded = files.upload()
df = pd.read_csv('IMDB Dataset.csv')  # Replace with your file name
df.head()

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
import re

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove punctuation and non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)

    # Convert to lowercase
    text = text.lower()

    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text

# Apply preprocessing to the reviews
df['cleaned_review'] = df['review'].apply(preprocess_text)

# Split the data
X = df['cleaned_review']
y = df['sentiment']  # Assuming the sentiment column has 0 or 1 labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load pre-trained model for sentiment analysis
classifier = pipeline('sentiment-analysis')

# Test it with a sample review
sample_text = input("Enter a text: ")
print(classifier(sample_text))

# Predict sentiments for the test set
predictions = []
for review in X_test:
    try:
        result = classifier(review)[0]
        # Map 'POSITIVE' to 1 and 'NEGATIVE' to 0
        if result['label'] == 'POSITIVE':
            predictions.append(1)
        else:
            predictions.append(0)
    except:
        # Handle cases where the model fails (e.g., empty text)
        predictions.append(0)  # Default to negative (0) if there's an error

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:")
print(classification_report(y_test, predictions))