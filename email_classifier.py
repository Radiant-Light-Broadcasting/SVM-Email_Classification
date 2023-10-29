import argparse
import joblib
import re
from bs4 import BeautifulSoup
from langdetect import detect
import nltk
from nltk.corpus import stopwords  # You might need to download the stopwords: nltk.download('stopwords')
from nltk.tokenize import word_tokenize

# List of supported languages (ISO 639-1 language codes)
SUPPORTED_LANGUAGES = ['en', 'es']


def clean_email(text, language):
    """
    Clean the email text by removing HTML tags, special characters, numbers, and stopwords.
    Convert text to lowercase.
    """
    # Remove HTML tags using BeautifulSoup
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    # Remove stopwords
    stop_words = set(stopwords.words(language))
    word_tokens = word_tokenize(text)
    text = ' '.join(word for word in word_tokens if word not in stop_words)

    return text


def classify_email(email_text):
    """
    Load a pre-trained model and a tool for converting texts into numbers.
    Use them to classify the given email text into a category.
    """
    # Detect language
    language = detect(email_text)
    if language not in SUPPORTED_LANGUAGES:
        return "Language not supported"

    # Load the model and the text conversion tool from disk
    model = joblib.load('svm_model.joblib')
    vectorizer = joblib.load('count_vectorizer.joblib')

    # Clean the email text
    cleaned_email = clean_email(email_text, language)

    # Convert the cleaned email text into numbers
    email_features = vectorizer.transform([cleaned_email])

    # Use the model to predict the category of the email
    prediction = model.predict(email_features)

    return prediction[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify an email text.')
    parser.add_argument('email_text', type=str, help='The email text to classify.')

    args = parser.parse_args()
    email_text = args.email_text

    # Classify the email and show the result
    result = classify_email(email_text)
    print(result)
