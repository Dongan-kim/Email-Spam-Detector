from flask import Flask, request, jsonify
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load the saved model and vectorizer
with open('model.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)

# Function to process the email text
def process_email(email_text):
    stemmer = PorterStemmer()
    stopwords_set = set(stopwords.words('english'))
    email_text = email_text.lower()
    email_text = email_text.translate(str.maketrans('', '', string.punctuation)).split()
    email_text = [stemmer.stem(word) for word in email_text if word not in stopwords_set]
    return ' '.join(email_text)

# Route to predict if an email is spam or not
@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.json['email_text']  # Get email text from request
    processed_email = process_email(email_text)  # Preprocess the email text
    vectorized_email = vectorizer.transform([processed_email]).toarray()  # Transform the email into numerical format
    prediction = clf.predict(vectorized_email)  # Make prediction

    if int(prediction[0]) == 1:
        return jsonify('It is likely spam!')
    else:
        return jsonify('It is likely NOT spam!') # Return the result (spam or not spam)

if __name__ == '__main__':
    app.run(debug=True)

