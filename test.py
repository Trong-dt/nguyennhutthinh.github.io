from flask import Flask, render_template, request
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load data
news_df = pd.read_csv('train.csv')
news_df = news_df.fillna(' ')
news_df['content'] = news_df['author'] + ' ' + news_df['title']
X = news_df.drop('label', axis=1)
y = news_df['label']

# Define stemming function
ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Apply stemming function to content column
news_df['content'] = news_df['content'].apply(stemming)

# Vectorize data
X = news_df['content'].values
y = news_df['label'].values
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Fit logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# HTML and CSS for the web page
html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fake News Detector</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            margin: 50px;
        }
        h1 {
            color: #333;
        }
        input {
            width: 70%;
            padding: 10px;
            margin: 10px;
            font-size: 16px;
        }
        button {
            padding: 10px 20px;
            font-size: 16px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
        }
        div {
            margin-top: 20px;
            font-size: 18px;
        }
    </style>
</head>
<body>
    <h1>Fake News Detector</h1>
    <input type="text" id="newsInput" placeholder="Enter news Article">
    <button onclick="predict()">Detect Fake News</button>
    <div id="result"></div>

    <script>
        function predict() {
            var inputText = document.getElementById("newsInput").value;

            // Send the input text to the server for prediction
            fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ input_text: inputText }),
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById("result").innerText = data.result;
            });
        }
    </script>
</body>
</html>
'''

# Route for the home page
@app.route('/')
def home():
    return html_content

# Route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.json['input_text']
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    result = 'The News is Fake' if prediction[0] == 1 else 'The News Is Real'
    return {'result': result}

if __name__ == '__main__':
    app.run(debug=True)

# import streamlit as st
# import numpy as np
# import re
# import pandas as pd
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression

# # Load a smaller sample for development
# news_df_sample = pd.read_csv('train.csv').sample(frac=0.1, random_state=42)
# news_df_sample = news_df_sample.fillna(' ')
# news_df_sample['content'] = news_df_sample['author'] + ' ' + news_df_sample['title']

# # Define stemming function
# ps = PorterStemmer()
# def stemming(content):
#     stemmed_content = re.sub('[^a-zA-Z]',' ',content)
#     stemmed_content = stemmed_content.lower()
#     stemmed_content = stemmed_content.split()
#     stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
#     stemmed_content = ' '.join(stemmed_content)
#     return stemmed_content

# # Apply stemming function to content column in the smaller sample
# news_df_sample['content'] = news_df_sample['content'].apply(stemming)

# # Vectorize data using the smaller sample
# X_sample = news_df_sample['content'].values
# max_features = 5000  # Adjust this number based on your data size
# vector = TfidfVectorizer(max_features=max_features)
# X_sample_transformed = vector.fit_transform(X_sample)

# # Load the entire dataset again
# news_df = pd.read_csv('train.csv')
# news_df = news_df.fillna(' ')

# # Apply stemming function to content column in the entire dataset
# news_df['content'] = news_df['author'] + ' ' + news_df['title']
# news_df['content'] = news_df['content'].apply(stemming)

# # Vectorize the entire dataset
# X = vector.transform(news_df['content'].values)
# y = news_df['label'].values

# # Split data into train and test sets
# X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# # Fit logistic regression model
# model = LogisticRegression()
# model.fit(X_train, Y_train)

# # website
# st.title('Fake News Detector')
# input_text = st.text_input('Enter news Article')

# def prediction(input_text):
#     input_data = vector.transform([input_text])
#     prediction = model.predict(input_data)
#     return prediction[0]

# if input_text:
#     pred = prediction(input_text)
#     if pred == 1:
#         st.write('The News is Fake')
#     else:
#         st.write('The News Is Real')
