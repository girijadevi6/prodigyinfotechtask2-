
# prodigyinfotechtask2-
explanation:
This code performs sentiment analysis on a dataset of tweets, utilizing a Decision Tree Classifier to predict the sentiment based on the text of the tweets. Here's a detailed breakdown of each step:

1. Import Required Libraries
The necessary libraries for data handling, vectorization, and machine learning are imported:

pandas for handling data in DataFrame format.
numpy for numerical computations.
matplotlib.pyplot for any future visualization.
sklearn tools for machine learning, including:
DecisionTreeClassifier for the model.
LabelEncoder for encoding target labels.
TfidfVectorizer for converting text data into a numerical matrix.
train_test_split for splitting the data into training and testing sets.
accuracy_score to evaluate model performance.
2. Load and Inspect Data
CSV File: The dataset is loaded from a remote CSV file hosted on GitHub using pandas.read_csv().
df.head(5): Displays the first 5 rows of the dataset.
df.columns & df.info(): Provides information about the columns and dataset structure, like missing values and data types.
3. Data Cleaning
The columns of interest (columns 2 and 3, which likely represent the sentiment and tweet text) are selected using df[[2,3]].
Reset Index: The index is reset after the column selection.
The selected columns are renamed to ['Sentiment', 'Text'].
Drop Missing Values: Rows with missing text values are dropped.
Fill Missing Text: Any remaining missing values in the Text column are filled with an empty string ('').
4. Prepare Input and Output Data
Input Data: input_data is the tweet text, which will be used to predict the sentiment.
Output Data: output_data contains the sentiment labels, which are the target values for classification.
5. TF-IDF Vectorization
TfidfVectorizer: Converts the textual data into numerical features using the Term Frequency-Inverse Document Frequency (TF-IDF) method. The max_features=5000 limits the vector size to 5000 features.
tfidf.fit_transform(): The Text column is transformed into a numerical matrix using the TF-IDF approach.
6. Label Encoding
LabelEncoder: Encodes the sentiment labels (e.g., negative, neutral, positive) as numeric values.
7. Train-Test Split
The data is split into training and testing sets using an 80-20 ratio (test_size=0.2) with a random state of 42 for reproducibility.
8. Train Decision Tree Classifier
A Decision Tree Classifier model is instantiated and trained on the training set (X_train and y_train).
9. Model Evaluation
Prediction: The trained model predicts the sentiments on the test data.
Accuracy Score: The accuracy of the model is calculated using accuracy_score(), which compares the predicted sentiments (y_pred) with the actual sentiments (y_test).
10. Predict User Input Sentiment
Input Transformation: The TF-IDF vectorizer is used to transform a new user input (e.g., "very happy to see u") into the appropriate numeric format.
Sentiment Prediction: The trained Decision Tree model predicts the sentiment of the user input.
Label Decoding: The numeric prediction is transformed back into the corresponding sentiment label using inverse_transform().

Code:
import pandas as pd
import numpy as np
import matplotlib.pyplot as mp
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/twitter_sentiment.csv", header=None, index_col=0)
df = df[[2, 3]].reset_index(drop=True)  # Select relevant columns and reset index
df.columns = ['Sentiment', 'Text']  # Rename columns
df = df.dropna(subset=['Text'])  # Drop rows with missing Text
df['Text'] = df['Text'].fillna('')  # Fill any remaining missing text with empty string

# Prepare input and output data
input_data = df.drop(columns=['Sentiment'])
output_data = df['Sentiment']

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)  # Max 5000 features
X = tfidf.fit_transform(input_data['Text'])  # Convert text to numeric vectors

# Label Encoding for Sentiment
label = LabelEncoder()
y = label.fit_transform(output_data)  # Convert sentiment to numeric

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make Predictions and Calculate Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Predict sentiment for user input
user_input_transformed = tfidf.transform(['very happy to see u'])  # Transform new input
predicted_sentiment_numeric = model.predict(user_input_transformed)
predicted_sentiment_label = label.inverse_transform(predicted_sentiment_numeric)  # Decode prediction
print("Predicted Sentiment:", predicted_sentiment_label)
![image](https://github.com/user-attachments/assets/7ecea31f-a0ac-4402-bcbf-e20f9ca5edbb)
