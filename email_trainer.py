import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
import joblib

# Load the data from a CSV file
dataframe = pd.read_csv("data.csv")

# Split the data into two parts: training and testing
x = dataframe["EmailText"]  # Email texts
y = dataframe["Label"]      # Labels indicating the category of each email

# Using train_test_split to randomly split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Convert email texts into numbers that can be used by the model
cv = CountVectorizer()
features = cv.fit_transform(x_train)

# Create and train the model using the training data, enabling probability estimation
model = svm.SVC(probability=True)
model.fit(features, y_train)

# Save the model and the text conversion tool to disk
joblib.dump(model, 'svm_model.joblib')
joblib.dump(cv, 'count_vectorizer.joblib')
