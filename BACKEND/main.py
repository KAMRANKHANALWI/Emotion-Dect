# Import necessary libraries
import pandas as pd
import seaborn as sns
import neattext.functions as nfx
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib

# Load Dataset
df = pd.read_csv("data_train.csv")
print(df.head)

# Data Cleaning
df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_special_characters)

# Features & Labels
Xfeatures = df['Clean_Text']
ylabels = df['Emotion']

# Split Data
x_train, x_test, y_train, y_test = train_test_split(Xfeatures, ylabels, test_size=0.3, random_state=42)

# Build Pipeline
pipe_lr = Pipeline(steps=[('cv', CountVectorizer()), ('lr', LogisticRegression())])

# Train and Fit Data
pipe_lr.fit(x_train, y_train)

# Check Accuracy
accuracy = pipe_lr.score(x_test, y_test)
print("Accuracy:", accuracy)

# Make A Prediction
exl = "The product was so amazing"
prediction = pipe_lr.predict([exl])
print("Prediction:", prediction)

# Prediction Prob
prediction_proba = pipe_lr.predict_proba([exl])
print("Prediction Probabilities:", prediction_proba)

# To know the classes
classes = pipe_lr.classes_
print("Classes:", classes)

# Save Model & Pipeline
pipeline_file = "emotion_classifier.pkl"
joblib.dump(pipe_lr, pipeline_file)
print("Model saved successfully.")