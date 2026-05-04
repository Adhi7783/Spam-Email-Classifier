import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score

# Example email content
emails = [
    "Congratulations! You've won a free ticket to the Bahamas. Click here to claim.",
    "Hi John, can we reschedule our meeting to next week?",
    "Limited time offer! Get 50% off on all products. Buy now!",
    "Dear customer, your account has been compromised. Please reset your password.",
    "Hey, are you coming to the party tonight?",
    "You have been selected for a $1000 gift card. Claim your prize now!",
    "Can you send me the report by tomorrow?",
    "Urgent: Update your billing information to avoid service interruption."
]

# Labels: 1 for spam, 0 for ham
labels = [1, 0, 1, 1, 0, 1, 0, 1]

# Initialize CountVectorizer
vectorizer = CountVectorizer()

# Transform text data to feature vectors
X = vectorizer.fit_transform(emails)

# Labels
y = np.array(labels)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Classification report
#print(classification_report(y_test, y_pred))
# Example usage with zero_division parameter
print(classification_report(y_test, y_pred, zero_division=0))
# Precision score with zero_division parameter
precision = precision_score(y_test, y_pred, zero_division=0)
print(f'Precision: {precision}')

# Recall score with zero_division parameter
recall = recall_score(y_test, y_pred, zero_division=0)
print(f'Recall: {recall}')