from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load the dataset
dataset = pd.read_csv("email.csv")

# Features and target
X = dataset["Message"]
y = dataset["Category"]

# Convert text to numerical features
vectorizer = CountVectorizer(max_features=5158)  # Limit to top 5000 features
X_vectorized = vectorizer.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Create the classifier
classifier = GaussianNB()

# Convert the sparse matrix to a dense array
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()

# Train the model with the dense data
classifier.fit(X_train_dense, y_train)

# Make predictions
y_pred = classifier.predict(X_test_dense)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
