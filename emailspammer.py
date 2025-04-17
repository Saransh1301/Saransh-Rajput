from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Dataset: 15 Non-spam + 7 Spam emails
emails = [
    # Non-spam
    "Hi John, could you please send me the presentation slides from yesterday's meeting?",
    "Reminder: Your dentist appointment is scheduled for Tuesday at 4 PM.",
    "Hey! Are we still on for coffee tomorrow afternoon?",
    "The project deadline has been extended to next Friday. Let’s discuss updates in the next call.",
    "Thanks for your payment. Your order has been confirmed and will be delivered soon.",
    "I’ve attached the revised document. Please review and share your feedback.",
    "Congratulations on completing your certification! Well deserved.",
    "The server maintenance will occur on Saturday from 1 AM to 3 AM. Please save your work.",
    "Can you help me debug this Python code? It’s throwing a TypeError.",
    "Our meeting agenda includes budget review, timeline update, and new team roles.",
    "The team lunch is scheduled for Thursday at 1 PM. Let me know if you're joining.",
    "Please find the invoice attached for your recent transaction.",
    "Don't forget to RSVP for the company’s annual retreat!",
    "Hi Professor, I’ve submitted the assignment via the student portal.",
    "Just checking in to see how you're doing. It’s been a while.",

    # Spam
    "Get rich Quick! Click here to win a million dollars!",
    "Congratulations! You've been selected for a free iPhone!",
    "Limited time offer! Buy now and save 70%",
    "Earn $5000 a week working from home. No experience needed!",
    "Your PayPal account has been compromised. Login to secure it.",
    "You’ve been selected for a free cruise to the Bahamas!",
    "Win a brand new laptop just by signing up here!",
]

labels = [
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1
]

# Vectorize text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("✅ Model Evaluation:")
print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Print actual vs predicted results
print("\n📩 Predictions on Test Emails:\n")
X_test_array = X_test.toarray()

for i in range(len(X_test_array)):
    # Find index of matching row in original X
    for j, x_row in enumerate(X.toarray()):
        if (X_test_array[i] == x_row).all():
            original_email = emails[j]
            break
    actual = y_test[i]
    predicted = y_pred[i]
    print(f"Email: {original_email}")
    print(f"Actual: {'Spam' if actual == 1 else 'Not Spam'}")
    print(f"Predicted: {'Spam' if predicted == 1 else 'Not Spam'}")
    print("-" * 80)

# Predict a new email
new_email = ["You've won a free cruise vacation"]
new_vector = vectorizer.transform(new_email)
predicted_label = model.predict(new_vector)

print("\n🧪 New Email Prediction:")
print("Email:", new_email[0])
print("Predicted as:", "Spam" if predicted_label[0] == 1 else "Not Spam")

