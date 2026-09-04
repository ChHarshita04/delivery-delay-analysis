import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the delivery dataset
data = pd.read_csv("delivery_data.csv")

# Convert date columns
data["Actual Delivery Date"] = pd.to_datetime(data["Actual Delivery Date"])
data["Promised Delivery Date"] = pd.to_datetime(data["Promised Delivery Date"])

# Identify delivery delays
data["Delay Status"] = (
    data["Actual Delivery Date"] > data["Promised Delivery Date"]
).astype(int)

# Calculate delivery delay rate
delay_rate = data["Delay Status"].mean() * 100
print("Delivery Delay Rate:", delay_rate)

# Select features and target
X = data[["Delivery Distance", "Order Volume"]]
y = data["Delay Status"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the classification model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict delivery delays
predictions = model.predict(X_test)

print("Predicted delay status:", predictions)
