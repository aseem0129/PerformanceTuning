# analysis.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Read the timing data from the CSV file
df = pd.read_csv('timing_results.csv')

# Display the DataFrame to check the data
print("Data from CSV:")
print(df)

# Step 2: Visualize the execution times with a bar plot
plt.figure(figsize=(10, 6))
plt.barh(df['Function'], df['Execution Time (seconds)'], color='skyblue')
plt.xlabel('Execution Time (seconds)')
plt.ylabel('Function')
plt.title('Comparison of Execution Times for Different Functions')
plt.show()

# Step 3: Add "Fast" or "Slow" labels based on execution time
df['Speed'] = df['Execution Time (seconds)'].apply(lambda x: 'Fast' if x < 0.5 else 'Slow')

# Step 4: Convert "Speed" labels to numerical values (Fast = 1, Slow = 0)
df['Speed_numeric'] = df['Speed'].apply(lambda x: 1 if x == 'Fast' else 0)

# Show the final DataFrame with numerical speed values for logistic regression
print("\nData with Numerical Speed Labels:")
print(df)

# Step 5: Prepare data for logistic regression
X = df[['Execution Time (seconds)']]  # Features (execution time)
y = df['Speed_numeric']               # Labels (Fast/Slow)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Predict the labels for the test set
y_pred = model.predict(X_test)

# Step 8: Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
