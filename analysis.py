# analysis.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df = pd.read_csv('timing_results.csv')


print("Data from CSV:")
print(df)


plt.figure(figsize=(10, 6))
plt.barh(df['Function'], df['Execution Time (seconds)'], color='skyblue')
plt.xlabel('Execution Time (seconds)')
plt.ylabel('Function')
plt.title('Comparison of Execution Times for Different Functions')
plt.show()


df['Speed'] = df['Execution Time (seconds)'].apply(lambda x: 'Fast' if x < 0.5 else 'Slow')
df['Speed_numeric'] = df['Speed'].apply(lambda x: 1 if x == 'Fast' else 0)


print("\nData with Numerical Speed Labels:")
print(df)

X = df[['Execution Time (seconds)']]  # Features (execution time)
y = df['Speed_numeric']               # Labels (Fast/Slow)

# (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
