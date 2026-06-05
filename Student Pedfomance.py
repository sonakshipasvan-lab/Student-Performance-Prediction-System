import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("student_exam_data_new.csv")

print("Dataset Preview:")
print(df.head())

X = df[["Study Hours", "Previous Exam Score"]]
y = df["Pass/Fail"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

study_hours = float(input("Enter Study Hours: "))
previous_score = float(input("Enter Previous Exam Score: "))

new_data = pd.DataFrame({
    "Study Hours": [study_hours],
    "Previous Exam Score": [previous_score]
})

prediction = model.predict(new_data)

if prediction[0] == 1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")