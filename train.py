import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import  LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import tensorflow as tf

data = pd.read_csv("data.csv")

X = data.drop("Label", axis=1) 
y = data["Label"]

label_encoder = LabelEncoder()
X['Gender'] = label_encoder.fit_transform(X['Gender'])

X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", x_test.shape)
print("y_train shape:", Y_train.shape)
print("y_test shape:", y_test.shape)


rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

rf_classifier.fit(X_train, Y_train)

y_pred = rf_classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
accuracy=accuracy*100
print(f'Accuracy: {accuracy:.2f}')

f1 = f1_score(y_test, y_pred,average='micro')
f1=f1*100
print(f'F1 Score: {f1:.2f}')


with open('model.pkl', 'wb') as model_file:
    pickle.dump(rf_classifier, model_file)


with open('model.pkl', 'rb') as file:
    model = pickle.load(file)


