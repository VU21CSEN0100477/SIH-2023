import pickle
import numpy as np

with open('model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

age=int(input("Enter the age: "))
g=str(input("Enter the Gender: "))
if g=='male' or 'Male':
    gender=1
elif g=='female' or 'Female':
    gender=0
screen_time= float(input("Enter the first input: "))
Active_lifestyle = float(input("Enter the second input: "))
sleep_time = float(input("Enter the third input: "))
stress_level = float(input("Enter the fourth input: "))
mood = float(input("Enter the fivth input: "))
social_relation = float(input("Enter the sixth input: "))


user_input_data = np.array([[age,gender,screen_time, Active_lifestyle,sleep_time,stress_level, mood,social_relation]])  

model_prediction = loaded_model.predict(user_input_data)

if model_prediction[0]==0:
    print("Mental Health Status: Severe")
elif model_prediction[0]==1:
    print("Mental Health Status: Moderate")    
elif model_prediction[0]==2:
    print("Mental Health Status: Mild")
elif model_prediction[0]==3:
    print("Mental Health Status: Good")
else:
    print("Mental Health Status: Optimal")
