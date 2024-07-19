from flask import Flask, render_template, request, redirect, url_for
import pickle

app = Flask(__name__, static_url_path='/static')


# Load the pickled machine learning model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input values from the form
        age = float(request.form['Age'])
        g = str(request.form['Gender'])
        if len(g)==4:
            gender=1
        else:
            gender=0
        screen_time = float(request.form['ScreenTime'])
        active_lifestyle = float(request.form['ActiveLifestyle'])
        sleep_time = float(request.form['SleepTime'])
        stress_levels = float(request.form['StressLevels'])
        mood = float(request.form['Mood'])
        social_relationship = float(request.form['SocialRelationship'])

        # Make predictions using the loaded model
        inputs = [[age, gender, screen_time, active_lifestyle, sleep_time, stress_levels, mood, social_relationship]]
        p = model.predict(inputs)[0]
        if p==0:
            prediction='Mental Health Status: Severe '
        if p==1:
            prediction='Mental Health Status: Moderate '
        if p==2:
            prediction='Mental Health Status: Mild '
        if p==3:
            prediction='Mental health status: Good '
        if p==4:
            prediction='Mental health status: Optimal '

        # Redirect to the results page with the prediction
        return redirect(url_for('results', prediction=prediction))

    return render_template('index.html')

@app.route('/results/<prediction>')
def results(prediction):
    return render_template('results.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
