print("RUNNING HealthLens Project")
from flask import Flask, request, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Doctor suggestion dictionary
doctor_map = {
    "Flu": "General Physician",
    "Cold": "General Physician",
    "Allergy": "Allergist",
    "Dengue": "Infectious Disease Specialist",
    "Migraine": "Neurologist"
}

html = """
<!DOCTYPE html>
<html>
<head>
<title>HealthLens Predictor</title>
<style>
body {
    background: linear-gradient(to right, #4facfe, #00f2fe);
    font-family: Arial;
    text-align: center;
    padding-top: 50px;
}
.container {
    background: white;
    width: 400px;
    margin: auto;
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.2);
}
input {
    padding: 8px;
    margin: 5px;
}
button {
    padding: 10px 20px;
    background: #4facfe;
    border: none;
    color: white;
    border-radius: 5px;
    cursor: pointer;
}
</style>
</head>
<body>

<div class="container">
<h2>🧠 Smart Disease Prediction System</h2>

<form method="post">
Fever (0/1): <input name="fever"><br>
Cough (0/1): <input name="cough"><br>
Headache (0/1): <input name="headache"><br>
Fatigue (0/1): <input name="fatigue"><br><br>
<button type="submit">Predict</button>
</form>

{% if result %}
<hr>
<h3>Disease: {{ result }}</h3>
<h4>Confidence: {{ confidence }}%</h4>
<h4>Recommended Doctor: {{ doctor }}</h4>
{% endif %}
</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        data = [
            int(request.form["fever"]),
            int(request.form["cough"]),
            int(request.form["headache"]),
            int(request.form["fatigue"])
        ]

        prediction = model.predict([data])[0]
        probability = max(model.predict_proba([data])[0]) * 100

        doctor = doctor_map.get(prediction, "Consult General Physician")

        return render_template_string(
            html,
            result=prediction,
            confidence=round(probability, 2),
            doctor=doctor
        )

    return render_template_string(html)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
