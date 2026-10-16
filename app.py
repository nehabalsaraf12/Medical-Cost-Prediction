from flask import Flask, render_template_string, request
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the trained Linear Regression model
model_path = 'linear_medical_cost_predict.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# HTML Template with modern medical-themed embedded CSS styling and clean layout
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Cost Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #0284c7;
            --primary-hover: #0369a1;
            --bg-gradient: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            --card-bg: #ffffff;
            --text-main: #1e293b;
            --text-muted: #64748b;
            --border: #cbd5e1;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: var(--card-bg);
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 10px 30px -5px rgba(14, 165, 233, 0.15), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
            width: 100%;
            max-width: 600px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .icon-box {
            width: 60px;
            height: 60px;
            background-color: #e0f2fe;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 12px auto;
        }
        h2 {
            margin-bottom: 6px;
            color: var(--primary);
            font-size: 24px;
            font-weight: 700;
        }
        p.subtitle {
            color: var(--text-muted);
            font-size: 14px;
        }
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        .full-width {
            grid-column: span 2;
        }
        label {
            margin-bottom: 6px;
            font-weight: 600;
            font-size: 12px;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        input, select {
            padding: 12px 14px;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 14px;
            background-color: #f8fafc;
            transition: all 0.2s ease;
            outline: none;
        }
        input:focus, select:focus {
            border-color: var(--primary);
            background-color: #fff;
            box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.15);
        }
        button {
            grid-column: span 2;
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 14px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.2s ease, transform 0.1s ease;
            margin-top: 10px;
        }
        button:hover {
            background-color: var(--primary-hover);
        }
        button:active {
            transform: scale(0.98);
        }
        .result-card {
            margin-top: 25px;
            padding: 20px;
            background-color: #e0f2fe;
            border: 1px solid #bae6fd;
            border-radius: 10px;
            text-align: center;
        }
        .result-title {
            color: #0369a1;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 4px;
            letter-spacing: 0.5px;
        }
        .result-value {
            font-size: 26px;
            font-weight: 700;
            color: #0c4a6e;
        }
        @media (max-width: 600px) {
            .form-grid { grid-template-columns: 1fr; }
            .full-width, button { grid-column: span 1; }
            .container { padding: 25px 20px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="icon-box">
                <!-- Medical Cross SVG Icon -->
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#0284c7" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
            </div>
            <h2>Medical Cost Predictor</h2>
            <p class="subtitle">Enter patient demographics to calculate estimated medical insurance expenses.</p>
        </div>
        
        <form method="POST">
            <div class="form-grid">
                <div class="form-group">
                    <label for="age">Age</label>
                    <input type="number" id="age" name="age" placeholder="e.g. 32" required>
                </div>
                
                <div class="form-group">
                    <label for="bmi">BMI</label>
                    <input type="number" step="0.1" id="bmi" name="bmi" placeholder="e.g. 24.5" required>
                </div>
                
                <div class="form-group">
                    <label for="children">Children</label>
                    <input type="number" id="children" name="children" placeholder="e.g. 1" required>
                </div>
                
                <div class="form-group">
                    <label for="smoker">Smoker Status</label>
                    <select id="smoker" name="smoker" required>
                        <option value="no">No</option>
                        <option value="yes">Yes</option>
                    </select>
                </div>
                
                <div class="form-group full-width">
                    <label for="region">Region</label>
                    <select id="region" name="region" required>
                        <option value="northeast">Northeast</option>
                        <option value="northwest">Northwest</option>
                        <option value="southeast">Southeast</option>
                        <option value="southwest">Southwest</option>
                    </select>
                </div>
                
                <button type="submit">Predict Insurance Cost</button>
            </div>
        </form>

        {% if prediction is not none %}
        <div class="result-card">
            <div class="result-title">Estimated Insurance Charges</div>
            <div class="result-value">{{ prediction }}</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Capture inputs from form
            age = float(request.form['age'])
            bmi = float(request.form['bmi'])
            children = float(request.form['children'])
            smoker = request.form['smoker']
            region = request.form['region']

            # Map categorical fields to numeric values
            smoker_val = 1 if smoker == 'yes' else 0
            region_map = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
            region_val = region_map.get(region, 0)

            # Build DataFrame matching the exact features expected by your pickle model
            input_data = pd.DataFrame([[age, bmi, children, smoker_val, region_val]], 
                                      columns=['age', 'bmi', 'children', 'smoker', 'region'])

            # Run prediction
            pred_cost = model.predict(input_data)[0]
            prediction = f"${pred_cost:,.2f}"
            
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
