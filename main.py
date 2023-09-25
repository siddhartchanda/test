import pandas as pd
from flask import Flask, render_template, request, jsonify
from predictor import make_prediction

app = Flask(__name__, template_folder='.')
data = pd.read_csv('TRAVEL.csv')

@app.route('/')
def index():
    agency_list = sorted(data['Agency'].unique())
    agency_type_list = sorted(data['Agency Type'].unique())
    distribution_channels = sorted(data['Distribution Channel'].unique())
    product_names = sorted(data['Product Name'].unique())
    d_names = sorted(data['Destination'].unique())
    return render_template('index.html', agencies=agency_list, agency_type=agency_type_list, ch=distribution_channels, pn=product_names, gender=['M', 'F', None], d=d_names)

@app.route('/predict', methods=['POST'])
def predict():
    l = []
    Agency = request.form.get('agency')
    Agency_Type = request.form.get('t')
    dis = request.form.get('s')
    pn = request.form.get('product')
    d = request.form.get('duration')
    des = request.form.get('destination')
    g = request.form.get('gender')
    a = request.form.get('age')
    input_data = {
        'Agency': Agency,
        'Agency Type': Agency_Type,
        'Distribution Channel': dis,
        'Product Name': pn,
        'Duration': int(d),  # Example duration in days
        'Destination': des,  # Example destination
        'Gender': g,  # You can specify gender if it's known, or use None
        'Age': int(a)  # Example age
    }
    result = make_prediction(input_data)
    return jsonify({'predicted_claim': result})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
