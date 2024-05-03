# app.py

from flask import Flask, jsonify, request
from flask_cors import CORS
from decisiontree import DecisionTree  # Import your DecisionTree class
from pymongo import MongoClient
from bson.json_util import dumps
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["finance"]
transactions = db["transactions"]

# Load the trained DecisionTree model
model = DecisionTree()

# Load dataset for preprocessing
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status', 'occupation',
             'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
             'income']
df = pd.read_csv("income.csv", names=col_names, skiprows=1)

# Data preprocessing
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country',
                    'gender']
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])


@app.route('/predict', methods=['POST'])
def predict_income():
    # Receive input data from frontend
    data = request.json

    # Preprocess input data
    input_data = pd.DataFrame(data, index=[0])
    for col in categorical_cols:
        input_data[col] = label_encoders[col].transform(input_data[col])

    # Make predictions using the model
    prediction = model.predict(input_data)

    # Return the prediction as JSON response
    return jsonify({'prediction': prediction})


@app.route('/transactions', methods=['GET'])
def get_transactions():
    # Retrieve transactions from MongoDB
    result = transactions.find()

    # Convert MongoDB cursor to JSON and return
    return dumps(result)


if __name__ == '__main__':
    app.run(debug=True)
