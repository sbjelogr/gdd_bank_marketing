"""
Simple api to serve predictions.
"""
from flask import Flask, jsonify
from flask_restful import reqparse, abort, Api, Resource
import json
import logging
import joblib
import pandas as pd

logging.basicConfig(filename='predict.log', level=logging.DEBUG)

app = Flask(__name__)
api = Api(app)

# A training process has stored predictions into a
# json file
with open('simple.json') as f:
    PREDICTIONS = json.load(f)
    print(PREDICTIONS)

def abort_if_prediction_doesnt_exist(sample_uuid):
    if sample_uuid not in PREDICTIONS:
        abort(404, message="User {} doesn't exist".format(sample_uuid))

def get_model():
    model = joblib.load('model/rf_model_5.pkl')
    return model

args_str = [
    'job', 'marital', 'education',
    'default', 'housing', 'loan', 'contact',
    'month', 'day_of_week', 'poutcome'
]
args_int = [
    'age', 'campaign', 'pdays', 'previous'
]
args_float = [
    'emp.var.rate', 'cons.price.idx',
    'cons.conf.idx', 'euribor3m', 'nr.employed'
]

label_encoder_dict = {
     'contact': {'cellular': 0, 'telephone': 1},
     'day_of_week': {'fri': 4, 'mon': 1, 'thu': 0, 'tue': 3, 'wed': 2},
     'default': {'no': 0, 'unknown': 1, 'yes': 2},
     'education': {'basic.4y': 4,
      'basic.6y': 5,
      'basic.9y': 2,
      'high.school': 1,
      'illiterate': 7,
      'professional.course': 3,
      'university.degree': 0,
      'unknown': 6},
     'housing': {'no': 1, 'unknown': 2, 'yes': 0},
     'job': {'admin.': 0,
      'blue-collar': 1,
      'entrepreneur': 6,
      'housemaid': 8,
      'management': 4,
      'retired': 5,
      'self-employed': 7,
      'services': 3,
      'student': 10,
      'technician': 2,
      'unemployed': 9,
      'unknown': 11},
     'loan': {'no': 0, 'unknown': 2, 'yes': 1},
     'marital': {'divorced': 2, 'married': 0, 'single': 1, 'unknown': 3},
     'month': {'apr': 5,
      'aug': 2,
      'dec': 9,
      'jul': 1,
      'jun': 3,
      'mar': 8,
      'may': 0,
      'nov': 4,
      'oct': 6,
      'sep': 7},
     'poutcome': {'failure': 1, 'nonexistent': 0, 'success': 2}
 }


def setup_arg_parsing():

    parser = reqparse.RequestParser()

    parser.add_argument('sample_uuid')

    for argument in args_str:
        parser.add_argument(argument)
    for argument in args_int:
        parser.add_argument(argument, type=int)
    for argument in args_float:
        parser.add_argument(argument, type=float)

    logging.debug('Set up argument parser.')

    return parser

def parse_dict(args_dict, features):
    df = pd.DataFrame([args_dict])
    return df[features]

predict_arg_parser = setup_arg_parsing()
model = get_model()

class SimpleModel(Resource):
    """
    The resource we want to expose
    """

    def get(self):
        args = predict_arg_parser.parse_args()

        logging.info("\n Request: \n {}".format(args))

        features = args_str + args_int + args_float
        X = parse_dict(args,features)

        for feat in args_str:
            X[feat] = X[feat].map(
                lambda old_val: label_encoder_dict[feat].get(old_val, 0))

        X.fillna(0, inplace=True)

        proba = float(model.predict_proba(X)[0,1])
        label = 0.
        if proba > 0.5:
            label = 1.

        response = {
            'sample_uuid': args['sample_uuid'],
            'probability': proba,
            'label': label
        }

        logging.info("Response: \n {}".format(response))

        return jsonify(response)


api.add_resource(SimpleModel, '/api/v1/predict')

if __name__ == '__main__':
     app.run(host="0.0.0.0", port=5000, debug=True)
