"""
Simple api to serve predictions.
"""
from flask import Flask, jsonify
from flask_restful import reqparse, abort, Api, Resource
import json
import logging

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

def setup_arg_parsing():
    args_str = [
        'job', 'marital', 'education',
        'default', 'housing', 'loan', 'contact',
        'month', 'day_of_week', 'duration',
        'poutcome', 'sample_uuid'
    ]
    args_int = [
        'age', 'campaign', 'pdays', 'previous'
    ]
    args_float = [
        'emp.var.rate', 'cons.price.idx',
        'cons.conf.idx', 'euribor3m', 'nr.employed'
    ]

    parser = reqparse.RequestParser()
    for argument in args_str:
        parser.add_argument(argument)
    for argument in args_int:
        parser.add_argument(argument, type=int)
    for argument in args_float:
        parser.add_argument(argument, type=float)

    logging.debug('Set up argument parser.')

    return parser

predict_arg_parser = setup_arg_parsing()

class SimpleModel(Resource):
    """
    The resource we want to expose
    """

    def get(self):
        args = predict_arg_parser.parse_args()

        features = ['age', 'job']

        response = {
            'sample_uuid': args['sample_uuid'],
            'probability':0.5,
            'label':1.0
        }
        return jsonify(response)


api.add_resource(SimpleModel, '/api/v1/predict')

if __name__ == '__main__':
     app.run(host="0.0.0.0", port=5000, debug=True)
