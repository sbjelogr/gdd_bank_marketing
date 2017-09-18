"""
Simple api to serve predictions.
"""
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import json

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

class SimpleModel(Resource):
    """
    The resource we want to expose
    """
    def get(self, sample_uuid):
        abort_if_prediction_doesnt_exist(sample_uuid)
        return PREDICTIONS[sample_uuid]


api.add_resource(SimpleModel, '/predict/<sample_uuid>')

if __name__ == '__main__':
     app.run(host="0.0.0.0", port=5000, debug=True)
