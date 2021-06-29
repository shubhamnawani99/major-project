# importing Flask and other modules

from flask import Flask
from flask_cors import CORS
from flask_restful import Resource, Api
from flask_jsonpify import jsonify
from attention_detection_app_main import run
import json

# Flask constructor
app = Flask(__name__)
api = Api(app)
CORS(app)


@app.route('/start')
def home():
    run()
    return jsonify({'text': 'Application Started'})


class Participants(Resource):
    def get(self):
        with open('attentions.json') as json_file:
            data = json.load(json_file)
        return data

api.add_resource(Participants, '/participants')

# @app.route('/result', methods=["POST"])
# def process():
#     if request.method == "POST":
#         print("ok")
#         items = modules.attention_detection_app.main()
#         print(items)
#         return render_template('output.html')


if __name__ == '__main__':
    # @Url = http://localhost:5000/participants
    app.run(port=5000)
