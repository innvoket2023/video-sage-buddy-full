from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from flask_cors import CORS
# Render Template --> Flask instance --> wrapper of flask using flask socket io --> initialization wrapper --> 
application = Flask(__name__)
application.config['SECRET_KEY'] = 'your_secret_key!'
# socketio = SocketIO(application, cors_allowed_origins=[
#     "http://localhost:8080",
#     "http://localhost:3000",
#     "http://localhost:5000"
#     # Add any other origins your frontend might run on
# ])
# CORS(application, 
#     resources={r"/*": {
#         "origins": ["http://localhost:8080", "http://localhost:3000", "http://localhost:5000"],
#         "supports_credentials": True,
#         "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"]
#     }})

socketio = SocketIO(application)

@application.route('/', methods = ["GET"])
def index():
    return render_template('index.html')

@socketio.on('client_event')
def client_event(responsee):
    print("hahahahahahahha")

@socketio.on('blah')
def message(responsee):
    responsee["data"] = "Yo wassup"
    emit("server_response", responsee)

if  __name__ == "__main__":
    socketio.run(application, debug = True, use_reloader=True, log_output=True)
