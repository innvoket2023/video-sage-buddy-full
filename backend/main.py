from flask import Flask
from logging.config import dictConfig
import os
from app.admin.llmusage import LLMUsage

def create_app(config_name="development"):
    # Create the Flask app
    app = Flask(__name__)
    
    # Load configuration
    from config import config_dict
    app.config.from_object(config_dict[config_name])
    
    # Initialize extensions
    from app.extensions import db, bcrypt
    db.init_app(app)
    bcrypt.init_app(app)
    
    # Setup CORS
    from flask_cors import CORS
    CORS(app, 
        resources={r"/*": {
            "origins": ["http://localhost:8080", "http://localhost:3000", "http://localhost:5000"],
            "supports_credentials": True,
            "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"]
        }})
    
    from app.routes import app_bp
    from app.auth_routes import auth_bp  # Import the auth blueprint
    from app.admin.admin_api import admin
    
    app.register_blueprint(app_bp)
    app.register_blueprint(auth_bp)  # Register auth blueprint
    app.register_blueprint(admin)
    
    # Create database tables
    with app.app_context():
        # Create directories for vector databases
        os.makedirs("faiss_indexes", exist_ok=True)
        db.create_all()
        # Create directories for LLM usage logs
        os.makedirs(app.config.get('LLM_USAGE_STORAGE_PATH'), exist_ok=True)

    # Below is the log config
    dictConfig({
    'version': 1,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
            'formatter': 'default'
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['console']
    }
})

    #usage_tracking using LLMUsage isntance
    from app.admin.llmusage import LLMUsage
    app.usage_tracker = LLMUsage(
        token_quota=app.config.get('LLM_TOKEN_QUOTA'),
        cost_budget=app.config.get('LLM_COST_BUDGET'),
        storage_path=app.config.get('LLM_USAGE_STORAGE_PATH')
    )
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(
        host=app.config.get('HOST', '0.0.0.0'),
        port=app.config.get('PORT', 5000),
        debug=app.config.get('DEBUG', True)
    )
