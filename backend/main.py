from flask import Flask
import os

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
    
    app.register_blueprint(app_bp)
    app.register_blueprint(auth_bp)  # Register auth blueprint
    
    # Create database tables
    with app.app_context():
        # Create directories for vector databases
        os.makedirs("faiss_indexes", exist_ok=True)
        db.create_all()
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(
        host=app.config.get('HOST', '0.0.0.0'),
        port=app.config.get('PORT', 5000),
        debug=app.config.get('DEBUG', True)
    )
