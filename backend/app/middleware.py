from flask import Blueprint, request, jsonify,current_app
from functools import wraps
from sqlalchemy.exc import IntegrityError
from app.services.auth_service import (
     decode_token, 
    get_user_by_id, 
)
from app.extensions import db

#Some Middleware before_request methods are defined in the app context, which is in main.py or wherever is the execution point
def get_and_decode_auth_header():
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        raise ValueError("Missing Authorization header")
        
    parts = auth_header.split(' ')
    if len(parts) != 2 or parts[0].lower() != 'bearer':
        raise ValueError("Invalid Authorization header format")
    token = parts[1]

    try:
        user_id = decode_token(token)
        if not user_id:
            raise ValueError("Invalid or expired token")
        return user_id
    except Exception as e:
        raise ValueError(f"Error decoding token: {e}")

def get_user_ip():
    ip_header = request.headers.get('X-Forwarded-For', None)
    if ip_header:
        ip = ip_header.split(',')[0]
    else:
        ip = request.remote_addr

    return ip

# JWT auth decorator
def jwt_required(function):
    @wraps(function)
    def decorated_function(*args, **kwargs):
        user_id = get_and_decode_auth_header()
        # Add user_id to request context
        request.user_id = user_id
        return function(*args, **kwargs)
    return decorated_function

def admin_required(function):
    @wraps(function)
    def decorated_function(*args, **kwargs):
        user_id = get_and_decode_auth_header()
        user = get_user_by_id(user_id)
        if not user.is_admin:
            return jsonify({"error": "Permission denied, only admins can access this endpoint"}), 409
        return function(*args, **kwargs)
    return decorated_function

