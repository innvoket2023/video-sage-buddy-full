from flask import Blueprint, request, jsonify
from app.services.auth_service import (
    generate_token, decode_token, verify_password, 
    create_user, get_user_by_id, get_user_by_identifier, 
    activate_user, update_user_login_timestamp, is_suspended
)
from app.extensions import db
from app.middleware import jwt_required
from functools import wraps
import re
from sqlalchemy.exc import IntegrityError

# Create the blueprint
auth_bp = Blueprint('auth', __name__, url_prefix='/api')

@auth_bp.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    # Validate required fields
    if not all(k in data for k in ('username', 'email', 'password')):
        return jsonify({'error': 'Missing required fields'}), 400
    
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    # Validate email format
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    if not email_pattern.match(email):
        return jsonify({'error': 'Invalid email format'}), 400
    
    # Validate password strength
    if len(password) < 8:
        return jsonify({'error': 'Password must be at least 8 characters long'}), 400
    
    try:
        # Create a new user
        new_user = create_user(username, email, password)
        
        return jsonify({
            'message': 'Registration successful! Please check your email to activate your account.',
            'user': {
                'id': str(new_user.user_id),
                'username': new_user.username,
                'email': new_user.email
            }
        }), 201

    except IntegrityError as e:
        db.session.rollback()
        # Check what kind of integrity error
        if 'username' in str(e.orig):
            return jsonify({'error': 'Username already exists'}), 409
        elif 'email' in str(e.orig):
            return jsonify({'error': 'Email already exists'}), 409
        else:
            return jsonify({'error': 'Registration failed due to database error'}), 500
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    
    # Check if login is via username or email
    identifier = data.get('username', '') or data.get('email', '')
    password = data.get('password', '')
    
    if not identifier or not password:
        return jsonify({'error': 'Username/email and password are required'}), 400
    
    # Find the user
    try:
        user = get_user_by_identifier(identifier)

        if is_suspended(user.suspended_till):
            return jsonify({'error': "An unexpected issue occurred. Please contact our support team for assistance."})
        
        if not user:
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Check password
        if not verify_password(user.password, password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Check if account is active
        if not user.is_activated:
            return jsonify({'error': 'Account not activated. Please check your email.'}), 403
        
        # Update last_login timestamp
        update_user_login_timestamp(user)
        
        # Generate JWT token
        token = generate_token(user.user_id)
        
        return jsonify({
            'message': 'Login successful',
            'user': user.to_dict(),
            'token': str(token)
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Login failed: {str(e)}'}), 500

@auth_bp.route('/logout', methods=['POST'])
def logout():
    # JWT is stateless, so no server-side logout is needed
    # Client should discard the token
    return jsonify({'message': 'Logged out successfully'}), 200

@auth_bp.route('/activate/<uuid:user_id>', methods=['GET'])
def activate_account(user_id):
    user = get_user_by_id(user_id)
    
    if not user:
        return jsonify({'error': 'Invalid activation link'}), 404
    
    if user.is_activated:
        return jsonify({'message': 'Account already activated'}), 200
    
    if activate_user(user_id):
        return jsonify({'message': 'Account activated successfully! You can now log in.'}), 200
    else:
        return jsonify({'error': 'Activation failed'}), 500

@auth_bp.route('/reset-password', methods=['POST'])
def request_password_reset():
    data = request.get_json()
    email = data.get('email')
    
    if not email:
        return jsonify({'error': 'Email is required'}), 400
    
    # Don't reveal if email exists for security
    return jsonify({'message': 'If your email exists in our system, you will receive a password reset link'}), 200

@auth_bp.route('/get_email_uname', methods=['GET'])
@jwt_required
def get_email_uname():
    user_id = request.user_id
    user = get_user_by_id(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"username": user.username, "email": user.email})

@auth_bp.route('/reset-email-uname', methods=['PUT'])
@jwt_required
def update_user():
    user_id = request.user_id
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
        
    username = data.get('username')
    email = data.get('email')
    
    try:
        user = get_user_by_id(user_id)
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        if username:
            user.username = username
        if email:
            user.email = email
        
        db.session.commit()
        
        return jsonify({
            "message": "User updated successfully",
            "user": user.to_dict()
        })
    
    except IntegrityError:
        db.session.rollback()
        return jsonify({"error": "Username or email already exists"}), 409
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@auth_bp.route('/assign/admin/<uuid:user_id>', methods=['PUT'])
def assign_admin_status(user_id):
    user = get_user_by_id(user_id)
    print(user)
    print(user.is_admin)
    if user.is_admin:
        return jsonify({"message": "This user is already an Admin"}), 409
    else:
        user.is_admin = True
        db.session.add(user)
        db.session.commit()
        return jsonify({"message": "Assigned admin status to this user"}), 200
