from flask import Blueprint, jsonify, request, current_app
from app.admin.llmusage import LLMUsage
from app.middleware import jwt_required, admin_required
from app.services.cloudinary import cloudinary_usage
from app.services.elevenlabsIO import elevenlabs_usage
from app.extensions import db
from app.models import User, Video
from sqlalchemy import desc
from datetime import datetime, timedelta

# Create the blueprint
admin = Blueprint('admin', __name__, url_prefix='/admin')

#API management

# Get the usage tracker from app context
def get_usage_tracker():
    # If usage_tracker doesn't exist yet in app context, initialize it
    if not hasattr(current_app, 'usage_tracker'):
        # Initialize with app configuration
        current_app.usage_tracker = LLMUsage(
            token_quota=current_app.config.get('LLM_TOKEN_QUOTA'),
            cost_budget=current_app.config.get('LLM_COST_BUDGET'),
            storage_path=current_app.config.get('LLM_USAGE_STORAGE_PATH')
        )
    return current_app.usage_tracker

# Token Usage APIs
@admin.route('/llm/usage/summary', methods=['GET'])
@jwt_required
@admin_required
def get_usage_summary():
    user_id = request.user_id  # From JWT middleware
    # Access control could be added here based on user_id
    usage_tracker = get_usage_tracker()
    return jsonify(usage_tracker.get_token_usage_summary())

# Time-based Analytics APIs
@admin.route('/llm/usage/trends', methods=['GET'])
@jwt_required
@admin_required
def get_usage_trends():
    user_id = request.user_id
    days = request.args.get('days', default=30, type=int)
    usage_tracker = get_usage_tracker()
    return jsonify(usage_tracker.get_usage_trends(days))

# Model & Provider Analytics APIs
@admin.route('/llm/usage/models', methods=['GET'])
@jwt_required
@admin_required
def get_model_breakdown():
    user_id = request.user_id
    usage_tracker = get_usage_tracker()
    return jsonify(usage_tracker.get_model_breakdown())

@admin.route('/llm/usage/providers', methods=['GET']) 
@jwt_required
@admin_required
def get_provider_breakdown():
    user_id = request.user_id
    usage_tracker = get_usage_tracker()
    return jsonify(usage_tracker.get_provider_breakdown())

# Performance Metrics API
@admin.route('/llm/metrics/performance', methods=['GET'])
@jwt_required
@admin_required
def get_performance_metrics():
    user_id = request.user_id
    usage_tracker = get_usage_tracker()
    return jsonify(usage_tracker.get_performance_metrics())

# Alerts API
@admin.route('/llm/alerts/recent', methods=['GET'])
@jwt_required
@admin_required
def get_alerts():
    user_id = request.user_id
    limit = request.args.get('limit', default=10, type=int)
    usage_tracker = get_usage_tracker()
    return jsonify(usage_tracker.get_recent_alerts(limit))

# Cost Management API
@admin.route('/llm/costs/summary', methods=['GET'])
@jwt_required
@admin_required
def get_cost_summary():
    user_id = request.user_id
    usage_tracker = get_usage_tracker()
    return jsonify({
        "total_cost": usage_tracker.total_cost,
        "budget": usage_tracker.cost_budget,
        "remaining": usage_tracker.cost_budget - usage_tracker.total_cost if usage_tracker.cost_budget else None,
        "by_model": usage_tracker.model_costs
    })

@admin.route('/cloudinary/usage/summary', methods=['GET'])
@jwt_required
@admin_required
def get_cloudinary_usage_summary():
    usage = cloudinary_usage()
    return jsonify(usage)

@admin.route('/elevenlabs/usage/summary', methods=['GET'])
@jwt_required
@admin_required
def get_elevenlabs_usage_summary():
    usage = elevenlabs_usage()
    summary = {
    "characters_used": usage.character_count,
    "characters_limit": usage.character_limit,
    "voice_slot_used": usage.voice_slots_used,
    "voice_slot_limit": usage.voice_limit,
    "remaining_characters": usage.character_limit - usage.character_count,
    "remaining_voices": usage.voice_limit - usage.voice_slots_used
    }
    return jsonify(summary)

#USER Management
@admin.route('/get/users', methods=['GET'])
@jwt_required
@admin_required
def list_all_users():
    users_batch = User.query.order_by(
        User.last_seen.is_(None).asc(),  # asc => NULL values last
        User.last_seen.desc()            # desc => newer times first
    ).all()
    users_batch_list = [] # Iterator is preferred cuz there can be many users and there is enough storage to store in python lists
    for user in users_batch:
        total_videos = Video.query.filter_by(user_id = user.user_id).count()
        user_info = {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "ip": user.ip,
            "last_login": user.last_login,
            "last_seen": user.last_seen,
            "suspended_till": user.suspended_till,
            "is_activated": user.is_activated,
            "is_admin": user.is_admin,
            "total_videos": total_videos
        }
        users_batch_list.append(user_info)
    return jsonify({"all_users": users_batch_list})

@admin.route('/suspend/<uuid:user_id>', methods=['PATCH'])
@jwt_required
@admin_required
def suspend_user(user_id):
    data = request.get_json()
    seconds = data.get("seconds", 0)
    minutes = data.get("minutes", 0)
    hours = data.get("hours", 0)
    days = data.get("days", 0)

    if seconds == 0 and minutes == 0 and hours == 0 and days == 0:
        return jsonify({"error": "Please specify suspension timeframe"})
    
    user = User.query.filter_by(user_id = user_id).one()
    delta = timedelta(days = days, hours = hours, minutes = minutes, seconds = seconds)
    current_time = datetime.now()
    unsuspend_at = current_time + delta
    user.suspended_till = unsuspend_at
    try:
        db.session.add(user)
        db.session.commit()
        return jsonify({"message": f"Suspended till {unsuspend_at}"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"{e}"})
        
