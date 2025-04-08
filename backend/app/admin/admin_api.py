from flask import Blueprint, jsonify, request, current_app
from app.admin.llmusage import LLMUsage
from app.auth_routes import jwt_required
from flask import current_app

# Create the blueprint
admin = Blueprint('admin', __name__, url_prefix='/admin')

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
@admin.route('/usage/summary', methods=['GET'])
@jwt_required
def get_usage_summary():
    user_id = request.user_id  # From JWT middleware
    # Access control could be added here based on user_id
    usage_tracker = get_usage_tracker()
    return jsonify(usage_tracker.get_token_usage_summary())

# Time-based Analytics APIs
@admin.route('/usage/trends', methods=['GET'])
@jwt_required
def get_usage_trends():
    user_id = request.user_id
    days = request.args.get('days', default=30, type=int)
    usage_tracker = get_usage_tracker()
    return jsonify(usage_tracker.get_usage_trends(days))

# Model & Provider Analytics APIs
@admin.route('/usage/models', methods=['GET'])
@jwt_required
def get_model_breakdown():
    user_id = request.user_id
    usage_tracker = get_usage_tracker()
    return jsonify(usage_tracker.get_model_breakdown())

@admin.route('/usage/providers', methods=['GET']) 
@jwt_required
def get_provider_breakdown():
    user_id = request.user_id
    usage_tracker = get_usage_tracker()
    return jsonify(usage_tracker.get_provider_breakdown())

# Performance Metrics API
@admin.route('/metrics/performance', methods=['GET'])
@jwt_required
def get_performance_metrics():
    user_id = request.user_id
    usage_tracker = get_usage_tracker()
    return jsonify(usage_tracker.get_performance_metrics())

# Alerts API
@admin.route('/alerts/recent', methods=['GET'])
@jwt_required
def get_alerts():
    user_id = request.user_id
    limit = request.args.get('limit', default=10, type=int)
    usage_tracker = get_usage_tracker()
    return jsonify(usage_tracker.get_recent_alerts(limit))

# Cost Management API
@admin.route('/costs/summary', methods=['GET'])
@jwt_required
def get_cost_summary():
    user_id = request.user_id
    usage_tracker = get_usage_tracker()
    return jsonify({
        "total_cost": usage_tracker.total_cost,
        "budget": usage_tracker.cost_budget,
        "remaining": usage_tracker.cost_budget - usage_tracker.total_cost if usage_tracker.cost_budget else None,
        "by_model": usage_tracker.model_costs
    })
