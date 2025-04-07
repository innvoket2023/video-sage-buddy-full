# # Token Usage APIs
# @app.route('/api/usage/summary', methods=['GET'])
# def get_usage_summary():
#     return jsonify(usage_tracker.get_token_usage_summary())
#
# # Time-based Analytics APIs
# @app.route('/api/usage/trends', methods=['GET'])
# def get_usage_trends():
#     days = request.args.get('days', default=30, type=int)
#     return jsonify(usage_tracker.get_usage_trends(days))
#
# # Model & Provider Analytics APIs
# @app.route('/api/usage/models', methods=['GET'])
# def get_model_breakdown():
#     return jsonify(usage_tracker.get_model_breakdown())
#
# @app.route('/api/usage/providers', methods=['GET']) 
# def get_provider_breakdown():
#     return jsonify(usage_tracker.get_provider_breakdown())
#
# # Performance Metrics API
# @app.route('/api/metrics/performance', methods=['GET'])
# def get_performance_metrics():
#     return jsonify(usage_tracker.get_performance_metrics())
#
# # Alerts API
# @app.route('/api/alerts/recent', methods=['GET'])
# def get_alerts():
#     limit = request.args.get('limit', default=10, type=int)
#     return jsonify(usage_tracker.get_recent_alerts(limit))
#
# # Cost Management API
# @app.route('/api/costs/summary', methods=['GET'])
# def get_cost_summary():
#     return jsonify({
#         "total_cost": usage_tracker.total_cost,
#         "budget": usage_tracker.cost_budget,
#         "remaining": usage_tracker.cost_budget - usage_tracker.total_cost if usage_tracker.cost_budget else None,
#         "by_model": usage_tracker.model_costs
#     })
