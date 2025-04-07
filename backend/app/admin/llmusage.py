from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
import threading
import time
import logging
import json
import os
from enum import Enum

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens = self.prompt_tokens + self.completion_tokens

@dataclass
class UsageAlert:
    timestamp: datetime
    level: AlertLevel
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LLMRequestEvent:
    request_id: str
    model: str
    provider: str
    timestamp: datetime
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    status: str
    error: Optional[str] = None
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    cost: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class LLMUsage:
    def __init__(
        self,
        token_quota: Optional[int] = None,
        cost_budget: Optional[float] = None,
        alert_handlers: Optional[Dict[AlertLevel, List[Callable]]] = None,
        quota_warning_threshold: float = 0.8,  # Alert at 80% usage
        storage_path: Optional[str] = None
    ):
        # Configuration
        self.token_quota = token_quota
        self.cost_budget = cost_budget
        self.quota_warning_threshold = quota_warning_threshold
        self.storage_path = storage_path or os.path.join(os.getcwd(), "llm_usage")
        
        # Alert handling
        self.alert_handlers = alert_handlers or {
            AlertLevel.INFO: [],
            AlertLevel.WARNING: [],
            AlertLevel.CRITICAL: []
        }
        
        # Runtime metrics
        self.monthly_usage = {}  # Dict[year-month, TokenUsage]
        self.model_usage = {}    # Dict[model_name, TokenUsage]
        self.provider_usage = {} # Dict[provider, TokenUsage]
        self.request_history = []  # List of LLMRequestEvent
        self.alerts = []         # List of alerts
        
        # Performance metrics
        self.latency_history = []  # List of latency values
        self.error_count = 0
        self.request_count = 0
        
        # Cost tracking
        self.total_cost = 0.0
        self.model_costs = {}    # Dict[model_name, cost]
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize storage
        os.makedirs(self.storage_path, exist_ok=True)
        self._load_history()
        
    def record_request(
        self,
        request_id: str,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        status: str,
        error: Optional[str] = None,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        cost: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a complete LLM request with all metrics
        """
        with self._lock:
            timestamp = datetime.now()
            year_month = timestamp.strftime("%Y-%m")
            
            # Create event
            event = LLMRequestEvent(
                request_id=request_id,
                model=model,
                provider=provider,
                timestamp=timestamp,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                status=status,
                error=error,
                user_id=user_id,
                project_id=project_id,
                cost=cost,
                metadata=metadata or {}
            )
            
            # Update request count
            self.request_count += 1
            
            # Update token usage
            if year_month not in self.monthly_usage:
                self.monthly_usage[year_month] = TokenUsage()
            self.monthly_usage[year_month].add(prompt_tokens, completion_tokens)
            
            if model not in self.model_usage:
                self.model_usage[model] = TokenUsage()
            self.model_usage[model].add(prompt_tokens, completion_tokens)
            
            if provider not in self.provider_usage:
                self.provider_usage[provider] = TokenUsage()
            self.provider_usage[provider].add(prompt_tokens, completion_tokens)
            
            # Update performance metrics
            self.latency_history.append(latency_ms)
            if status != "success" and error:
                self.error_count += 1
            
            # Update cost tracking
            if cost:
                self.total_cost += cost
                if model not in self.model_costs:
                    self.model_costs[model] = 0.0
                self.model_costs[model] += cost
            
            # Store request
            self.request_history.append(event)
            
            # Check quotas and budgets
            self._check_quota_alerts(year_month)
            
            # Persist data
            self._persist_event(event)
    
    def _check_quota_alerts(self, year_month: str) -> None:
        """Check if any quotas or budgets are exceeded and trigger alerts"""
        # Token quota check
        if self.token_quota:
            current_usage = self.monthly_usage[year_month].total_tokens
            usage_percent = current_usage / self.token_quota
            
            if usage_percent >= 1.0:
                self._create_alert(
                    AlertLevel.CRITICAL,
                    f"Token quota exceeded: {current_usage}/{self.token_quota} tokens used ({usage_percent:.1%})"
                )
            elif usage_percent >= self.quota_warning_threshold:
                self._create_alert(
                    AlertLevel.WARNING,
                    f"Token quota warning: {current_usage}/{self.token_quota} tokens used ({usage_percent:.1%})"
                )
        
        # Cost budget check
        if self.cost_budget:
            usage_percent = self.total_cost / self.cost_budget
            
            if usage_percent >= 1.0:
                self._create_alert(
                    AlertLevel.CRITICAL,
                    f"Cost budget exceeded: ${self.total_cost:.2f}/${self.cost_budget:.2f} spent ({usage_percent:.1%})"
                )
            elif usage_percent >= self.quota_warning_threshold:
                self._create_alert(
                    AlertLevel.WARNING,
                    f"Cost budget warning: ${self.total_cost:.2f}/${self.cost_budget:.2f} spent ({usage_percent:.1%})"
                )
    
    def _create_alert(self, level: AlertLevel, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Create and dispatch an alert"""
        alert = UsageAlert(
            timestamp=datetime.now(),
            level=level,
            message=message,
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        
        # Dispatch to handlers
        for handler in self.alert_handlers[level]:
            try:
                handler(alert)
            except Exception as e:
                logging.error(f"Error in alert handler: {str(e)}")
    
    def _persist_event(self, event: LLMRequestEvent) -> None:
        """Persist event to storage"""
        if not self.storage_path:
            return
            
        try:
            # Convert event to dict
            event_dict = {
                "request_id": event.request_id,
                "model": event.model,
                "provider": event.provider,
                "timestamp": event.timestamp.isoformat(),
                "prompt_tokens": event.prompt_tokens,
                "completion_tokens": event.completion_tokens,
                "latency_ms": event.latency_ms,
                "status": event.status,
                "error": event.error,
                "user_id": event.user_id,
                "project_id": event.project_id,
                "cost": event.cost,
                "metadata": event.metadata
            }
            
            # Write to daily log file
            date_str = event.timestamp.strftime("%Y-%m-%d")
            file_path = os.path.join(self.storage_path, f"events_{date_str}.jsonl")
            
            with open(file_path, "a") as f:
                f.write(json.dumps(event_dict) + "\n")
        except Exception as e:
            logging.error(f"Error persisting event: {str(e)}")
    
    def _load_history(self) -> None:
        """Load usage history from storage"""
        if not self.storage_path or not os.path.exists(self.storage_path):
            return
            
        try:
            # Find all event files
            for filename in os.listdir(self.storage_path):
                if filename.startswith("events_") and filename.endswith(".jsonl"):
                    file_path = os.path.join(self.storage_path, filename)
                    
                    with open(file_path, "r") as f:
                        for line in f:
                            try:
                                event_dict = json.loads(line.strip())
                                
                                # Convert dict to event
                                event = LLMRequestEvent(
                                    request_id=event_dict["request_id"],
                                    model=event_dict["model"],
                                    provider=event_dict["provider"],
                                    timestamp=datetime.fromisoformat(event_dict["timestamp"]),
                                    prompt_tokens=event_dict["prompt_tokens"],
                                    completion_tokens=event_dict["completion_tokens"],
                                    latency_ms=event_dict["latency_ms"],
                                    status=event_dict["status"],
                                    error=event_dict["error"],
                                    user_id=event_dict["user_id"],
                                    project_id=event_dict["project_id"],
                                    cost=event_dict["cost"],
                                    metadata=event_dict["metadata"]
                                )
                                
                                # Update metrics (but don't trigger alerts)
                                self._update_metrics_from_event(event)
                            except Exception as e:
                                logging.error(f"Error parsing event: {str(e)}")
        except Exception as e:
            logging.error(f"Error loading history: {str(e)}")
    
    def _update_metrics_from_event(self, event: LLMRequestEvent) -> None:
        """Update metrics from a loaded event without triggering alerts"""
        year_month = event.timestamp.strftime("%Y-%m")
        
        # Update request count
        self.request_count += 1
        
        # Update token usage
        if year_month not in self.monthly_usage:
            self.monthly_usage[year_month] = TokenUsage()
        self.monthly_usage[year_month].add(event.prompt_tokens, event.completion_tokens)
        
        if event.model not in self.model_usage:
            self.model_usage[event.model] = TokenUsage()
        self.model_usage[event.model].add(event.prompt_tokens, event.completion_tokens)
        
        if event.provider not in self.provider_usage:
            self.provider_usage[event.provider] = TokenUsage()
        self.provider_usage[event.provider].add(event.prompt_tokens, event.completion_tokens)
        
        # Update performance metrics
        self.latency_history.append(event.latency_ms)
        if event.status != "success" and event.error:
            self.error_count += 1
        
        # Update cost tracking
        if event.cost:
            self.total_cost += event.cost
            if event.model not in self.model_costs:
                self.model_costs[event.model] = 0.0
            self.model_costs[event.model] += event.cost
        
        # Store request
        self.request_history.append(event)
    
    # Dashboard metrics retrieval methods
    
    def get_token_usage_summary(self) -> Dict[str, Any]:
        """Get token usage summary for current month"""
        current_month = datetime.now().strftime("%Y-%m")
        current_usage = self.monthly_usage.get(current_month, TokenUsage())
        
        remaining = None
        if self.token_quota:
            remaining = max(0, self.token_quota - current_usage.total_tokens)
        
        return {
            "current_month": current_month,
            "prompt_tokens": current_usage.prompt_tokens,
            "completion_tokens": current_usage.completion_tokens,
            "total_tokens": current_usage.total_tokens,
            "quota": self.token_quota,
            "remaining": remaining,
            "usage_percent": (current_usage.total_tokens / self.token_quota * 100) if self.token_quota else None
        }
    
    def get_usage_trends(self, days: int = 30) -> Dict[str, List]:
        """Get usage trends for the specified number of days"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        dates = []
        token_counts = []
        request_counts = []
        error_rates = []
        avg_latencies = []
        costs = []
        
        current = start_date
        while current <= end_date:
            date_str = current.strftime("%Y-%m-%d")
            dates.append(date_str)
            
            # Filter events for this day
            day_events = [e for e in self.request_history if e.timestamp.date() == current.date()]
            
            # Token counts
            day_tokens = sum(e.prompt_tokens + e.completion_tokens for e in day_events)
            token_counts.append(day_tokens)
            
            # Request counts
            request_counts.append(len(day_events))
            
            # Error rates
            day_errors = len([e for e in day_events if e.status != "success"])
            error_rate = day_errors / len(day_events) if day_events else 0
            error_rates.append(error_rate * 100)  # As percentage
            
            # Average latency
            day_latencies = [e.latency_ms for e in day_events]
            avg_latency = sum(day_latencies) / len(day_latencies) if day_latencies else 0
            avg_latencies.append(avg_latency)
            
            # Costs
            day_cost = sum(e.cost for e in day_events if e.cost is not None)
            costs.append(day_cost)
            
            current += timedelta(days=1)
        
        return {
            "dates": dates,
            "token_counts": token_counts,
            "request_counts": request_counts,
            "error_rates": error_rates,
            "avg_latencies": avg_latencies,
            "costs": costs
        }
    
    def get_model_breakdown(self) -> Dict[str, List]:
        """Get usage breakdown by model"""
        models = list(self.model_usage.keys())
        prompt_tokens = [self.model_usage[m].prompt_tokens for m in models]
        completion_tokens = [self.model_usage[m].completion_tokens for m in models]
        total_tokens = [self.model_usage[m].total_tokens for m in models]
        costs = [self.model_costs.get(m, 0) for m in models]
        
        return {
            "models": models,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "costs": costs
        }
    
    def get_provider_breakdown(self) -> Dict[str, List]:
        """Get usage breakdown by provider"""
        providers = list(self.provider_usage.keys())
        prompt_tokens = [self.provider_usage[p].prompt_tokens for p in providers]
        completion_tokens = [self.provider_usage[p].completion_tokens for p in providers]
        total_tokens = [self.provider_usage[p].total_tokens for p in providers]
        
        return {
            "providers": providers,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        latencies = self.latency_history[-1000:] if len(self.latency_history) > 1000 else self.latency_history
        
        # Skip calculations if no data
        if not latencies:
            return {
                "avg_latency": 0,
                "p50_latency": 0,
                "p95_latency": 0,
                "p99_latency": 0,
                "error_rate": 0,
                "total_requests": 0,
                "success_rate": 100
            }
        
        # Sort for percentiles
        sorted_latencies = sorted(latencies)
        
        return {
            "avg_latency": sum(latencies) / len(latencies),
            "p50_latency": sorted_latencies[len(sorted_latencies) // 2],
            "p95_latency": sorted_latencies[int(len(sorted_latencies) * 0.95)],
            "p99_latency": sorted_latencies[int(len(sorted_latencies) * 0.99)],
            "error_rate": (self.error_count / self.request_count * 100) if self.request_count else 0,
            "total_requests": self.request_count,
            "success_rate": ((self.request_count - self.error_count) / self.request_count * 100) if self.request_count else 100
        }
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        recent = sorted(self.alerts, key=lambda a: a.timestamp, reverse=True)[:limit]
        
        return [{
            "timestamp": a.timestamp.isoformat(),
            "level": a.level.value,
            "message": a.message,
            "metadata": a.metadata
        } for a in recent]
