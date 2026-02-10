"""Predictive task module - analyzes user behavior patterns and predicts needs."""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from loguru import logger


class PatternType(Enum):
    """Types of behavior patterns."""
    TIME_BASED = "time_based"      # Regular time patterns (e.g., Monday mornings)
    EVENT_TRIGGERED = "event"      # Triggered by specific events
    PERIODIC = "periodic"          # Periodic tasks (daily/weekly)
    SEQUENTIAL = "sequential"      # Action sequences


@dataclass
class UserAction:
    """Represents a single user action."""
    action_type: str               # e.g., "query_weekly_report", "check_email"
    timestamp: datetime
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BehaviorPattern:
    """Discovered behavior pattern."""
    pattern_id: str
    pattern_type: PatternType
    action_type: str
    confidence: float              # 0.0 - 1.0
    frequency: int                 # How many times observed
    typical_times: list[dict]      # For time-based patterns
    triggers: list[str]            # For event-triggered patterns
    period_hours: Optional[float]     # For periodic patterns
    next_predicted: Optional[datetime]
    created_at: datetime
    updated_at: datetime


@dataclass
class Prediction:
    """A single prediction."""
    prediction_id: str
    pattern_id: str
    predicted_action: str
    confidence: float
    reason: str
    suggested_preparation: list[str]
    predicted_time: datetime
    created_at: datetime


class Predictor:
    """
    Predictive task module that analyzes user behavior patterns
    and predicts future needs.
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory_dir = workspace / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.data_file = self.memory_dir / "predictor_data.json"
        self.predictions_file = self.memory_dir / "PREDICTIONS.md"

        self.actions: list[UserAction] = []
        self.patterns: dict[str, BehaviorPattern] = {}
        self.predictions: list[Prediction] = []

        self._load_data()

    def _load_data(self) -> None:
        """Load historical data from disk."""
        if not self.data_file.exists():
            return

        try:
            data = json.loads(self.data_file.read_text())

            # Load actions
            for a in data.get("actions", []):
                self.actions.append(UserAction(
                    action_type=a["action_type"],
                    timestamp=datetime.fromisoformat(a["timestamp"]),
                    context=a.get("context", {}),
                    metadata=a.get("metadata", {}),
                ))

            # Load patterns
            for p in data.get("patterns", []):
                pattern = BehaviorPattern(
                    pattern_id=p["pattern_id"],
                    pattern_type=PatternType(p["pattern_type"]),
                    action_type=p["action_type"],
                    confidence=p["confidence"],
                    frequency=p["frequency"],
                    typical_times=p.get("typical_times", []),
                    triggers=p.get("triggers", []),
                    period_hours=p.get("period_hours"),
                    next_predicted=datetime.fromisoformat(p["next_predicted"]) if p.get("next_predicted") else None,
                    created_at=datetime.fromisoformat(p["created_at"]),
                    updated_at=datetime.fromisoformat(p["updated_at"]),
                )
                self.patterns[pattern.pattern_id] = pattern

            # Load predictions
            for pred in data.get("predictions", []):
                self.predictions.append(Prediction(
                    prediction_id=pred["prediction_id"],
                    pattern_id=pred["pattern_id"],
                    predicted_action=pred["predicted_action"],
                    confidence=pred["confidence"],
                    reason=pred["reason"],
                    suggested_preparation=pred.get("suggested_preparation", []),
                    predicted_time=datetime.fromisoformat(pred["predicted_time"]),
                    created_at=datetime.fromisoformat(pred["created_at"]),
                ))

            logger.info(f"Predictor: loaded {len(self.actions)} actions, {len(self.patterns)} patterns")
        except Exception as e:
            logger.warning(f"Failed to load predictor data: {e}")

    def _save_data(self) -> None:
        """Save data to disk."""
        data = {
            "actions": [
                {
                    "action_type": a.action_type,
                    "timestamp": a.timestamp.isoformat(),
                    "context": a.context,
                    "metadata": a.metadata,
                }
                for a in self.actions
            ],
            "patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "pattern_type": p.pattern_type.value,
                    "action_type": p.action_type,
                    "confidence": p.confidence,
                    "frequency": p.frequency,
                    "typical_times": p.typical_times,
                    "triggers": p.triggers,
                    "period_hours": p.period_hours,
                    "next_predicted": p.next_predicted.isoformat() if p.next_predicted else None,
                    "created_at": p.created_at.isoformat(),
                    "updated_at": p.updated_at.isoformat(),
                }
                for p in self.patterns.values()
            ],
            "predictions": [
                {
                    "prediction_id": p.prediction_id,
                    "pattern_id": p.pattern_id,
                    "predicted_action": p.predicted_action,
                    "confidence": p.confidence,
                    "reason": p.reason,
                    "suggested_preparation": p.suggested_preparation,
                    "predicted_time": p.predicted_time.isoformat(),
                    "created_at": p.created_at.isoformat(),
                }
                for p in self.predictions
            ],
        }

        self.data_file.write_text(json.dumps(data, indent=2))

    def record_action(
        self,
        action_type: str,
        context: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Record a user action for pattern analysis.

        Args:
            action_type: Type of action (e.g., "query_weekly_report")
            context: Contextual information (e.g., {"day_of_week": "Monday"})
            metadata: Additional metadata
        """
        action = UserAction(
            action_type=action_type,
            timestamp=datetime.now(),
            context=context or {},
            metadata=metadata or {},
        )
        self.actions.append(action)

        # Keep only last 1000 actions to prevent memory bloat
        if len(self.actions) > 1000:
            self.actions = self.actions[-1000:]

        logger.debug(f"Predictor: recorded action '{action_type}'")

        # Auto-analyze patterns after recording
        self._analyze_patterns()
        self._save_data()

    def _analyze_patterns(self) -> None:
        """Analyze recorded actions to discover patterns."""
        if len(self.actions) < 3:
            return

        # Group actions by type
        actions_by_type: dict[str, list[UserAction]] = defaultdict(list)
        for action in self.actions:
            actions_by_type[action.action_type].append(action)

        for action_type, actions in actions_by_type.items():
            if len(actions) < 3:
                continue

            # Analyze time-based patterns
            self._detect_time_pattern(action_type, actions)

            # Analyze periodic patterns
            self._detect_periodic_pattern(action_type, actions)

            # Analyze event-triggered patterns
            self._detect_event_triggers(action_type, actions)

    def _detect_time_pattern(self, action_type: str, actions: list[UserAction]) -> None:
        """Detect time-based patterns (e.g., Monday mornings)."""
        # Count by day of week and hour
        time_distribution: dict[tuple[int, int], int] = defaultdict(int)
        for action in actions:
            dow = action.timestamp.weekday()  # 0=Monday
            hour = action.timestamp.hour
            time_distribution[(dow, hour)] += 1

        # Find significant patterns (occurred at least 3 times)
        for (dow, hour), count in time_distribution.items():
            if count >= 3:
                pattern_id = f"time_{action_type}_{dow}_{hour}"

                days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                day_name = days[dow]

                confidence = min(0.95, 0.5 + (count * 0.1))

                if pattern_id in self.patterns:
                    self.patterns[pattern_id].frequency = count
                    self.patterns[pattern_id].confidence = confidence
                    self.patterns[pattern_id].updated_at = datetime.now()
                else:
                    self.patterns[pattern_id] = BehaviorPattern(
                        pattern_id=pattern_id,
                        pattern_type=PatternType.TIME_BASED,
                        action_type=action_type,
                        confidence=confidence,
                        frequency=count,
                        typical_times=[{"day": day_name, "hour": hour}],
                        triggers=[],
                        period_hours=None,
                        next_predicted=self._calculate_next_time(dow, hour),
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                    )

                logger.debug(f"Predictor: detected time pattern '{pattern_id}' (confidence: {confidence:.2f})")

    def _detect_periodic_pattern(self, action_type: str, actions: list[UserAction]) -> None:
        """Detect periodic patterns (e.g., every 24 hours)."""
        if len(actions) < 4:
            return

        # Calculate intervals between consecutive actions
        intervals: list[float] = []
        sorted_actions = sorted(actions, key=lambda a: a.timestamp)

        for i in range(1, len(sorted_actions)):
            delta = sorted_actions[i].timestamp - sorted_actions[i-1].timestamp
            intervals.append(delta.total_seconds() / 3600)  # Convert to hours

        if not intervals:
            return

        # Check for consistent intervals (within 20% variance)
        avg_interval = sum(intervals) / len(intervals)
        variance = sum((i - avg_interval) ** 2 for i in intervals) / len(intervals)
        std_dev = variance ** 0.5

        if std_dev / avg_interval < 0.2:  # Less than 20% variance
            pattern_id = f"periodic_{action_type}"
            confidence = min(0.95, 0.6 + (len(actions) * 0.05))

            # Predict next occurrence
            last_action = sorted_actions[-1]
            next_predicted = last_action.timestamp + timedelta(hours=avg_interval)

            if pattern_id in self.patterns:
                self.patterns[pattern_id].frequency = len(actions)
                self.patterns[pattern_id].confidence = confidence
                self.patterns[pattern_id].period_hours = avg_interval
                self.patterns[pattern_id].next_predicted = next_predicted
                self.patterns[pattern_id].updated_at = datetime.now()
            else:
                self.patterns[pattern_id] = BehaviorPattern(
                    pattern_id=pattern_id,
                    pattern_type=PatternType.PERIODIC,
                    action_type=action_type,
                    confidence=confidence,
                    frequency=len(actions),
                    typical_times=[],
                    triggers=[],
                    period_hours=avg_interval,
                    next_predicted=next_predicted,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )

            logger.debug(f"Predictor: detected periodic pattern '{pattern_id}' (every {avg_interval:.1f}h)")

    def _detect_event_triggers(self, action_type: str, actions: list[UserAction]) -> None:
        """Detect event-triggered patterns from context."""
        # Look for common trigger events in context
        trigger_counts: dict[str, int] = defaultdict(int)

        for action in actions:
            if "trigger" in action.context:
                trigger = action.context["trigger"]
                if isinstance(trigger, str):
                    trigger_counts[trigger] += 1

        for trigger, count in trigger_counts.items():
            if count >= 2:
                pattern_id = f"event_{action_type}_{trigger}"
                confidence = min(0.9, 0.4 + (count * 0.15))

                if pattern_id in self.patterns:
                    self.patterns[pattern_id].frequency = count
                    self.patterns[pattern_id].confidence = confidence
                    self.patterns[pattern_id].updated_at = datetime.now()
                else:
                    self.patterns[pattern_id] = BehaviorPattern(
                        pattern_id=pattern_id,
                        pattern_type=PatternType.EVENT_TRIGGERED,
                        action_type=action_type,
                        confidence=confidence,
                        frequency=count,
                        typical_times=[],
                        triggers=[trigger],
                        period_hours=None,
                        next_predicted=None,
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                    )

                logger.debug(f"Predictor: detected event pattern '{pattern_id}'")

    def _calculate_next_time(self, day_of_week: int, hour: int) -> datetime:
        """Calculate the next occurrence of a specific day/hour."""
        now = datetime.now()
        days_ahead = day_of_week - now.weekday()

        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7

        next_date = now + timedelta(days=days_ahead)
        return next_date.replace(hour=hour, minute=0, second=0, microsecond=0)

    def predict_needs(
        self,
        look_ahead_hours: float = 24,
        min_confidence: float = 0.5,
    ) -> list[Prediction]:
        """
        Predict user needs based on discovered patterns.

        Args:
            look_ahead_hours: How far ahead to predict
            min_confidence: Minimum confidence threshold

        Returns:
            List of predictions sorted by confidence
        """
        now = datetime.now()
        predictions: list[Prediction] = []

        for pattern in self.patterns.values():
            if pattern.confidence < min_confidence:
                continue

            predicted_time: Optional[datetime] = None
            reason = ""
            suggestions: list[str] = []

            if pattern.pattern_type == PatternType.TIME_BASED:
                # Check if pattern will occur within look_ahead
                if pattern.typical_times:
                    tt = pattern.typical_times[0]
                    next_time = self._calculate_next_time(
                        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(tt["day"]),
                        tt["hour"]
                    )
                    if next_time <= now + timedelta(hours=look_ahead_hours):
                        predicted_time = next_time
                        reason = f"You usually {pattern.action_type} on {tt['day']} around {tt['hour']}:00"
                        suggestions = self._generate_suggestions(pattern.action_type)

            elif pattern.pattern_type == PatternType.PERIODIC and pattern.next_predicted:
                if pattern.next_predicted <= now + timedelta(hours=look_ahead_hours):
                    predicted_time = pattern.next_predicted
                    period_desc = f"every {pattern.period_hours:.1f} hours" if pattern.period_hours else "periodically"
                    reason = f"You {pattern.action_type} {period_desc}"
                    suggestions = self._generate_suggestions(pattern.action_type)

            elif pattern.pattern_type == PatternType.EVENT_TRIGGERED:
                # Event-triggered patterns don't have a predicted time
                # They are checked when events occur
                continue

            if predicted_time:
                prediction = Prediction(
                    prediction_id=f"pred_{pattern.pattern_id}_{int(time.time())}",
                    pattern_id=pattern.pattern_id,
                    predicted_action=pattern.action_type,
                    confidence=pattern.confidence,
                    reason=reason,
                    suggested_preparation=suggestions,
                    predicted_time=predicted_time,
                    created_at=now,
                )
                predictions.append(prediction)

        # Sort by confidence (highest first)
        predictions.sort(key=lambda p: p.confidence, reverse=True)

        # Store predictions
        self.predictions = predictions
        self._save_data()

        return predictions

    def _generate_suggestions(self, action_type: str) -> list[str]:
        """Generate preparation suggestions based on action type."""
        suggestions: dict[str, list[str]] = {
            "query_weekly_report": [
                "Pre-generate weekly summary from recent activities",
                "Collect metrics and statistics for the week",
                "Prepare template with last week's data",
            ],
            "check_email": [
                "Summarize unread emails",
                "Flag high-priority messages",
                "Draft responses to common inquiries",
            ],
            "review_code": [
                "Fetch recent commits and PRs",
                "Check CI/CD status",
                "Summarize code changes",
            ],
            "daily_standup": [
                "Compile yesterday's accomplishments",
                "List today's planned tasks",
                "Identify blockers from messages",
            ],
        }

        # Default suggestions
        defaults = [
            f"Prepare relevant information for {action_type}",
            "Check related resources and dependencies",
        ]

        return suggestions.get(action_type, defaults)

    def check_event_trigger(self, event_type: str, event_data: dict[str, Any]) -> list[Prediction]:
        """
        Check if any event-triggered patterns match the given event.

        Args:
            event_type: Type of event (e.g., "email_received")
            event_data: Event data

        Returns:
            List of matching predictions
        """
        predictions: list[Prediction] = []

        for pattern in self.patterns.values():
            if pattern.pattern_type != PatternType.EVENT_TRIGGERED:
                continue

            if event_type in pattern.triggers or any(t in str(event_data) for t in pattern.triggers):
                prediction = Prediction(
                    prediction_id=f"pred_event_{pattern.pattern_id}_{int(time.time())}",
                    pattern_id=pattern.pattern_id,
                    predicted_action=pattern.action_type,
                    confidence=pattern.confidence,
                    reason=f"Triggered by {event_type}",
                    suggested_preparation=self._generate_suggestions(pattern.action_type),
                    predicted_time=datetime.now(),
                    created_at=datetime.now(),
                )
                predictions.append(prediction)

        return predictions

    def generate_report(self) -> str:
        """Generate a prediction report to PREDICTIONS.md."""
        now = datetime.now()

        lines = [
            "# Prediction Report",
            "",
            f"Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Discovered Patterns",
            "",
        ]

        if self.patterns:
            for pattern in sorted(self.patterns.values(), key=lambda p: p.confidence, reverse=True):
                lines.append(f"### {pattern.pattern_id}")
                lines.append(f"- **Type**: {pattern.pattern_type.value}")
                lines.append(f"- **Action**: {pattern.action_type}")
                lines.append(f"- **Confidence**: {pattern.confidence:.2f}")
                lines.append(f"- **Frequency**: {pattern.frequency} times observed")

                if pattern.typical_times:
                    times_str = ", ".join(f"{t['day']} {t['hour']}:00" for t in pattern.typical_times)
                    lines.append(f"- **Typical Times**: {times_str}")

                if pattern.triggers:
                    lines.append(f"- **Triggers**: {', '.join(pattern.triggers)}")

                if pattern.period_hours:
                    lines.append(f"- **Period**: {pattern.period_hours:.1f} hours")

                if pattern.next_predicted:
                    lines.append(f"- **Next Predicted**: {pattern.next_predicted.strftime('%Y-%m-%d %H:%M')}")

                lines.append("")
        else:
            lines.append("No patterns discovered yet. Keep using the system to build patterns.")
            lines.append("")

        lines.extend([
            "## Active Predictions",
            "",
        ])

        # Generate fresh predictions
        predictions = self.predict_needs(look_ahead_hours=48)

        if predictions:
            for pred in predictions:
                lines.append(f"### {pred.predicted_action}")
                lines.append(f"- **Confidence**: {pred.confidence:.2f}")
                lines.append(f"- **Predicted Time**: {pred.predicted_time.strftime('%Y-%m-%d %H:%M')}")
                lines.append(f"- **Reason**: {pred.reason}")
                lines.append("- **Suggested Preparation**:")
                for suggestion in pred.suggested_preparation:
                    lines.append(f"  - {suggestion}")
                lines.append("")
        else:
            lines.append("No active predictions for the next 48 hours.")
            lines.append("")

        lines.extend([
            "## Statistics",
            "",
            f"- Total Actions Recorded: {len(self.actions)}",
            f"- Patterns Discovered: {len(self.patterns)}",
            f"- Active Predictions: {len(predictions)}",
            "",
            "---",
            "",
            "*This report is auto-generated by the Predictor module.*",
        ])

        report = "\n".join(lines)
        self.predictions_file.write_text(report)

        return report

    def get_suggestions_for_now(self) -> list[dict[str, Any]]:
        """
        Get proactive suggestions for the current time.

        Returns:
            List of suggestions with confidence scores
        """
        now = datetime.now()
        suggestions: list[dict[str, Any]] = []

        # Check time-based patterns
        current_dow = now.weekday()
        current_hour = now.hour

        for pattern in self.patterns.values():
            if pattern.pattern_type == PatternType.TIME_BASED and pattern.typical_times:
                tt = pattern.typical_times[0]
                dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                if dow_names.index(tt["day"]) == current_dow and abs(tt["hour"] - current_hour) <= 1:
                    suggestions.append({
                        "action": pattern.action_type,
                        "confidence": pattern.confidence,
                        "message": f"You usually {pattern.action_type} around this time. Want me to prepare?",
                        "preparations": self._generate_suggestions(pattern.action_type),
                    })

        return sorted(suggestions, key=lambda s: s["confidence"], reverse=True)

    def clear_old_data(self, days: int = 30) -> int:
        """
        Clear action data older than specified days.

        Args:
            days: Number of days to keep

        Returns:
            Number of actions removed
        """
        cutoff = datetime.now() - timedelta(days=days)
        original_count = len(self.actions)
        self.actions = [a for a in self.actions if a.timestamp > cutoff]
        removed = original_count - len(self.actions)

        if removed > 0:
            self._save_data()
            logger.info(f"Predictor: cleared {removed} old actions")

        return removed
