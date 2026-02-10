"""Integration module for Predictor with Cron and Heartbeat services."""

import asyncio
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

from loguru import logger

from nanobot.agent.predictor import Predictor
from nanobot.cron.service import CronService
from nanobot.cron.types import CronSchedule
from nanobot.heartbeat.service import HeartbeatService


class PredictorIntegration:
    """
    Integrates the Predictor module with Cron and Heartbeat services.

    This class handles:
    - Scheduling periodic pattern analysis
    - Generating prediction reports
    - Proactive suggestions via heartbeat
    """

    def __init__(
        self,
        workspace: Path,
        predictor: Optional[Predictor] = None,
        cron_service: Optional[CronService] = None,
        heartbeat_service: Optional[HeartbeatService] = None,
        on_suggestion: Optional[Callable[[dict[str, Any]], Coroutine[Any, Any, None]]] = None,
    ):
        self.workspace = workspace
        self.predictor = predictor or Predictor(workspace)
        self.cron_service = cron_service
        self.heartbeat_service = heartbeat_service
        self.on_suggestion = on_suggestion

        self._prediction_job_id: Optional[str] = None
        self._report_job_id: Optional[str] = None
        self._original_heartbeat_handler: Optional[Callable[[str], Coroutine[Any, Any, str]]] = None

    async def setup(self) -> None:
        """Setup predictor integration with cron and heartbeat."""
        # Setup cron jobs for periodic tasks
        if self.cron_service:
            await self._setup_cron_jobs()

        # Setup heartbeat integration
        if self.heartbeat_service:
            await self._setup_heartbeat()

        logger.info("Predictor integration setup complete")

    async def _setup_cron_jobs(self) -> None:
        """Setup scheduled cron jobs for predictor."""
        # Job 1: Generate prediction report daily at 8 AM
        self._report_job_id = self.cron_service.add_job(
            name="predictor_daily_report",
            schedule=CronSchedule(kind="cron", expr="0 8 * * *"),  # 8 AM daily
            message="Generate daily prediction report",
            deliver=False,
        ).id

        # Job 2: Analyze patterns and update predictions every 6 hours
        self._prediction_job_id = self.cron_service.add_job(
            name="predictor_pattern_analysis",
            schedule=CronSchedule(kind="every", every_ms=6 * 60 * 60 * 1000),  # 6 hours
            message="Analyze behavior patterns and update predictions",
            deliver=False,
        ).id

        logger.info(f"Predictor: scheduled cron jobs (report: {self._report_job_id}, analysis: {self._prediction_job_id})")

    async def _setup_heartbeat(self) -> None:
        """Setup heartbeat integration for proactive suggestions."""
        # Store original handler
        self._original_heartbeat_handler = self.heartbeat_service.on_heartbeat

        # Wrap with predictor suggestions
        self.heartbeat_service.on_heartbeat = self._heartbeat_with_predictions

        logger.info("Predictor: integrated with heartbeat service")

    async def _heartbeat_with_predictions(self, prompt: str) -> str:
        """Enhanced heartbeat handler with prediction suggestions."""
        # Check for proactive suggestions
        suggestions = self.predictor.get_suggestions_for_now()

        if suggestions and self.on_suggestion:
            for suggestion in suggestions:
                if suggestion["confidence"] >= 0.6:  # Only high-confidence suggestions
                    try:
                        await self.on_suggestion(suggestion)
                    except Exception as e:
                        logger.error(f"Failed to send suggestion: {e}")

        # Call original handler if exists
        if self._original_heartbeat_handler:
            return await self._original_heartbeat_handler(prompt)

        return "HEARTBEAT_OK"

    async def handle_cron_job(self, job_name: str) -> str:
        """
        Handle predictor-related cron jobs.

        Args:
            job_name: Name of the cron job

        Returns:
            Result message
        """
        if job_name == "predictor_daily_report":
            return await self._generate_report()

        elif job_name == "predictor_pattern_analysis":
            return await self._analyze_patterns()

        return "Unknown predictor job"

    async def _generate_report(self) -> str:
        """Generate prediction report."""
        try:
            report = self.predictor.generate_report()
            logger.info("Predictor: generated daily report")
            return "Report generated successfully"
        except Exception as e:
            logger.error(f"Predictor: failed to generate report: {e}")
            return f"Failed to generate report: {e}"

    async def _analyze_patterns(self) -> str:
        """Analyze patterns and update predictions."""
        try:
            # Pattern analysis happens automatically on record_action
            # This job can trigger additional analysis if needed
            predictions = self.predictor.predict_needs(look_ahead_hours=48)
            logger.info(f"Predictor: analyzed patterns, {len(predictions)} predictions active")
            return f"Pattern analysis complete. {len(predictions)} active predictions."
        except Exception as e:
            logger.error(f"Predictor: failed to analyze patterns: {e}")
            return f"Failed to analyze patterns: {e}"

    def record_user_action(
        self,
        action_type: str,
        context: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Record a user action for pattern analysis.

        This is a convenience method that wraps Predictor.record_action().
        """
        self.predictor.record_action(action_type, context, metadata)

    def get_predictions(self, look_ahead_hours: float = 24) -> list[dict[str, Any]]:
        """
        Get current predictions.

        Args:
            look_ahead_hours: How far ahead to predict

        Returns:
            List of predictions as dictionaries
        """
        predictions = self.predictor.predict_needs(look_ahead_hours=look_ahead_hours)
        return [
            {
                "action": p.predicted_action,
                "confidence": p.confidence,
                "reason": p.reason,
                "predicted_time": p.predicted_time.isoformat(),
                "suggestions": p.suggested_preparation,
            }
            for p in predictions
        ]

    def check_event(self, event_type: str, event_data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Check for event-triggered predictions.

        Args:
            event_type: Type of event
            event_data: Event data

        Returns:
            List of triggered predictions
        """
        predictions = self.predictor.check_event_trigger(event_type, event_data)
        return [
            {
                "action": p.predicted_action,
                "confidence": p.confidence,
                "reason": p.reason,
                "suggestions": p.suggested_preparation,
            }
            for p in predictions
        ]

    async def cleanup(self) -> None:
        """Cleanup and remove scheduled jobs."""
        # Remove cron jobs
        if self.cron_service:
            if self._report_job_id:
                self.cron_service.remove_job(self._report_job_id)
            if self._prediction_job_id:
                self.cron_service.remove_job(self._prediction_job_id)

        # Restore original heartbeat handler
        if self.heartbeat_service and self._original_heartbeat_handler:
            self.heartbeat_service.on_heartbeat = self._original_heartbeat_handler

        logger.info("Predictor integration cleanup complete")


class PredictorPlugin:
    """
    Plugin class for easy integration into the agent system.

    Usage:
        plugin = PredictorPlugin(workspace)
        await plugin.setup(cron_service, heartbeat_service)

        # Record actions
        plugin.record_action("query_weekly_report", {"day_of_week": "Monday"})

        # Get suggestions
        suggestions = plugin.get_suggestions()
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.predictor = Predictor(workspace)
        self.integration: Optional[PredictorIntegration] = None

    async def setup(
        self,
        cron_service: Optional[CronService] = None,
        heartbeat_service: Optional[HeartbeatService] = None,
        on_suggestion: Optional[Callable[[dict[str, Any]], Coroutine[Any, Any, None]]] = None,
    ) -> None:
        """Setup the predictor plugin."""
        self.integration = PredictorIntegration(
            workspace=self.workspace,
            predictor=self.predictor,
            cron_service=cron_service,
            heartbeat_service=heartbeat_service,
            on_suggestion=on_suggestion,
        )
        await self.integration.setup()

    def record_action(
        self,
        action_type: str,
        context: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record a user action."""
        self.predictor.record_action(action_type, context, metadata)

    def get_suggestions(self) -> list[dict[str, Any]]:
        """Get proactive suggestions for current time."""
        return self.predictor.get_suggestions_for_now()

    def get_predictions(self, hours: float = 24) -> list[dict[str, Any]]:
        """Get predictions for the next N hours."""
        if self.integration:
            return self.integration.get_predictions(hours)
        return []

    def generate_report(self) -> str:
        """Generate prediction report."""
        return self.predictor.generate_report()

    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        if self.integration:
            await self.integration.cleanup()
