"""Unit tests for RagasMetrics."""

import concurrent.futures
import math

import pytest
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.embedding.manager import EmbeddingError
from lightspeed_evaluation.core.metrics.ragas import RagasMetrics, _clamp_score
from lightspeed_evaluation.core.models import EvaluationScope
from lightspeed_evaluation.core.system.exceptions import (
    ConfigurationError,
    EvaluationError,
)


class TestLazyEmbeddingManagerProperty:
    """Test lazy initialization of the embedding_manager property."""

    def test_initialized_on_first_access(
        self, ragas_metrics: RagasMetrics, mocker: MockerFixture
    ) -> None:
        """First property access should create and return RagasEmbeddingManager."""
        mock_cls = mocker.patch(
            "lightspeed_evaluation.core.metrics.ragas.RagasEmbeddingManager",
        )

        result = ragas_metrics.embedding_manager

        assert result is mock_cls.return_value
        mock_cls.assert_called_once()

    def test_cached_after_first_access(
        self, ragas_metrics: RagasMetrics, mocker: MockerFixture
    ) -> None:
        """Subsequent accesses should return the cached instance."""
        mock_cls = mocker.patch(
            "lightspeed_evaluation.core.metrics.ragas.RagasEmbeddingManager",
        )

        first = ragas_metrics.embedding_manager
        second = ragas_metrics.embedding_manager

        assert first is second
        mock_cls.assert_called_once()

    def test_concurrent_access_creates_single_instance(
        self, ragas_metrics: RagasMetrics, mocker: MockerFixture
    ) -> None:
        """Concurrent threads should only create one RagasEmbeddingManager."""
        mock_cls = mocker.patch(
            "lightspeed_evaluation.core.metrics.ragas.RagasEmbeddingManager",
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(lambda: ragas_metrics.embedding_manager)
                for _ in range(8)
            ]
            results = [f.result() for f in futures]

        mock_cls.assert_called_once()
        assert all(r is results[0] for r in results)


class TestEvaluateExceptionHandling:
    """Test that evaluate() catches the expected exception types."""

    @pytest.mark.parametrize(
        "exception_case",
        [
            (EvaluationError, "base evaluation error"),
            (ConfigurationError, "unknown provider xyz"),
            (EmbeddingError, "unsupported embedding provider"),
            (RuntimeError, "unexpected runtime failure"),
            (ValueError, "invalid value"),
            (TypeError, "type mismatch"),
            (ImportError, "missing module"),
        ],
    )
    def test_catches_exception_gracefully(
        self,
        ragas_metrics: RagasMetrics,
        turn_scope: EvaluationScope,
        mocker: MockerFixture,
        exception_case: tuple[type, str],
    ) -> None:
        """evaluate() should return (None, error_message) for caught exceptions."""
        exception_class, exception_msg = exception_case
        ragas_metrics.supported_metrics["faithfulness"] = mocker.MagicMock(
            side_effect=exception_class(exception_msg)
        )

        score, reason = ragas_metrics.evaluate(
            "faithfulness", mocker.MagicMock(), turn_scope
        )

        assert score is None
        assert exception_msg in reason
        assert "evaluation failed" in reason

    def test_unsupported_metric_returns_none(
        self,
        ragas_metrics: RagasMetrics,
        turn_scope: EvaluationScope,
        mocker: MockerFixture,
    ) -> None:
        """Unsupported metric name should return (None, message) without raising."""
        score, reason = ragas_metrics.evaluate(
            "nonexistent_metric", mocker.MagicMock(), turn_scope
        )

        assert score is None
        assert "Unsupported Ragas metric" in reason


class TestClampScore:
    """NaN must survive clamping so a failed Ragas metric is not a silent pass."""

    def test_nan_passes_through(self) -> None:
        """A NaN score is returned unchanged instead of being clamped to 1.0."""
        assert math.isnan(_clamp_score(float("nan")))

    @pytest.mark.parametrize(
        ("raw", "clamped"),
        [
            (1.0000000000000007, 1.0),
            (-1e-12, 0.0),
            (0.5, 0.5),
        ],
    )
    def test_still_clamps_finite_scores(self, raw: float, clamped: float) -> None:
        """Finite scores stay inside [0, 1], including float noise at the bounds."""
        assert _clamp_score(raw) == clamped

    @pytest.mark.parametrize(
        "metric_case",
        [
            ("Faithfulness", "faithfulness"),
            ("ContextRelevance", "context_relevance"),
        ],
    )
    def test_nan_score_becomes_malformed_output_error(
        self,
        ragas_metrics: RagasMetrics,
        turn_scope: EvaluationScope,
        mocker: MockerFixture,
        metric_case: tuple[str, str],
    ) -> None:
        """A mocked NaN score is reported as an error, not a 1.0 pass."""
        metric_class, metric_name = metric_case
        mocked = mocker.patch(
            f"lightspeed_evaluation.core.metrics.ragas.{metric_class}"
        )
        mocked.return_value.score.return_value.value = float("nan")

        score, reason = ragas_metrics.evaluate(
            metric_name, mocker.MagicMock(), turn_scope
        )

        assert score is None
        assert (
            reason == f"Ragas {metric_name} evaluation failed due to malformed "
            "output from the LLM"
        )
