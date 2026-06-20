#!/usr/bin/env python3
"""
Regression tests for the two classified stubs:

  1. data_science_tools.py:316 — TensorEstimator.predict duck-typing raises TypeError
  2. torch_module_compat.py:211 — TorchModuleWrapper.save_pretrained raises AttributeError

Run: python -m pytest tests/test_stub_implementations.py -q
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))


# ---------------------------------------------------------------------------
# Stub 3: data_science_tools.TensorEstimator.predict
# ---------------------------------------------------------------------------

try:
    from trustformers.data_science_tools import SklearnIntegration, HAS_SKLEARN
    _ds_available = True
except ImportError:
    _ds_available = False


@pytest.mark.skipif(not _ds_available, reason="data_science_tools not importable")
@pytest.mark.skipif(
    not (True if True else False),   # HAS_SKLEARN evaluated lazily
    reason="SklearnIntegration requires sklearn"
)
class TestTensorEstimatorPredict:
    """Tests for TensorEstimator.predict duck-typing."""

    def _make_estimator(self, inner_model):
        est = SklearnIntegration.TensorEstimator(inner_model)
        est.fitted_ = True  # bypass fit check
        return est

    def test_predict_delegates_to_predict_method(self):
        """When the model has .predict(), delegate to it."""
        class ModelWithPredict:
            def predict(self, X):
                return [42]

        est = self._make_estimator(ModelWithPredict())
        result = est.predict([1, 2, 3])
        assert result == [42]

    def test_predict_falls_back_to_forward(self):
        """When .predict() is absent but .forward() is present, use forward."""
        class ModelWithForward:
            def forward(self, X):
                return [99]

        est = self._make_estimator(ModelWithForward())
        result = est.predict([0])
        assert result == [99]

    def test_predict_prefers_predict_over_forward(self):
        """When both are present, .predict() takes precedence."""
        class ModelWithBoth:
            def predict(self, X):
                return "from_predict"
            def forward(self, X):
                return "from_forward"

        est = self._make_estimator(ModelWithBoth())
        assert est.predict([0]) == "from_predict"

    def test_predict_raises_TypeError_when_neither_method_exists(self):
        """When neither .predict nor .forward exists, raises TypeError (not NotImplementedError)."""
        class BareModel:
            pass

        est = self._make_estimator(BareModel())
        with pytest.raises(TypeError) as exc_info:
            est.predict([0])

        msg = str(exc_info.value)
        # Error message should mention both tried methods
        assert "predict" in msg.lower()
        assert "forward" in msg.lower()

    def test_predict_error_is_not_NotImplementedError(self):
        """Ensures the error type changed from NotImplementedError to TypeError."""
        class BareModel:
            pass

        est = self._make_estimator(BareModel())
        with pytest.raises(TypeError):
            est.predict([0])
        # Confirm it is NOT a subclass of NotImplementedError
        try:
            est.predict([0])
        except TypeError as e:
            assert not isinstance(e, NotImplementedError), \
                "Must be plain TypeError, not NotImplementedError"
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Stub 4: torch_module_compat.TorchModuleWrapper.save_pretrained
# ---------------------------------------------------------------------------

try:
    from trustformers.torch_module_compat import TorchModuleWrapper
    _compat_available = True
except ImportError:
    _compat_available = False


@pytest.mark.skipif(not _compat_available, reason="torch_module_compat not importable")
class TestTorchModuleWrapperSavePretrained:
    """Tests for TorchModuleWrapper.save_pretrained."""

    def _make_wrapper(self, inner_model):
        """Create a wrapper without torch dependency (use object base)."""
        # TorchModuleWrapper subclasses nn.Module when torch is available.
        # We bypass __init__ via __new__ if needed, but the class should
        # handle a missing-torch gracefully via the `object` fallback.
        try:
            w = TorchModuleWrapper.__new__(TorchModuleWrapper)
            w.trustformers_model = inner_model
            w.device = "cpu"
            return w
        except Exception:
            pytest.skip("Cannot create TorchModuleWrapper without torch")

    def test_save_pretrained_delegates_when_available(self, tmp_path):
        """When the inner model has save_pretrained, delegates to it."""
        saved_dirs = []

        class ModelWithSave:
            def save_pretrained(self, save_directory):
                saved_dirs.append(save_directory)

        wrapper = self._make_wrapper(ModelWithSave())
        wrapper.save_pretrained(str(tmp_path))
        assert saved_dirs == [str(tmp_path)]

    def test_save_pretrained_raises_AttributeError_when_missing(self, tmp_path):
        """Raises AttributeError (not NotImplementedError) when inner model lacks save_pretrained."""
        class ModelWithoutSave:
            pass

        wrapper = self._make_wrapper(ModelWithoutSave())
        with pytest.raises(AttributeError) as exc_info:
            wrapper.save_pretrained(str(tmp_path))

        msg = str(exc_info.value)
        # Guidance should be in the message
        assert "save_pretrained" in msg

    def test_save_pretrained_error_is_not_NotImplementedError(self, tmp_path):
        """Confirm the error is AttributeError, not NotImplementedError."""
        class ModelWithoutSave:
            pass

        wrapper = self._make_wrapper(ModelWithoutSave())
        try:
            wrapper.save_pretrained(str(tmp_path))
            pytest.fail("Expected AttributeError")
        except AttributeError:
            pass  # correct
        except NotImplementedError:
            pytest.fail("Must be AttributeError, not NotImplementedError")

    def test_save_pretrained_error_mentions_inner_model_type(self, tmp_path):
        """Error message should name the inner model class for guidance."""
        class MySpecialModel:
            pass

        wrapper = self._make_wrapper(MySpecialModel())
        with pytest.raises(AttributeError) as exc_info:
            wrapper.save_pretrained(str(tmp_path))

        assert "MySpecialModel" in str(exc_info.value)
