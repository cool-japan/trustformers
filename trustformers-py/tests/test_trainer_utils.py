"""
Tests for the trainer utilities module
"""

import pytest
import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from typing import Dict, Any, List, Optional

# Test trainer utilities module
try:
    from trustformers.trainer_utils import (
        TrainerState,
        TrainerControl,
        TrainerCallback,
        EarlyStoppingCallback,
        ProgressCallback,
        MetricsCallback,
        TensorBoardCallback,
        WandbCallback,
        ModelCheckpointCallback,
        CallbackHandler,
        LearningRateScheduler,
        LinearScheduler,
        CosineScheduler,
        get_default_callbacks,
        setup_logging_callbacks,
    )
    TRAINER_UTILS_AVAILABLE = True
except ImportError:
    TRAINER_UTILS_AVAILABLE = False

# Skip all tests if trainer utils module not available
pytestmark = pytest.mark.skipif(not TRAINER_UTILS_AVAILABLE, reason="Trainer utils module not available")

class TestTrainerState:
    """Test TrainerState functionality"""
    
    def test_trainer_state_creation(self):
        """Test basic trainer state creation"""
        state = TrainerState()
        
        # Check default values
        assert state.epoch == 0
        assert state.global_step == 0
        assert state.max_steps is None
        assert state.num_train_epochs is None
        assert state.log_history == []
        assert state.best_metric is None
        assert state.best_model_checkpoint is None
        assert state.is_local_process_zero is True
        assert state.is_world_process_zero is True
    
    def test_trainer_state_with_values(self):
        """Test trainer state with custom values"""
        state = TrainerState(
            epoch=5,
            global_step=1000,
            max_steps=2000,
            num_train_epochs=10
        )
        
        assert state.epoch == 5
        assert state.global_step == 1000
        assert state.max_steps == 2000
        assert state.num_train_epochs == 10
    
    def test_trainer_state_save_load(self):
        """Test trainer state save and load"""
        state = TrainerState(
            epoch=3,
            global_step=500,
            max_steps=1000
        )
        
        # Add some log history
        state.log_history.append({"loss": 0.5, "epoch": 1})
        state.log_history.append({"loss": 0.3, "epoch": 2})
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            # Save state
            state.save_to_json(filepath)
            
            # Load state
            loaded_state = TrainerState.load_from_json(filepath)
            
            assert loaded_state.epoch == 3
            assert loaded_state.global_step == 500
            assert loaded_state.max_steps == 1000
            assert len(loaded_state.log_history) == 2
        finally:
            Path(filepath).unlink(missing_ok=True)
    
    def test_trainer_state_log_addition(self):
        """Test adding logs to trainer state"""
        state = TrainerState()
        
        # Add log entry
        log_entry = {"loss": 0.5, "learning_rate": 1e-5, "epoch": 1}
        state.log_history.append(log_entry)
        
        assert len(state.log_history) == 1
        assert state.log_history[0] == log_entry
    
    def test_trainer_state_best_metric_tracking(self):
        """Test best metric tracking"""
        state = TrainerState()
        
        # Set best metric
        state.best_metric = 0.95
        state.best_model_checkpoint = "/path/to/best/model"
        
        assert state.best_metric == 0.95
        assert state.best_model_checkpoint == "/path/to/best/model"

class TestTrainerControl:
    """Test TrainerControl functionality"""
    
    def test_trainer_control_creation(self):
        """Test basic trainer control creation"""
        control = TrainerControl()
        
        # Check default values
        assert control.should_training_stop is False
        assert control.should_epoch_stop is False
        assert control.should_save is False
        assert control.should_evaluate is False
        assert control.should_log is False
    
    def test_trainer_control_flags(self):
        """Test setting trainer control flags"""
        control = TrainerControl()
        
        # Set flags
        control.should_training_stop = True
        control.should_save = True
        control.should_evaluate = True
        
        assert control.should_training_stop is True
        assert control.should_save is True
        assert control.should_evaluate is True
        assert control.should_epoch_stop is False
        assert control.should_log is False

class TestTrainerCallback:
    """Test TrainerCallback base class"""
    
    def test_trainer_callback_interface(self):
        """Test trainer callback interface"""
        callback = TrainerCallback()
        
        # Should have all expected methods
        assert hasattr(callback, 'on_init_end')
        assert hasattr(callback, 'on_train_begin')
        assert hasattr(callback, 'on_train_end')
        assert hasattr(callback, 'on_epoch_begin')
        assert hasattr(callback, 'on_epoch_end')
        assert hasattr(callback, 'on_step_begin')
        assert hasattr(callback, 'on_step_end')
        assert hasattr(callback, 'on_evaluate')
        assert hasattr(callback, 'on_save')
        assert hasattr(callback, 'on_log')
    
    def test_trainer_callback_default_behavior(self):
        """Test trainer callback default behavior"""
        callback = TrainerCallback()
        model = MagicMock()
        state = TrainerState()
        control = TrainerControl()
        
        # All callbacks should return control unchanged by default
        result = callback.on_train_begin(None, model, state, control)
        assert result is control
        
        result = callback.on_step_end(None, model, state, control)
        assert result is control

class TestEarlyStoppingCallback:
    """Test EarlyStoppingCallback functionality"""
    
    def test_early_stopping_creation(self):
        """Test early stopping callback creation"""
        callback = EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.01
        )
        
        assert callback.early_stopping_patience == 3
        assert callback.early_stopping_threshold == 0.01
        assert callback.early_stopping_patience_counter == 0
        assert callback.best_metric is None
    
    def test_early_stopping_patience(self):
        """Test early stopping patience mechanism"""
        callback = EarlyStoppingCallback(early_stopping_patience=2)
        
        model = MagicMock()
        state = TrainerState()
        control = TrainerControl()
        
        # First evaluation - sets baseline
        state.log_history = [{"eval_loss": 1.0}]
        control = callback.on_evaluate(None, model, state, control)
        assert not control.should_training_stop
        assert callback.early_stopping_patience_counter == 0
        
        # Second evaluation - worse metric
        state.log_history.append({"eval_loss": 1.5})
        control = callback.on_evaluate(None, model, state, control)
        assert not control.should_training_stop
        assert callback.early_stopping_patience_counter == 1
        
        # Third evaluation - still worse
        state.log_history.append({"eval_loss": 1.6})
        control = callback.on_evaluate(None, model, state, control)
        assert control.should_training_stop  # Should stop now
        assert callback.early_stopping_patience_counter == 2
    
    def test_early_stopping_improvement(self):
        """Test early stopping with improvement"""
        callback = EarlyStoppingCallback(early_stopping_patience=2)
        
        model = MagicMock()
        state = TrainerState()
        control = TrainerControl()
        
        # Set up initial logs
        state.log_history = [
            {"eval_loss": 1.0},
            {"eval_loss": 1.1},  # Worse
            {"eval_loss": 0.9}   # Better - should reset counter
        ]
        
        control = callback.on_evaluate(None, model, state, control)
        assert not control.should_training_stop
        assert callback.early_stopping_patience_counter == 0  # Reset

class TestProgressCallback:
    """Test ProgressCallback functionality"""
    
    def test_progress_callback_creation(self):
        """Test progress callback creation"""
        callback = ProgressCallback()
        assert callback is not None
    
    def test_progress_callback_train_begin(self):
        """Test progress callback on training begin"""
        callback = ProgressCallback()
        
        model = MagicMock()
        state = TrainerState(max_steps=100)
        control = TrainerControl()
        
        # Should initialize progress tracking
        result = callback.on_train_begin(None, model, state, control)
        assert result is control
    
    def test_progress_callback_step_end(self):
        """Test progress callback on step end"""
        callback = ProgressCallback()
        
        model = MagicMock()
        state = TrainerState(global_step=50, max_steps=100)
        control = TrainerControl()
        
        # Should update progress
        result = callback.on_step_end(None, model, state, control)
        assert result is control

class TestMetricsCallback:
    """Test MetricsCallback functionality"""
    
    def test_metrics_callback_creation(self):
        """Test metrics callback creation"""
        callback = MetricsCallback()
        assert callback is not None
    
    def test_metrics_callback_log(self):
        """Test metrics callback logging"""
        callback = MetricsCallback()
        
        model = MagicMock()
        state = TrainerState()
        state.log_history = [{"loss": 0.5, "learning_rate": 1e-5}]
        control = TrainerControl()
        
        # Should handle logging
        result = callback.on_log(None, model, state, control)
        assert result is control

class TestTensorBoardCallback:
    """Test TensorBoardCallback functionality"""
    
    def test_tensorboard_callback_creation(self):
        """Test TensorBoard callback creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = TensorBoardCallback(log_dir=tmpdir)
            assert callback.log_dir == tmpdir
    
    def test_tensorboard_callback_logging(self):
        """Test TensorBoard callback logging"""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = TensorBoardCallback(log_dir=tmpdir)
            
            model = MagicMock()
            state = TrainerState(global_step=100)
            state.log_history = [{"loss": 0.5, "learning_rate": 1e-5}]
            control = TrainerControl()
            
            # Should handle TensorBoard logging
            with patch('trustformers.trainer_utils.SummaryWriter') as mock_writer:
                result = callback.on_log(None, model, state, control)
                assert result is control

class TestWandbCallback:
    """Test WandbCallback functionality"""
    
    def test_wandb_callback_creation(self):
        """Test Wandb callback creation"""
        callback = WandbCallback(project="test_project")
        assert callback.project == "test_project"
    
    def test_wandb_callback_initialization(self):
        """Test Wandb callback initialization"""
        callback = WandbCallback(project="test_project")
        
        model = MagicMock()
        state = TrainerState()
        control = TrainerControl()
        
        # Mock wandb
        with patch('trustformers.trainer_utils.wandb') as mock_wandb:
            result = callback.on_train_begin(None, model, state, control)
            assert result is control
    
    def test_wandb_callback_logging(self):
        """Test Wandb callback logging"""
        callback = WandbCallback(project="test_project")
        
        model = MagicMock()
        state = TrainerState(global_step=50)
        state.log_history = [{"loss": 0.3, "accuracy": 0.85}]
        control = TrainerControl()
        
        # Mock wandb
        with patch('trustformers.trainer_utils.wandb') as mock_wandb:
            result = callback.on_log(None, model, state, control)
            assert result is control

class TestModelCheckpointCallback:
    """Test ModelCheckpointCallback functionality"""
    
    def test_checkpoint_callback_creation(self):
        """Test model checkpoint callback creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = ModelCheckpointCallback(
                output_dir=tmpdir,
                save_steps=100
            )
            assert callback.output_dir == tmpdir
            assert callback.save_steps == 100
    
    def test_checkpoint_callback_save_condition(self):
        """Test checkpoint save condition"""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = ModelCheckpointCallback(
                output_dir=tmpdir,
                save_steps=100
            )
            
            model = MagicMock()
            state = TrainerState(global_step=100)
            control = TrainerControl()
            
            # Should trigger save at save_steps
            result = callback.on_step_end(None, model, state, control)
            assert result.should_save is True
    
    def test_checkpoint_callback_best_model(self):
        """Test best model checkpointing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = ModelCheckpointCallback(
                output_dir=tmpdir,
                save_best_model=True,
                metric_for_best_model="eval_accuracy",
                greater_is_better=True
            )
            
            model = MagicMock()
            state = TrainerState()
            state.log_history = [{"eval_accuracy": 0.85}]
            control = TrainerControl()
            
            # Should save best model
            result = callback.on_evaluate(None, model, state, control)
            assert result.should_save is True

class TestCallbackHandler:
    """Test CallbackHandler functionality"""
    
    def test_callback_handler_creation(self):
        """Test callback handler creation"""
        callbacks = [ProgressCallback(), MetricsCallback()]
        handler = CallbackHandler(callbacks)
        
        assert len(handler.callbacks) == 2
        assert isinstance(handler.callbacks[0], ProgressCallback)
        assert isinstance(handler.callbacks[1], MetricsCallback)
    
    def test_callback_handler_add_remove(self):
        """Test adding and removing callbacks"""
        handler = CallbackHandler([])
        
        # Add callback
        callback = ProgressCallback()
        handler.add_callback(callback)
        assert len(handler.callbacks) == 1
        
        # Remove callback
        handler.remove_callback(ProgressCallback)
        assert len(handler.callbacks) == 0
    
    def test_callback_handler_call_event(self):
        """Test calling event on all callbacks"""
        mock_callback1 = MagicMock(spec=TrainerCallback)
        mock_callback2 = MagicMock(spec=TrainerCallback)
        
        handler = CallbackHandler([mock_callback1, mock_callback2])
        
        model = MagicMock()
        state = TrainerState()
        control = TrainerControl()
        
        # Call event
        result = handler.call_event("on_train_begin", None, model, state, control)
        
        # Both callbacks should be called
        mock_callback1.on_train_begin.assert_called_once()
        mock_callback2.on_train_begin.assert_called_once()
        assert result is control
    
    def test_callback_handler_control_flow(self):
        """Test callback handler control flow"""
        # Create a callback that modifies control
        class StopCallback(TrainerCallback):
            def on_step_end(self, args, model, state, control, **kwargs):
                control.should_training_stop = True
                return control
        
        handler = CallbackHandler([StopCallback()])
        
        model = MagicMock()
        state = TrainerState()
        control = TrainerControl()
        
        # Call should modify control
        result = handler.call_event("on_step_end", None, model, state, control)
        assert result.should_training_stop is True

class TestLearningRateScheduler:
    """Test LearningRateScheduler base class"""
    
    def test_lr_scheduler_interface(self):
        """Test learning rate scheduler interface"""
        scheduler = LearningRateScheduler()
        
        # Should have expected methods
        assert hasattr(scheduler, 'get_lr')
        assert hasattr(scheduler, 'step')

class TestLinearScheduler:
    """Test LinearScheduler functionality"""
    
    def test_linear_scheduler_creation(self):
        """Test linear scheduler creation"""
        scheduler = LinearScheduler(
            optimizer=None,
            num_warmup_steps=100,
            num_training_steps=1000
        )
        
        assert scheduler.num_warmup_steps == 100
        assert scheduler.num_training_steps == 1000
    
    def test_linear_scheduler_warmup(self):
        """Test linear scheduler warmup phase"""
        scheduler = LinearScheduler(
            optimizer=None,
            num_warmup_steps=100,
            num_training_steps=1000
        )
        
        # During warmup, LR should increase linearly
        lr_0 = scheduler.get_lr(step=0)
        lr_50 = scheduler.get_lr(step=50)
        lr_100 = scheduler.get_lr(step=100)
        
        assert lr_0 < lr_50 < lr_100
    
    def test_linear_scheduler_decay(self):
        """Test linear scheduler decay phase"""
        scheduler = LinearScheduler(
            optimizer=None,
            num_warmup_steps=100,
            num_training_steps=1000
        )
        
        # After warmup, LR should decrease linearly
        lr_200 = scheduler.get_lr(step=200)
        lr_500 = scheduler.get_lr(step=500)
        lr_900 = scheduler.get_lr(step=900)
        
        assert lr_200 > lr_500 > lr_900

class TestCosineScheduler:
    """Test CosineScheduler functionality"""
    
    def test_cosine_scheduler_creation(self):
        """Test cosine scheduler creation"""
        scheduler = CosineScheduler(
            optimizer=None,
            num_warmup_steps=100,
            num_training_steps=1000
        )
        
        assert scheduler.num_warmup_steps == 100
        assert scheduler.num_training_steps == 1000
    
    def test_cosine_scheduler_warmup(self):
        """Test cosine scheduler warmup phase"""
        scheduler = CosineScheduler(
            optimizer=None,
            num_warmup_steps=100,
            num_training_steps=1000
        )
        
        # During warmup, LR should increase
        lr_0 = scheduler.get_lr(step=0)
        lr_50 = scheduler.get_lr(step=50)
        lr_100 = scheduler.get_lr(step=100)
        
        assert lr_0 < lr_50 < lr_100
    
    def test_cosine_scheduler_cosine_decay(self):
        """Test cosine scheduler cosine decay"""
        scheduler = CosineScheduler(
            optimizer=None,
            num_warmup_steps=100,
            num_training_steps=1000
        )
        
        # After warmup, should follow cosine curve
        lr_150 = scheduler.get_lr(step=150)
        lr_550 = scheduler.get_lr(step=550)  # Middle of decay
        lr_950 = scheduler.get_lr(step=950)
        
        # Should decrease but not linearly
        assert lr_150 > lr_550 > lr_950

class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_get_default_callbacks(self):
        """Test get_default_callbacks function"""
        callbacks = get_default_callbacks()
        
        assert isinstance(callbacks, list)
        assert len(callbacks) > 0
        
        # Should contain standard callbacks
        callback_types = [type(cb).__name__ for cb in callbacks]
        assert "ProgressCallback" in callback_types
        assert "MetricsCallback" in callback_types
    
    def test_get_default_callbacks_with_options(self):
        """Test get_default_callbacks with options"""
        with tempfile.TemporaryDirectory() as tmpdir:
            callbacks = get_default_callbacks(
                output_dir=tmpdir,
                logging_dir=tmpdir,
                save_steps=100,
                early_stopping_patience=3
            )
            
            callback_types = [type(cb).__name__ for cb in callbacks]
            assert "ModelCheckpointCallback" in callback_types
            assert "EarlyStoppingCallback" in callback_types
    
    def test_setup_logging_callbacks(self):
        """Test setup_logging_callbacks function"""
        with tempfile.TemporaryDirectory() as tmpdir:
            callbacks = setup_logging_callbacks(
                logging_dir=tmpdir,
                tensorboard=True,
                wandb_project="test_project"
            )
            
            assert isinstance(callbacks, list)
            callback_types = [type(cb).__name__ for cb in callbacks]
            assert "TensorBoardCallback" in callback_types
            assert "WandbCallback" in callback_types

class TestTrainerUtilsIntegration:
    """Integration tests for trainer utils"""
    
    def test_full_training_simulation(self):
        """Test simulating a full training loop with callbacks"""
        # Set up callbacks
        callbacks = [
            ProgressCallback(),
            MetricsCallback(),
            EarlyStoppingCallback(early_stopping_patience=2)
        ]
        
        handler = CallbackHandler(callbacks)
        
        # Set up state and control
        state = TrainerState(max_steps=10, num_train_epochs=2)
        control = TrainerControl()
        model = MagicMock()
        
        # Simulate training begin
        control = handler.call_event("on_train_begin", None, model, state, control)
        
        # Simulate training steps
        for step in range(10):
            state.global_step = step + 1
            
            # Step begin
            control = handler.call_event("on_step_begin", None, model, state, control)
            
            # Simulate some training metrics
            if step % 3 == 0:  # Evaluate every 3 steps
                state.log_history.append({
                    "eval_loss": 1.0 - step * 0.05,  # Improving loss
                    "step": step
                })
                control = handler.call_event("on_evaluate", None, model, state, control)
            
            # Step end
            control = handler.call_event("on_step_end", None, model, state, control)
            
            if control.should_training_stop:
                break
        
        # Simulate training end
        control = handler.call_event("on_train_end", None, model, state, control)
        
        # Should have completed without early stopping (loss was improving)
        assert not control.should_training_stop
    
    def test_callback_interaction(self):
        """Test interaction between different callbacks"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create callbacks that might interact
            callbacks = [
                ModelCheckpointCallback(output_dir=tmpdir, save_steps=5),
                EarlyStoppingCallback(early_stopping_patience=1),
                MetricsCallback()
            ]
            
            handler = CallbackHandler(callbacks)
            
            state = TrainerState(max_steps=10)
            control = TrainerControl()
            model = MagicMock()
            
            # Simulate steps with worsening performance
            for step in range(8):
                state.global_step = step + 1
                
                # Add worsening metrics
                if step % 2 == 0:
                    state.log_history.append({
                        "eval_loss": 1.0 + step * 0.1,  # Worsening
                        "step": step
                    })
                    control = handler.call_event("on_evaluate", None, model, state, control)
                
                control = handler.call_event("on_step_end", None, model, state, control)
                
                if control.should_training_stop:
                    break
            
            # Should have stopped early due to worsening performance
            assert control.should_training_stop

if __name__ == "__main__":
    pytest.main([__file__])