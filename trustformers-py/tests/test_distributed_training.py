"""
Tests for distributed training functionality

This module contains comprehensive tests for:
- DistributedConfig and DistributedManager
- DDPModelWrapper and HorovodModelWrapper  
- FaultToleranceManager
- ElasticTrainingManager
- DistributedTrainer
- Utility functions
"""

import os
import sys
import time
import tempfile
import unittest
import threading
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import trustformers.distributed_training as dt


class TestDistributedConfig(unittest.TestCase):
    """Test DistributedConfig class"""
    
    def setUp(self):
        # Clear environment variables
        self.env_backup = {}
        for key in ["WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
            self.env_backup[key] = os.environ.get(key)
            if key in os.environ:
                del os.environ[key]
    
    def tearDown(self):
        # Restore environment variables
        for key, value in self.env_backup.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]
    
    def test_default_config(self):
        """Test default configuration"""
        config = dt.DistributedConfig()
        
        self.assertEqual(config.backend, dt.DistributedBackend.NCCL)
        self.assertEqual(config.world_size, 1)
        self.assertEqual(config.rank, 0)
        self.assertEqual(config.local_rank, 0)
        self.assertEqual(config.master_addr, "localhost")
        self.assertEqual(config.master_port, "29500")
        self.assertTrue(config.use_env)
        self.assertEqual(config.timeout_seconds, 1800)
    
    def test_config_from_env(self):
        """Test configuration from environment variables"""
        os.environ["WORLD_SIZE"] = "4"
        os.environ["RANK"] = "2"
        os.environ["LOCAL_RANK"] = "1"
        
        config = dt.DistributedConfig()
        
        self.assertEqual(config.world_size, 4)
        self.assertEqual(config.rank, 2)
        self.assertEqual(config.local_rank, 1)
    
    def test_custom_config(self):
        """Test custom configuration parameters"""
        config = dt.DistributedConfig(
            backend=dt.DistributedBackend.GLOO,
            world_size=8,
            rank=3,
            local_rank=2,
            master_addr="192.168.1.100",
            master_port="12345",
            timeout_seconds=3600,
            find_unused_parameters=True,
            bucket_cap_mb=50,
        )
        
        self.assertEqual(config.backend, dt.DistributedBackend.GLOO)
        self.assertEqual(config.world_size, 8)
        self.assertEqual(config.rank, 3)
        self.assertEqual(config.local_rank, 2)
        self.assertEqual(config.master_addr, "192.168.1.100")
        self.assertEqual(config.master_port, "12345")
        self.assertEqual(config.timeout_seconds, 3600)
        self.assertTrue(config.find_unused_parameters)
        self.assertEqual(config.bucket_cap_mb, 50)
    
    def test_setup_environment(self):
        """Test environment setup"""
        config = dt.DistributedConfig(
            world_size=4,
            rank=2,
            local_rank=1,
            master_addr="192.168.1.100",
            master_port="12345",
        )
        
        config.setup_environment()
        
        self.assertEqual(os.environ["WORLD_SIZE"], "4")
        self.assertEqual(os.environ["RANK"], "2")
        self.assertEqual(os.environ["LOCAL_RANK"], "1")
        self.assertEqual(os.environ["MASTER_ADDR"], "192.168.1.100")
        self.assertEqual(os.environ["MASTER_PORT"], "12345")


class TestDistributedManager(unittest.TestCase):
    """Test DistributedManager class"""
    
    def setUp(self):
        self.config = dt.DistributedConfig(
            world_size=2,
            rank=0,
            local_rank=0,
        )
        self.manager = dt.DistributedManager(self.config)
    
    def test_initialization(self):
        """Test manager initialization"""
        self.assertEqual(self.manager.config, self.config)
        self.assertFalse(self.manager._is_initialized)
        self.assertIsNone(self.manager._process_group)
    
    def test_properties_before_init(self):
        """Test properties before initialization"""
        self.assertEqual(self.manager.rank, 0)
        self.assertEqual(self.manager.world_size, 2)
        self.assertEqual(self.manager.local_rank, 0)
        self.assertTrue(self.manager.is_main_process)
    
    @patch('trustformers.distributed_training.TORCH_AVAILABLE', False)
    def test_initialize_without_torch(self):
        """Test initialization fails without PyTorch"""
        with self.assertRaises(RuntimeError):
            self.manager.initialize()
    
    @patch('trustformers.distributed_training.TORCH_AVAILABLE', True)
    @patch('trustformers.distributed_training.dist')
    @patch('trustformers.distributed_training.torch')
    def test_initialize_pytorch(self, mock_torch, mock_dist):
        """Test PyTorch initialization"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.timedelta.return_value = MagicMock()
        
        self.manager.initialize()
        
        self.assertTrue(self.manager._is_initialized)
        mock_dist.init_process_group.assert_called_once()
        mock_torch.cuda.set_device.assert_called_once_with(0)
    
    @patch('trustformers.distributed_training.HOROVOD_AVAILABLE', True)
    @patch('trustformers.distributed_training.torch')
    def test_initialize_horovod(self, mock_torch):
        """Test Horovod initialization"""
        # Mock hvd module at the module level
        with patch('trustformers.distributed_training.hvd', create=True) as mock_hvd:
            self.config.backend = dt.DistributedBackend.HOROVOD
            mock_torch.cuda.is_available.return_value = True
            mock_hvd.local_rank.return_value = 0
            
            self.manager.initialize()
            
            self.assertTrue(self.manager._is_initialized)
            mock_hvd.init.assert_called_once()
            mock_torch.cuda.set_device.assert_called_once_with(0)


class TestDDPModelWrapper(unittest.TestCase):
    """Test DDPModelWrapper class"""
    
    @patch('trustformers.distributed_training.TORCH_AVAILABLE', False)
    def test_wrapper_without_torch(self):
        """Test wrapper fails without PyTorch"""
        with self.assertRaises(RuntimeError):
            dt.DDPModelWrapper(Mock())
    
    @patch('trustformers.distributed_training.TORCH_AVAILABLE', True)
    @patch('trustformers.distributed_training.DDP')
    @patch('trustformers.distributed_training.torch')
    def test_wrapper_creation(self, mock_torch, mock_ddp):
        """Test DDP wrapper creation"""
        mock_model = Mock()
        mock_ddp_instance = Mock()
        mock_ddp.return_value = mock_ddp_instance
        
        wrapper = dt.DDPModelWrapper(
            mock_model,
            device_ids=[0],
            find_unused_parameters=True,
            bucket_cap_mb=50,
        )
        
        self.assertEqual(wrapper.original_model, mock_model)
        self.assertEqual(wrapper.model, mock_ddp_instance)
        mock_ddp.assert_called_once_with(
            mock_model,
            device_ids=[0],
            output_device=None,
            broadcast_buffers=True,
            find_unused_parameters=True,
            bucket_cap_mb=50,
            gradient_as_bucket_view=True,
            static_graph=False,
        )
    
    @patch('trustformers.distributed_training.TORCH_AVAILABLE', True)
    @patch('trustformers.distributed_training.DDP')
    def test_wrapper_methods(self, mock_ddp):
        """Test wrapper methods"""
        mock_model = Mock()
        mock_ddp_instance = Mock()
        mock_ddp.return_value = mock_ddp_instance
        
        wrapper = dt.DDPModelWrapper(mock_model)
        
        # Test forward
        args = (1, 2, 3)
        kwargs = {'a': 1, 'b': 2}
        wrapper.forward(*args, **kwargs)
        mock_ddp_instance.assert_called_once_with(*args, **kwargs)
        
        # Test __call__
        mock_ddp_instance.reset_mock()
        wrapper(*args, **kwargs)
        mock_ddp_instance.assert_called_once_with(*args, **kwargs)
        
        # Test state_dict
        mock_model.state_dict.return_value = {'param': 'value'}
        result = wrapper.state_dict()
        self.assertEqual(result, {'param': 'value'})
        
        # Test load_state_dict
        state_dict = {'param': 'new_value'}
        wrapper.load_state_dict(state_dict)
        mock_model.load_state_dict.assert_called_once_with(state_dict)
        
        # Test parameters
        wrapper.parameters()
        mock_ddp_instance.parameters.assert_called_once()
        
        # Test named_parameters
        wrapper.named_parameters()
        mock_ddp_instance.named_parameters.assert_called_once()


class TestHorovodModelWrapper(unittest.TestCase):
    """Test HorovodModelWrapper class"""
    
    @patch('trustformers.distributed_training.HOROVOD_AVAILABLE', False)
    def test_wrapper_without_horovod(self):
        """Test wrapper fails without Horovod"""
        with self.assertRaises(RuntimeError):
            dt.HorovodModelWrapper(Mock(), Mock())
    
    @patch('trustformers.distributed_training.HOROVOD_AVAILABLE', True)
    def test_wrapper_creation(self):
        """Test Horovod wrapper creation"""
        with patch('trustformers.distributed_training.hvd', create=True) as mock_hvd:
            mock_model = Mock()
            mock_optimizer = Mock()
            mock_dist_optimizer = Mock()
            mock_hvd.DistributedOptimizer.return_value = mock_dist_optimizer
            
            wrapper = dt.HorovodModelWrapper(mock_model, mock_optimizer)
            
            self.assertEqual(wrapper.model, mock_model)
            self.assertEqual(wrapper.optimizer, mock_optimizer)
            self.assertEqual(wrapper.distributed_optimizer, mock_dist_optimizer)
            
            mock_hvd.broadcast_parameters.assert_called_once_with(
                mock_model.state_dict(), root_rank=0
            )
            mock_hvd.broadcast_optimizer_state.assert_called_once_with(
                mock_optimizer, root_rank=0
            )
            mock_hvd.DistributedOptimizer.assert_called_once_with(
                mock_optimizer, named_parameters=mock_model.named_parameters()
            )
    
    @patch('trustformers.distributed_training.HOROVOD_AVAILABLE', True)
    def test_wrapper_methods(self):
        """Test wrapper methods"""
        with patch('trustformers.distributed_training.hvd', create=True) as mock_hvd:
            mock_model = Mock()
            mock_optimizer = Mock()
            mock_dist_optimizer = Mock()
            mock_hvd.DistributedOptimizer.return_value = mock_dist_optimizer
            
            wrapper = dt.HorovodModelWrapper(mock_model, mock_optimizer)
            
            # Test forward
            args = (1, 2, 3)
            kwargs = {'a': 1, 'b': 2}
            wrapper.forward(*args, **kwargs)
            mock_model.assert_called_once_with(*args, **kwargs)
            
            # Test __call__
            mock_model.reset_mock()
            wrapper(*args, **kwargs)
            mock_model.assert_called_once_with(*args, **kwargs)
            
            # Test step
            wrapper.step()
            mock_dist_optimizer.step.assert_called_once()
            
            # Test zero_grad
            wrapper.zero_grad()
            mock_optimizer.zero_grad.assert_called_once()


class TestFaultToleranceManager(unittest.TestCase):
    """Test FaultToleranceManager class"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = dt.FaultToleranceManager(
            checkpoint_dir=self.temp_dir,
            checkpoint_interval=10,
            max_retries=2,
            retry_delay=1.0,
        )
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test manager initialization"""
        self.assertEqual(self.manager.checkpoint_dir, Path(self.temp_dir))
        self.assertEqual(self.manager.checkpoint_interval, 10)
        self.assertEqual(self.manager.max_retries, 2)
        self.assertEqual(self.manager.retry_delay, 1.0)
        self.assertTrue(self.manager.checkpoint_dir.exists())
    
    @patch('trustformers.distributed_training.torch')
    def test_save_checkpoint(self, mock_torch):
        """Test checkpoint saving"""
        mock_model = Mock()
        mock_optimizer = Mock()
        mock_model.state_dict.return_value = {'model': 'state'}
        mock_optimizer.state_dict.return_value = {'optimizer': 'state'}
        
        self.manager.save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            epoch=5,
            step=100,
            loss=0.5,
            metadata={'custom': 'data'}
        )
        
        # Check that torch.save was called
        mock_torch.save.assert_called_once()
        
        # Check that metadata file was created
        metadata_path = self.manager.checkpoint_dir / "checkpoint_metadata.json"
        self.assertTrue(metadata_path.exists())
        
        # Check that checkpoint was recorded in metadata
        self.assertIn(100, self.manager._checkpoint_metadata)
        self.assertEqual(self.manager._checkpoint_metadata[100]['epoch'], 5)
        self.assertEqual(self.manager._checkpoint_metadata[100]['loss'], 0.5)
    
    @patch('trustformers.distributed_training.torch')
    def test_load_latest_checkpoint(self, mock_torch):
        """Test loading latest checkpoint"""
        # First save a checkpoint
        mock_model = Mock()
        mock_optimizer = Mock()
        mock_model.state_dict.return_value = {'model': 'state'}
        mock_optimizer.state_dict.return_value = {'optimizer': 'state'}
        
        # Mock torch.save to create the checkpoint file
        def mock_save(data, path):
            # Create the file when torch.save is called
            Path(path).touch()
        
        mock_torch.save.side_effect = mock_save
        
        self.manager.save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            epoch=5,
            step=100,
            loss=0.5,
        )
        
        # Mock torch.load to return the expected checkpoint
        mock_checkpoint = {
            'model_state_dict': {'model': 'state'},
            'optimizer_state_dict': {'optimizer': 'state'},
            'epoch': 5,
            'step': 100,
            'loss': 0.5,
            'metadata': {},
            'timestamp': time.time(),
        }
        mock_torch.load.return_value = mock_checkpoint
        
        # Test loading
        new_model = Mock()
        new_optimizer = Mock()
        result = self.manager.load_latest_checkpoint(new_model, new_optimizer)
        
        self.assertEqual(result, mock_checkpoint)
        new_model.load_state_dict.assert_called_once_with({'model': 'state'})
        new_optimizer.load_state_dict.assert_called_once_with({'optimizer': 'state'})
    
    def test_load_checkpoint_no_metadata(self):
        """Test loading checkpoint when no metadata exists"""
        mock_model = Mock()
        mock_optimizer = Mock()
        
        result = self.manager.load_latest_checkpoint(mock_model, mock_optimizer)
        self.assertIsNone(result)
    
    def test_should_checkpoint(self):
        """Test checkpoint scheduling"""
        self.assertTrue(self.manager.should_checkpoint(0))
        self.assertTrue(self.manager.should_checkpoint(10))
        self.assertTrue(self.manager.should_checkpoint(20))
        self.assertFalse(self.manager.should_checkpoint(5))
        self.assertFalse(self.manager.should_checkpoint(15))


class TestElasticTrainingManager(unittest.TestCase):
    """Test ElasticTrainingManager class"""
    
    def setUp(self):
        self.manager = dt.ElasticTrainingManager(
            min_nodes=1,
            max_nodes=4,
            node_discovery_interval=0.1,  # Short interval for testing
            scaling_cooldown=0.5,  # Short cooldown for testing
        )
    
    def tearDown(self):
        self.manager.stop_discovery()
    
    def test_initialization(self):
        """Test manager initialization"""
        self.assertEqual(self.manager.min_nodes, 1)
        self.assertEqual(self.manager.max_nodes, 4)
        self.assertEqual(self.manager.node_discovery_interval, 0.1)
        self.assertEqual(self.manager.scaling_cooldown, 0.5)
        self.assertEqual(len(self.manager._active_nodes), 0)
        self.assertIsNone(self.manager._discovery_thread)
    
    def test_start_stop_discovery(self):
        """Test starting and stopping discovery"""
        self.assertIsNone(self.manager._discovery_thread)
        
        self.manager.start_discovery()
        self.assertIsNotNone(self.manager._discovery_thread)
        self.assertTrue(self.manager._discovery_thread.is_alive())
        
        # Start again should not create new thread
        old_thread = self.manager._discovery_thread
        self.manager.start_discovery()
        self.assertEqual(self.manager._discovery_thread, old_thread)
        
        self.manager.stop_discovery()
        # Give thread time to finish
        time.sleep(0.2)
        self.assertIsNone(self.manager._discovery_thread)
    
    def test_scale_up_down(self):
        """Test scaling operations"""
        import time
        
        # Test scale up
        self.manager.scale_up(3)
        self.assertGreater(self.manager._last_scaling_time, 0)
        
        # Test scale up beyond max
        old_time = self.manager._last_scaling_time
        time.sleep(0.001)  # Small delay to ensure different timestamps
        self.manager.scale_up(10)  # Should be capped at max_nodes=4
        self.assertGreater(self.manager._last_scaling_time, old_time)
        
        # Test scale down
        old_time = self.manager._last_scaling_time
        time.sleep(0.001)  # Small delay to ensure different timestamps
        self.manager.scale_down(2)
        self.assertGreater(self.manager._last_scaling_time, old_time)
        
        # Test scale down below min
        old_time = self.manager._last_scaling_time
        time.sleep(0.001)  # Small delay to ensure different timestamps
        self.manager.scale_down(0)  # Should be capped at min_nodes=1
        self.assertGreater(self.manager._last_scaling_time, old_time)


class TestDistributedTrainer(unittest.TestCase):
    """Test DistributedTrainer class"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mock_model = Mock()
        self.mock_optimizer = Mock()
        self.config = dt.DistributedConfig(world_size=1, rank=0, local_rank=0)
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test trainer initialization"""
        trainer = dt.DistributedTrainer(
            model=self.mock_model,
            optimizer=self.mock_optimizer,
            config=self.config,
            checkpoint_dir=self.temp_dir,
            fault_tolerance=True,
            elastic_training=True,
        )
        
        self.assertEqual(trainer.config, self.config)
        self.assertEqual(trainer.original_model, self.mock_model)
        self.assertEqual(trainer.optimizer, self.mock_optimizer)
        self.assertIsNotNone(trainer.fault_tolerance_manager)
        self.assertIsNotNone(trainer.elastic_manager)
    
    def test_initialization_without_optional_features(self):
        """Test trainer initialization without optional features"""
        trainer = dt.DistributedTrainer(
            model=self.mock_model,
            optimizer=self.mock_optimizer,
            config=self.config,
            fault_tolerance=False,
            elastic_training=False,
        )
        
        self.assertIsNone(trainer.fault_tolerance_manager)
        self.assertIsNone(trainer.elastic_manager)
    
    @patch('trustformers.distributed_training.TORCH_AVAILABLE', True)
    @patch('trustformers.distributed_training.DDPModelWrapper')
    @patch('trustformers.distributed_training.torch')
    def test_setup_ddp(self, mock_torch, mock_ddp_wrapper):
        """Test trainer setup with DDP"""
        mock_torch.cuda.is_available.return_value = True
        mock_wrapper_instance = Mock()
        mock_ddp_wrapper.return_value = mock_wrapper_instance
        
        trainer = dt.DistributedTrainer(
            model=self.mock_model,
            optimizer=self.mock_optimizer,
            config=self.config,
        )
        
        with patch.object(trainer.dist_manager, 'initialize'):
            trainer.setup()
        
        self.assertEqual(trainer.model_wrapper, mock_wrapper_instance)
        mock_ddp_wrapper.assert_called_once_with(
            self.mock_model,
            device_ids=[0],
            find_unused_parameters=False,
            bucket_cap_mb=25,
            gradient_as_bucket_view=True,
        )
    
    @patch('trustformers.distributed_training.HOROVOD_AVAILABLE', True)
    @patch('trustformers.distributed_training.HorovodModelWrapper')
    def test_setup_horovod(self, mock_hvd_wrapper):
        """Test trainer setup with Horovod"""
        self.config.backend = dt.DistributedBackend.HOROVOD
        mock_wrapper_instance = Mock()
        mock_hvd_wrapper.return_value = mock_wrapper_instance
        
        trainer = dt.DistributedTrainer(
            model=self.mock_model,
            optimizer=self.mock_optimizer,
            config=self.config,
        )
        
        with patch.object(trainer.dist_manager, 'initialize'):
            trainer.setup()
        
        self.assertEqual(trainer.model_wrapper, mock_wrapper_instance)
        mock_hvd_wrapper.assert_called_once_with(self.mock_model, self.mock_optimizer)
    
    def test_distributed_context(self):
        """Test distributed context manager"""
        trainer = dt.DistributedTrainer(
            model=self.mock_model,
            optimizer=self.mock_optimizer,
            config=self.config,
        )
        
        with patch.object(trainer, 'setup') as mock_setup, \
             patch.object(trainer, 'cleanup') as mock_cleanup:
            
            with trainer.distributed_context() as ctx:
                self.assertEqual(ctx, trainer)
            
            mock_setup.assert_called_once()
            mock_cleanup.assert_called_once()


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_get_free_port(self):
        """Test getting a free port"""
        port = dt.get_free_port()
        self.assertIsInstance(port, int)
        self.assertGreater(port, 0)
        self.assertLess(port, 65536)
    
    def test_setup_logging_for_distributed(self):
        """Test logging setup"""
        # This should not raise an exception
        dt.setup_logging_for_distributed(rank=0)
        dt.setup_logging_for_distributed(rank=1)
    
    @patch('trustformers.distributed_training.TORCH_AVAILABLE', False)
    def test_utility_functions_without_torch(self):
        """Test utility functions without PyTorch"""
        obj = {'test': 'data'}
        
        # all_gather_object should return single object list
        result = dt.all_gather_object(obj, 4)
        self.assertEqual(result, [obj])
        
        # broadcast_object should return the same object
        result = dt.broadcast_object(obj)
        self.assertEqual(result, obj)
        
        # barrier should not raise
        dt.barrier()
        
        # Global functions should return defaults
        self.assertFalse(dt.is_distributed())
        self.assertEqual(dt.get_rank(), 0)
        self.assertEqual(dt.get_world_size(), 1)
        self.assertTrue(dt.is_main_process())
    
    @patch('trustformers.distributed_training.TORCH_AVAILABLE', True)
    @patch('trustformers.distributed_training.dist')
    def test_reduce_tensor_without_dist(self, mock_dist):
        """Test tensor reduction without distributed backend"""
        mock_dist.is_initialized.return_value = False
        
        # Mock a simple tensor
        mock_tensor = Mock()
        result = dt.reduce_tensor(mock_tensor)
        self.assertEqual(result, mock_tensor)
    
    def test_get_default_distributed_manager(self):
        """Test getting default distributed manager"""
        manager1 = dt.get_default_distributed_manager()
        manager2 = dt.get_default_distributed_manager()
        
        # Should return the same instance
        self.assertEqual(manager1, manager2)
        self.assertIsInstance(manager1, dt.DistributedManager)


class TestDistributedBackend(unittest.TestCase):
    """Test DistributedBackend constants"""
    
    def test_backend_constants(self):
        """Test that backend constants are correctly defined"""
        self.assertEqual(dt.DistributedBackend.NCCL, "nccl")
        self.assertEqual(dt.DistributedBackend.GLOO, "gloo") 
        self.assertEqual(dt.DistributedBackend.MPI, "mpi")
        self.assertEqual(dt.DistributedBackend.HOROVOD, "horovod")


if __name__ == '__main__':
    # Run tests with high verbosity
    unittest.main(verbosity=2)