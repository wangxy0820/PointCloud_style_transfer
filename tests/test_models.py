#!/usr/bin/env python3
"""
Unit tests for point cloud models
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pointnet2 import PointNet2AutoEncoder, PointNet2Encoder, PointNet2Decoder
from models.generator import PointCloudGenerator, CycleConsistentGenerator
from models.discriminator import PointCloudDiscriminator, HybridDiscriminator
from models.losses import ChamferLoss, EMDLoss, AdversarialLoss


class TestPointNet2Models:
    """Test PointNet++ models"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def sample_data(self, device):
        batch_size = 2
        num_points = 1024
        return torch.randn(batch_size, num_points, 3).to(device)
    
    def test_pointnet2_encoder(self, device, sample_data):
        """Test PointNet++ encoder"""
        encoder = PointNet2Encoder(
            input_channels=3,
            output_dim=256,
            k=20
        ).to(device)
        
        encoder.eval()
        with torch.no_grad():
            local_features, global_features = encoder(sample_data)
        
        # Check output shapes
        assert local_features.shape[0] == sample_data.shape[0]  # Batch size
        assert local_features.shape[1] == 256  # Feature dimension
        assert global_features.shape == (sample_data.shape[0], 256)
        
        # Check that features are not all zeros
        assert torch.sum(torch.abs(local_features)) > 0
        assert torch.sum(torch.abs(global_features)) > 0
    
    def test_pointnet2_decoder(self, device):
        """Test PointNet++ decoder"""
        batch_size = 2
        num_points = 1024
        feature_dim = 256
        
        decoder = PointNet2Decoder(
            input_dim=feature_dim,
            output_channels=3,
            num_points=num_points
        ).to(device)
        
        # Create dummy features
        local_features = torch.randn(batch_size, feature_dim, num_points).to(device)
        global_features = torch.randn(batch_size, feature_dim).to(device)
        
        decoder.eval()
        with torch.no_grad():
            output = decoder(local_features, global_features)
        
        # Check output shape
        assert output.shape == (batch_size, num_points, 3)
        
        # Check that output is not all zeros
        assert torch.sum(torch.abs(output)) > 0
    
    def test_pointnet2_autoencoder(self, device, sample_data):
        """Test PointNet++ autoencoder"""
        autoencoder = PointNet2AutoEncoder(
            input_channels=3,
            latent_dim=256,
            num_points=sample_data.shape[1]
        ).to(device)
        
        autoencoder.eval()
        with torch.no_grad():
            reconstructed, local_features, global_features = autoencoder(sample_data)
        
        # Check output shapes
        assert reconstructed.shape == sample_data.shape
        assert local_features.shape[0] == sample_data.shape[0]
        assert global_features.shape == (sample_data.shape[0], 256)
        
        # Check that reconstruction is reasonable
        assert torch.sum(torch.abs(reconstructed)) > 0


class TestGeneratorModels:
    """Test generator models"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def sample_data(self, device):
        batch_size = 2
        num_points = 1024
        content_points = torch.randn(batch_size, num_points, 3).to(device)
        style_points = torch.randn(batch_size, num_points, 3).to(device)
        return content_points, style_points
    
    def test_pointcloud_generator(self, device, sample_data):
        """Test point cloud generator"""
        content_points, style_points = sample_data
        
        generator = PointCloudGenerator(
            input_channels=3,
            style_dim=128,
            latent_dim=256,
            num_points=content_points.shape[1]
        ).to(device)
        
        generator.eval()
        with torch.no_grad():
            generated = generator(content_points, style_points)
        
        # Check output shape
        assert generated.shape == content_points.shape
        
        # Check that output is not all zeros
        assert torch.sum(torch.abs(generated)) > 0
    
    def test_cycle_consistent_generator(self, device, sample_data):
        """Test cycle consistent generator"""
        sim_points, real_points = sample_data
        
        generator = CycleConsistentGenerator(
            input_channels=3,
            style_dim=128,
            latent_dim=256,
            num_points=sim_points.shape[1]
        ).to(device)
        
        generator.eval()
        with torch.no_grad():
            # Test forward pass
            fake_real, fake_sim = generator(sim_points, real_points)
            
            # Test cycle consistency
            cycled_sim, cycled_real = generator.cycle_forward(sim_points, real_points)
        
        # Check output shapes
        assert fake_real.shape == sim_points.shape
        assert fake_sim.shape == real_points.shape
        assert cycled_sim.shape == sim_points.shape
        assert cycled_real.shape == real_points.shape
        
        # Check that outputs are not all zeros
        assert torch.sum(torch.abs(fake_real)) > 0
        assert torch.sum(torch.abs(fake_sim)) > 0


class TestDiscriminatorModels:
    """Test discriminator models"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def sample_data(self, device):
        batch_size = 2
        num_points = 1024
        return torch.randn(batch_size, num_points, 3).to(device)
    
    def test_pointcloud_discriminator(self, device, sample_data):
        """Test point cloud discriminator"""
        discriminator = PointCloudDiscriminator(
            input_channels=3,
            feature_channels=[32, 64, 128, 256]
        ).to(device)
        
        discriminator.eval()
        with torch.no_grad():
            output, features = discriminator(sample_data)
        
        # Check output shapes
        assert output.shape == (sample_data.shape[0], 1)
        assert features.shape[0] == sample_data.shape[0]
        
        # Check that outputs are not all zeros
        assert torch.sum(torch.abs(output)) > 0
        assert torch.sum(torch.abs(features)) > 0
    
    def test_hybrid_discriminator(self, device, sample_data):
        """Test hybrid discriminator"""
        discriminator = HybridDiscriminator(
            input_channels=3,
            patch_size=256
        ).to(device)
        
        discriminator.eval()
        with torch.no_grad():
            global_output, global_features, patch_outputs = discriminator(sample_data)
        
        # Check output shapes
        assert global_output.shape == (sample_data.shape[0], 1)
        assert global_features.shape[0] == sample_data.shape[0]
        assert patch_outputs.shape[0] == sample_data.shape[0]
        
        # Check that outputs are not all zeros
        assert torch.sum(torch.abs(global_output)) > 0
        assert torch.sum(torch.abs(patch_outputs)) > 0


class TestLossFunctions:
    """Test loss functions"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def sample_point_clouds(self, device):
        batch_size = 2
        num_points = 512
        pred = torch.randn(batch_size, num_points, 3).to(device)
        target = torch.randn(batch_size, num_points, 3).to(device)
        return pred, target
    
    def test_chamfer_loss(self, device, sample_point_clouds):
        """Test Chamfer distance loss"""
        pred, target = sample_point_clouds
        
        loss_fn = ChamferLoss(use_sqrt=False)
        loss = loss_fn(pred, target)
        
        # Check that loss is a scalar and positive
        assert loss.dim() == 0
        assert loss.item() >= 0
        
        # Test with sqrt
        loss_fn_sqrt = ChamferLoss(use_sqrt=True)
        loss_sqrt = loss_fn_sqrt(pred, target)
        assert loss_sqrt.dim() == 0
        assert loss_sqrt.item() >= 0
    
    def test_emd_loss(self, device, sample_point_clouds):
        """Test Earth Mover's Distance loss"""
        pred, target = sample_point_clouds
        
        loss_fn = EMDLoss()
        loss = loss_fn(pred, target)
        
        # Check that loss is a scalar and positive
        assert loss.dim() == 0
        assert loss.item() >= 0
    
    def test_adversarial_loss(self, device):
        """Test adversarial loss"""
        batch_size = 2
        discriminator_output = torch.randn(batch_size, 1).to(device)
        
        loss_fn = AdversarialLoss('lsgan')
        
        # Test real target
        real_loss = loss_fn(discriminator_output, target_is_real=True)
        assert real_loss.dim() == 0
        assert real_loss.item() >= 0
        
        # Test fake target
        fake_loss = loss_fn(discriminator_output, target_is_real=False)
        assert fake_loss.dim() == 0
        assert fake_loss.item() >= 0


class TestModelIntegration:
    """Integration tests for model combinations"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_generator_discriminator_integration(self, device):
        """Test generator and discriminator working together"""
        batch_size = 2
        num_points = 512
        
        # Create models
        generator = CycleConsistentGenerator(
            input_channels=3,
            style_dim=64,
            latent_dim=128,
            num_points=num_points
        ).to(device)
        
        discriminator = PointCloudDiscriminator(
            input_channels=3,
            feature_channels=[32, 64, 128]
        ).to(device)
        
        # Create sample data
        sim_points = torch.randn(batch_size, num_points, 3).to(device)
        real_points = torch.randn(batch_size, num_points, 3).to(device)
        
        # Test forward pass
        generator.eval()
        discriminator.eval()
        
        with torch.no_grad():
            # Generate fake data
            fake_real, fake_sim = generator(sim_points, real_points)
            
            # Discriminate real and fake data
            real_output, _ = discriminator(real_points)
            fake_output, _ = discriminator(fake_real)
        
        # Check shapes
        assert fake_real.shape == sim_points.shape
        assert fake_sim.shape == real_points.shape
        assert real_output.shape == (batch_size, 1)
        assert fake_output.shape == (batch_size, 1)
        
        # Check that discriminator gives different outputs for real and fake
        # (This is a basic sanity check, not a guarantee of performance)
        assert not torch.allclose(real_output, fake_output, atol=1e-6)


# Test configuration and utilities
class TestConfiguration:
    """Test configuration and utilities"""
    
    def test_config_import(self):
        """Test that configuration can be imported"""
        from config.config import Config
        
        config = Config()
        
        # Check that essential attributes exist
        assert hasattr(config, 'chunk_size')
        assert hasattr(config, 'batch_size')
        assert hasattr(config, 'learning_rate_g')
        assert hasattr(config, 'learning_rate_d')
        
        # Check reasonable default values
        assert config.chunk_size > 0
        assert config.batch_size > 0
        assert 0 < config.learning_rate_g < 1
        assert 0 < config.learning_rate_d < 1


# Fixtures for parameterized testing
@pytest.fixture(params=[512, 1024, 2048])
def different_point_counts(request):
    """Test with different point cloud sizes"""
    return request.param


@pytest.fixture(params=[1, 2, 4])
def different_batch_sizes(request):
    """Test with different batch sizes"""
    return request.param


# Performance and memory tests
class TestModelPerformance:
    """Test model performance and memory usage"""
    
    @pytest.mark.slow
    def test_memory_usage(self):
        """Test that models don't use excessive memory"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device('cuda')
        
        # Clear cache
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Create and test models
        batch_size = 4
        num_points = 2048
        
        generator = CycleConsistentGenerator(
            input_channels=3,
            style_dim=256,
            latent_dim=512,
            num_points=num_points
        ).to(device)
        
        test_data = torch.randn(batch_size, num_points, 3).to(device)
        
        with torch.no_grad():
            output = generator(test_data, test_data)
        
        final_memory = torch.cuda.memory_allocated()
        memory_used = final_memory - initial_memory
        
        # Check that memory usage is reasonable (less than 2GB for this test)
        assert memory_used < 2 * 1024**3, f"Memory usage too high: {memory_used / 1024**3:.2f} GB"


if __name__ == "__main__":
    # Run specific test
    pytest.main([__file__, "-v"])