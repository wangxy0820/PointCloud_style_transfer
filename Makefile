# Point Cloud Style Transfer Project Makefile

.PHONY: help setup install clean test lint format train inference docker-build docker-run docs

# Default target
help:
	@echo "Point Cloud Style Transfer - Available Commands"
	@echo "=============================================="
	@echo ""
	@echo "Setup and Installation:"
	@echo "  setup              Set up development environment"
	@echo "  install            Install project dependencies"
	@echo "  install-dev        Install development dependencies"
	@echo "  clean              Clean build artifacts and cache"
	@echo ""
	@echo "Development:"
	@echo "  test               Run unit tests"
	@echo "  lint               Run code linting"
	@echo "  format             Format code with black and isort"
	@echo "  type-check         Run type checking with mypy"
	@echo ""
	@echo "Data and Training:"
	@echo "  preprocess         Preprocess point cloud data"
	@echo "  train              Start training with default config"
	@echo "  train-quick        Quick training for testing"
	@echo "  train-quality      High quality training"
	@echo "  inference          Run inference on sample data"
	@echo "  benchmark          Run model benchmarks"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build       Build Docker image"
	@echo "  docker-run         Run container interactively"
	@echo "  docker-train       Run training in Docker"
	@echo "  docker-jupyter     Start Jupyter notebook in Docker"
	@echo "  docker-tensorboard Start TensorBoard in Docker"
	@echo ""
	@echo "Utilities:"
	@echo "  check-data         Check data quality"
	@echo "  visualize          Create visualizations"
	@echo "  docs               Generate documentation"
	@echo "  demo               Run complete demo pipeline"

# Variables
PYTHON := python3
PIP := pip
DOCKER := docker
DOCKER_COMPOSE := docker-compose

# Project directories
SIM_DIR := datasets/simulation
REAL_DIR := datasets/real_world
PROCESSED_DIR := datasets/processed
RESULTS_DIR := results

# Setup and Installation
setup:
	@echo "Setting up development environment..."
	chmod +x scripts/setup.sh
	./scripts/setup.sh

install:
	@echo "Installing project dependencies..."
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev:
	@echo "Installing development dependencies..."
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"
	pre-commit install

clean:
	@echo "Cleaning build artifacts..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/
	rm -rf $(RESULTS_DIR)/temp/ logs/temp/

# Development
test:
	@echo "Running tests..."
	pytest tests/ -v --cov=./ --cov-report=html

lint:
	@echo "Running linting..."
	flake8 --config setup.cfg
	black --check .
	isort --check-only .

format:
	@echo "Formatting code..."
	black .
	isort .

type-check:
	@echo "Running type checking..."
	mypy --config-file setup.cfg .

# Data processing
preprocess:
	@echo "Preprocessing point cloud data..."
	@mkdir -p $(PROCESSED_DIR)
	$(PYTHON) data/preprocess.py \
		--sim_dir $(SIM_DIR) \
		--real_dir $(REAL_DIR) \
		--output_dir $(PROCESSED_DIR) \
		--chunk_size 8192 \
		--chunk_method spatial

check-data:
	@echo "Checking data quality..."
	@mkdir -p $(RESULTS_DIR)/data_quality
	$(PYTHON) scripts/check_data_quality.py \
		--data_dir datasets \
		--output_dir $(RESULTS_DIR)/data_quality \
		--check_duplicates \
		--check_outliers \
		--save_plots

# Training
train:
	@echo "Starting training with standard configuration..."
	$(PYTHON) scripts/train.py \
		--data_dir $(PROCESSED_DIR) \
		--batch_size 8 \
		--num_epochs 200 \
		--experiment_name standard_training \
		--use_augmentation

train-quick:
	@echo "Starting quick training for testing..."
	$(PYTHON) scripts/train.py \
		--data_dir $(PROCESSED_DIR) \
		--batch_size 2 \
		--num_epochs 5 \
		--experiment_name quick_test \
		--eval_interval 2

train-quality:
	@echo "Starting high quality training..."
	$(PYTHON) scripts/train.py \
		--data_dir $(PROCESSED_DIR) \
		--batch_size 4 \
		--num_epochs 500 \
		--experiment_name high_quality \
		--chunk_size 10240 \
		--latent_dim 1024 \
		--lambda_recon 15.0 \
		--lambda_cycle 10.0

# Inference and evaluation
inference:
	@echo "Running inference..."
	@mkdir -p $(RESULTS_DIR)/inference
	$(PYTHON) scripts/inference.py \
		--model_path experiments/latest/checkpoints/best_model.pth \
		--input_dir $(SIM_DIR) \
		--output_dir $(RESULTS_DIR)/inference \
		--create_visualization

test-model:
	@echo "Testing trained model..."
	@mkdir -p $(RESULTS_DIR)/evaluation
	$(PYTHON) scripts/test.py \
		--model_path experiments/latest/checkpoints/best_model.pth \
		--data_dir $(PROCESSED_DIR) \
		--output_dir $(RESULTS_DIR)/evaluation \
		--compute_all_metrics \
		--save_visualizations

benchmark:
	@echo "Running model benchmarks..."
	@mkdir -p $(RESULTS_DIR)/benchmark
	$(PYTHON) scripts/benchmark.py \
		--benchmark_type all \
		--output_dir $(RESULTS_DIR)/benchmark

# Visualization
visualize:
	@echo "Creating visualizations..."
	@mkdir -p $(RESULTS_DIR)/visualizations
	$(PYTHON) scripts/visualize.py \
		--input_dir $(RESULTS_DIR) \
		--output_dir $(RESULTS_DIR)/visualizations \
		--mode style_transfer \
		--save_html

# Docker commands
docker-build:
	@echo "Building Docker image..."
	$(DOCKER) build -t pointcloud-style-transfer .

docker-run:
	@echo "Running Docker container interactively..."
	$(DOCKER) run -it --gpus all \
		-v $(PWD)/datasets:/app/datasets \
		-v $(PWD)/checkpoints:/app/checkpoints \
		-v $(PWD)/results:/app/results \
		pointcloud-style-transfer /bin/bash

docker-train:
	@echo "Running training in Docker..."
	$(DOCKER_COMPOSE) --profile training up training

docker-jupyter:
	@echo "Starting Jupyter notebook in Docker..."
	$(DOCKER_COMPOSE) --profile development up jupyter
	@echo "Jupyter notebook available at http://localhost:8888"

docker-tensorboard:
	@echo "Starting TensorBoard in Docker..."
	$(DOCKER_COMPOSE) --profile monitoring up tensorboard
	@echo "TensorBoard available at http://localhost:6006"

# Complete demo pipeline
demo:
	@echo "Running complete demo pipeline..."
	chmod +x scripts/quick_start.sh
	./scripts/quick_start.sh --config-type quick_test --experiment-name demo

# Data generation (for testing without real data)
generate-sample-data:
	@echo "Generating sample data for testing..."
	@mkdir -p $(SIM_DIR) $(REAL_DIR)
	$(PYTHON) -c "
import numpy as np
import os

# Generate sample simulation data
os.makedirs('$(SIM_DIR)', exist_ok=True)
for i in range(10):
    points = np.random.randn(8192, 3).astype(np.float32)
    np.save(f'$(SIM_DIR)/sim_{i:03d}.npy', points)

# Generate sample real world data
os.makedirs('$(REAL_DIR)', exist_ok=True)
for i in range(10):
    # Different distribution to simulate domain gap
    points = (np.random.randn(8192, 3) * 0.5 + np.random.randn(3)).astype(np.float32)
    np.save(f'$(REAL_DIR)/real_{i:03d}.npy', points)

print('Sample data generated successfully!')
"

# Documentation
docs:
	@echo "Generating documentation..."
	@mkdir -p docs
	$(PYTHON) -c "
import os
import sys
sys.path.append('.')

# Generate model documentation
from models import pointnet2, generator, discriminator
import inspect

with open('docs/model_api.md', 'w') as f:
    f.write('# Model API Documentation\n\n')
    
    modules = [pointnet2, generator, discriminator]
    for module in modules:
        f.write(f'## {module.__name__.split(\".\")[-1]}\n\n')
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and obj.__module__ == module.__name__:
                f.write(f'### {name}\n\n')
                if obj.__doc__:
                    f.write(f'{obj.__doc__}\n\n')

print('Documentation generated in docs/')
"

# Continuous Integration helpers
ci-install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"

ci-test:
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) test

# Quick start for new users
quickstart:
	@echo "Quick start for new users..."
	@echo "1. Setting up environment..."
	$(MAKE) setup
	@echo "2. Generating sample data..."
	$(MAKE) generate-sample-data
	@echo "3. Preprocessing data..."
	$(MAKE) preprocess
	@echo "4. Running quick training..."
	$(MAKE) train-quick
	@echo "Quick start completed! Check results/ directory for outputs."

# Show project status
status:
	@echo "Project Status"
	@echo "=============="
	@echo "Data directories:"
	@echo "  Simulation: $(SIM_DIR) ($(shell find $(SIM_DIR) -name "*.npy" 2>/dev/null | wc -l) files)"
	@echo "  Real world: $(REAL_DIR) ($(shell find $(REAL_DIR) -name "*.npy" 2>/dev/null | wc -l) files)"
	@echo "  Processed: $(PROCESSED_DIR) ($(shell test -f $(PROCESSED_DIR)/dataset_splits.pkl && echo "Ready" || echo "Not processed"))"
	@echo ""
	@echo "Experiments:"
	@echo "  $(shell find experiments -name "*.pth" 2>/dev/null | wc -l) trained models found"
	@echo ""
	@echo "Results:"
	@echo "  $(shell find $(RESULTS_DIR) -type f 2>/dev/null | wc -l) result files"

# Environment information
env-info:
	@echo "Environment Information"
	@echo "======================"
	@echo "Python version: $(shell $(PYTHON) --version)"
	@echo "PyTorch version: $(shell $(PYTHON) -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "CUDA available: $(shell $(PYTHON) -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"
	@echo "GPU count: $(shell $(PYTHON) -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 'Unknown')"
	@echo "Working directory: $(PWD)"