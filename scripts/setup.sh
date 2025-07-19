#!/bin/bash

# Point Cloud Style Transfer Project Setup Script
# This script sets up the development environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Print banner
echo "=========================================="
echo "Point Cloud Style Transfer Setup"
echo "=========================================="

# Check Python version
print_status "Checking Python version..."
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python $PYTHON_VERSION found"
    
    # Check if Python version is >= 3.8
    if python3 -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)"; then
        print_success "Python version is compatible (>= 3.8)"
    else
        print_error "Python version must be >= 3.8. Current version: $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python3 not found. Please install Python 3.8 or later."
    exit 1
fi

# Check for CUDA
print_status "Checking CUDA availability..."
if command_exists nvcc; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
    print_success "CUDA $CUDA_VERSION found"
else
    print_warning "CUDA not found. GPU acceleration will not be available."
fi

# Check for Git
print_status "Checking Git..."
if command_exists git; then
    print_success "Git found"
else
    print_error "Git not found. Please install Git."
    exit 1
fi

# Create virtual environment
print_status "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip
print_success "Pip upgraded"

# Install PyTorch with CUDA support if available
print_status "Installing PyTorch..."
if command_exists nvcc; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    print_success "PyTorch with CUDA support installed"
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    print_success "PyTorch CPU-only installed"
fi

# Install requirements
print_status "Installing requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_success "Requirements installed"
else
    print_error "requirements.txt not found"
    exit 1
fi

# Install project in development mode
print_status "Installing project in development mode..."
pip install -e .
print_success "Project installed"

# Create necessary directories
print_status "Creating project directories..."
mkdir -p datasets/{simulation,real_world,processed}
mkdir -p checkpoints
mkdir -p logs
mkdir -p results
mkdir -p experiments
print_success "Directories created"

# Set up pre-commit hooks (optional)
if command_exists pre-commit; then
    print_status "Setting up pre-commit hooks..."
    pre-commit install
    print_success "Pre-commit hooks installed"
else
    print_warning "pre-commit not found. Skipping pre-commit setup."
fi

# Create sample config
print_status "Creating sample configuration..."
cat > config/local_config.py << 'EOF'
# Local configuration file
# Copy this file and modify for your specific setup

from .config import Config

class LocalConfig(Config):
    """Local development configuration"""
    
    # Override settings for your local environment
    data_root = "datasets"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4
    
    # Adjust these based on your hardware
    batch_size = 8
    chunk_size = 8192
    
    # Development settings
    log_interval = 10
    save_interval = 5
    eval_interval = 2

# Example usage:
# config = LocalConfig()
EOF
print_success "Sample configuration created"

# Test installation
print_status "Testing installation..."
python -c "
import torch
import numpy as np
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
print('Installation test passed!')
"
print_success "Installation test completed"

# Print final instructions
echo ""
echo "=========================================="
print_success "Setup completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Place your point cloud data in:"
echo "   - datasets/simulation/ (for simulation data)"
echo "   - datasets/real_world/ (for real world data)"
echo "3. Preprocess your data:"
echo "   python data/preprocess.py --sim_dir datasets/simulation --real_dir datasets/real_world --output_dir datasets/processed"
echo "4. Start training:"
echo "   python scripts/train.py --data_dir datasets/processed"
echo ""
echo "For more information, see the README.md file."
echo ""

# Deactivate virtual environment
deactivate