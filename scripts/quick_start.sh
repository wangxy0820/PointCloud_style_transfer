#!/bin/bash

# Quick Start Script for Point Cloud Style Transfer
# This script provides a complete workflow from data preparation to training

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Default parameters
SIM_DIR="datasets/simulation"
REAL_DIR="datasets/real_world"
PROCESSED_DIR="datasets/processed"
EXPERIMENT_NAME="quickstart_$(date +%Y%m%d_%H%M%S)"
CONFIG_TYPE="quick_test"
SKIP_PREPROCESSING=false
SKIP_TRAINING=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sim-dir)
            SIM_DIR="$2"
            shift 2
            ;;
        --real-dir)
            REAL_DIR="$2"
            shift 2
            ;;
        --processed-dir)
            PROCESSED_DIR="$2"
            shift 2
            ;;
        --experiment-name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --config-type)
            CONFIG_TYPE="$2"
            shift 2
            ;;
        --skip-preprocessing)
            SKIP_PREPROCESSING=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --help)
            echo "Quick Start Script for Point Cloud Style Transfer"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --sim-dir DIR              Simulation data directory (default: datasets/simulation)"
            echo "  --real-dir DIR             Real world data directory (default: datasets/real_world)"
            echo "  --processed-dir DIR        Processed data directory (default: datasets/processed)"
            echo "  --experiment-name NAME     Experiment name (default: quickstart_TIMESTAMP)"
            echo "  --config-type TYPE         Configuration type (default: quick_test)"
            echo "                             Options: quick_test, standard, high_quality, fast_training"
            echo "  --skip-preprocessing       Skip data preprocessing step"
            echo "  --skip-training           Skip training step"
            echo "  --help                    Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --config-type standard --experiment-name my_experiment"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "Point Cloud Style Transfer - Quick Start"
echo "============================================"
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    print_warning "Virtual environment not detected. Activating..."
    if [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
        print_success "Virtual environment activated"
    else
        print_error "Virtual environment not found. Please run setup.sh first."
        exit 1
    fi
fi

# Verify installation
print_status "Verifying installation..."
python -c "
try:
    import torch
    import numpy as np
    from config.config import Config
    print('✓ All dependencies available')
except ImportError as e:
    print(f'✗ Missing dependency: {e}')
    exit(1)
"

# Check data directories
if [[ "$SKIP_PREPROCESSING" == false ]]; then
    print_status "Checking data directories..."
    
    if [[ ! -d "$SIM_DIR" ]]; then
        print_error "Simulation data directory not found: $SIM_DIR"
        print_status "Creating directory: $SIM_DIR"
        mkdir -p "$SIM_DIR"
        print_warning "Please place your simulation .npy files in $SIM_DIR"
    fi
    
    if [[ ! -d "$REAL_DIR" ]]; then
        print_error "Real world data directory not found: $REAL_DIR"
        print_status "Creating directory: $REAL_DIR"
        mkdir -p "$REAL_DIR"
        print_warning "Please place your real world .npy files in $REAL_DIR"
    fi
    
    # Check if data files exist
    SIM_FILES=$(find "$SIM_DIR" -name "*.npy" | wc -l)
    REAL_FILES=$(find "$REAL_DIR" -name "*.npy" | wc -l)
    
    if [[ $SIM_FILES -eq 0 ]]; then
        print_warning "No .npy files found in $SIM_DIR"
        print_status "Creating sample data for demonstration..."
        python -c "
import numpy as np
import os
os.makedirs('$SIM_DIR', exist_ok=True)
for i in range(5):
    # Create sample point clouds
    points = np.random.randn(8192, 3).astype(np.float32)
    np.save(f'$SIM_DIR/sim_{i:03d}.npy', points)
print('Sample simulation data created')
"
        SIM_FILES=5
    fi
    
    if [[ $REAL_FILES -eq 0 ]]; then
        print_warning "No .npy files found in $REAL_DIR"
        print_status "Creating sample data for demonstration..."
        python -c "
import numpy as np
import os
os.makedirs('$REAL_DIR', exist_ok=True)
for i in range(5):
    # Create sample point clouds with different distribution
    points = (np.random.randn(8192, 3) * 0.5 + np.random.randn(3)).astype(np.float32)
    np.save(f'$REAL_DIR/real_{i:03d}.npy', points)
print('Sample real world data created')
"
        REAL_FILES=5
    fi
    
    print_success "Found $SIM_FILES simulation files and $REAL_FILES real world files"
fi

# Data preprocessing
if [[ "$SKIP_PREPROCESSING" == false ]]; then
    print_status "Starting data preprocessing..."
    
    python data/preprocess.py \
        --sim_dir "$SIM_DIR" \
        --real_dir "$REAL_DIR" \
        --output_dir "$PROCESSED_DIR" \
        --chunk_size 8192 \
        --chunk_method spatial
    
    if [[ $? -eq 0 ]]; then
        print_success "Data preprocessing completed"
    else
        print_error "Data preprocessing failed"
        exit 1
    fi
else
    print_status "Skipping data preprocessing"
    
    # Check if processed data exists
    if [[ ! -f "$PROCESSED_DIR/dataset_splits.pkl" ]]; then
        print_error "Processed data not found. Please run preprocessing first or remove --skip-preprocessing flag."
        exit 1
    fi
fi

# Training
if [[ "$SKIP_TRAINING" == false ]]; then
    print_status "Starting training with configuration: $CONFIG_TYPE"
    
    # Create experiment directory
    EXPERIMENT_DIR="experiments/$EXPERIMENT_NAME"
    mkdir -p "$EXPERIMENT_DIR"
    
    # Determine training parameters based on config type
    case $CONFIG_TYPE in
        quick_test)
            EPOCHS=5
            BATCH_SIZE=2
            EVAL_INTERVAL=2
            ;;
        standard)
            EPOCHS=200
            BATCH_SIZE=8
            EVAL_INTERVAL=5
            ;;
        high_quality)
            EPOCHS=500
            BATCH_SIZE=4
            EVAL_INTERVAL=10
            ;;
        fast_training)
            EPOCHS=100
            BATCH_SIZE=16
            EVAL_INTERVAL=5
            ;;
        *)
            print_warning "Unknown config type: $CONFIG_TYPE. Using standard settings."
            EPOCHS=200
            BATCH_SIZE=8
            EVAL_INTERVAL=5
            ;;
    esac
    
    print_status "Training parameters:"
    echo "  - Epochs: $EPOCHS"
    echo "  - Batch size: $BATCH_SIZE"
    echo "  - Evaluation interval: $EVAL_INTERVAL"
    echo "  - Experiment directory: $EXPERIMENT_DIR"
    
    # Start training
    python scripts/train.py \
        --data_dir "$PROCESSED_DIR" \
        --batch_size $BATCH_SIZE \
        --num_epochs $EPOCHS \
        --eval_interval $EVAL_INTERVAL \
        --experiment_name "$EXPERIMENT_NAME" \
        --use_augmentation \
        --save_interval 10
    
    if [[ $? -eq 0 ]]; then
        print_success "Training completed successfully!"
        
        # Check if model was saved
        if [[ -f "$EXPERIMENT_DIR/checkpoints/best_model.pth" ]]; then
            print_success "Best model saved at: $EXPERIMENT_DIR/checkpoints/best_model.pth"
        fi
        
        # Show TensorBoard command
        print_status "To view training progress:"
        echo "  tensorboard --logdir $EXPERIMENT_DIR/logs"
        
    else
        print_error "Training failed"
        exit 1
    fi
else
    print_status "Skipping training"
fi

# Quick evaluation (if model exists)
if [[ -f "experiments/$EXPERIMENT_NAME/checkpoints/best_model.pth" ]]; then
    print_status "Running quick evaluation..."
    
    python scripts/test.py \
        --model_path "experiments/$EXPERIMENT_NAME/checkpoints/best_model.pth" \
        --data_dir "$PROCESSED_DIR" \
        --output_dir "experiments/$EXPERIMENT_NAME/evaluation" \
        --compute_all_metrics \
        --save_visualizations \
        --num_vis_samples 3
    
    if [[ $? -eq 0 ]]; then
        print_success "Evaluation completed"
        print_status "Results saved in: experiments/$EXPERIMENT_NAME/evaluation"
    else
        print_warning "Evaluation failed, but training was successful"
    fi
fi

# Final summary
echo ""
echo "============================================"
print_success "Quick start completed!"
echo "============================================"
echo ""
echo "Summary:"
echo "  - Experiment name: $EXPERIMENT_NAME"
echo "  - Configuration: $CONFIG_TYPE"
echo "  - Data directory: $PROCESSED_DIR"
echo "  - Results directory: experiments/$EXPERIMENT_NAME"
echo ""
echo "Next steps:"
echo "1. View training progress:"
echo "   tensorboard --logdir experiments/$EXPERIMENT_NAME/logs"
echo ""
echo "2. Run inference on new data:"
echo "   python scripts/inference.py \\"
echo "     --model_path experiments/$EXPERIMENT_NAME/checkpoints/best_model.pth \\"
echo "     --input_dir your_new_sim_data \\"
echo "     --output_dir generated_results"
echo ""
echo "3. Create visualizations:"
echo "   python scripts/visualize.py \\"
echo "     --input_dir experiments/$EXPERIMENT_NAME/evaluation \\"
echo "     --mode style_transfer"
echo ""
print_success "Happy experimenting!"