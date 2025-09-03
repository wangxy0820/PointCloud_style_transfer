# Point Cloud Style Transfer - 项目自动化脚本
# ====================================================

# 变量定义
DOCKER_COMPOSE = docker-compose
PROJECT_NAME = pointcloud-style-transfer
CONTAINER_NAME = pointcloud-style-transfer
JUPYTER_CONTAINER = pc-jupyter
TENSORBOARD_CONTAINER = pc-tensorboard

# 数据路径
DATA_ROOT = datasets
SIM_DATA_DIR = $(DATA_ROOT)/simulation
REAL_DATA_DIR = $(DATA_ROOT)/real_world
PROCESSED_DATA_DIR = $(DATA_ROOT)/processed_hierarchical

# 模型路径
CHECKPOINT_DIR = checkpoints
LOG_DIR = logs
RESULT_DIR = results

.PHONY: help setup build up down restart logs shell jupyter tensorboard \
        preprocess train test inference visualize clean status \
        install-deps check-gpu monitor

# 默认目标
help:
	@echo "Point Cloud Style Transfer - Docker Commands"
	@echo "============================================="
	@echo ""
	@echo "🐳 Docker 管理:"
	@echo "  make build         - 构建Docker镜像"
	@echo "  make up            - 启动所有服务"
	@echo "  make down          - 停止所有服务"
	@echo "  make restart       - 重启所有服务"
	@echo "  make status        - 查看服务状态"
	@echo ""
	@echo "📊 服务访问:"
	@echo "  make shell         - 进入主容器shell"
	@echo "  make jupyter       - 启动Jupyter Lab"
	@echo "  make tensorboard   - 启动TensorBoard"
	@echo "  make logs          - 查看实时日志"
	@echo ""
	@echo "🔄 数据处理:"
	@echo "  make preprocess    - 数据预处理"
	@echo "  make setup-dirs    - 创建必需目录"
	@echo ""
	@echo "🚀 模型训练:"
	@echo "  make train         - 开始训练"
	@echo "  make train-resume  - 恢复训练"
	@echo "  make monitor       - 监控训练进度"
	@echo ""
	@echo "🧪 测试与推理:"
	@echo "  make test          - 运行测试"
	@echo "  make inference     - 单文件推理"
	@echo "  make visualize     - 可视化结果"
	@echo ""
	@echo "🔧 系统工具:"
	@echo "  make check-gpu     - 检查GPU状态"
	@echo "  make clean         - 清理临时文件"
	@echo "  make clean-all     - 深度清理"

# ==================== Docker 管理 ====================

# 构建镜像
build:
	@echo "🔨 构建Docker镜像..."
	$(DOCKER_COMPOSE) build --no-cache

# 快速构建（使用缓存）
build-fast:
	@echo "⚡ 快速构建Docker镜像..."
	$(DOCKER_COMPOSE) build

# 启动服务
up:
	@echo "🚀 启动所有服务..."
	$(DOCKER_COMPOSE) up -d
	@echo "✅ 服务已启动!"
	@echo "📊 Jupyter Lab: http://localhost:8888"
	@echo "📈 TensorBoard: http://localhost:6006"

# 停止服务
down:
	@echo "🛑 停止所有服务..."
	$(DOCKER_COMPOSE) down

# 重启服务
restart:
	@echo "🔄 重启服务..."
	$(DOCKER_COMPOSE) restart

# 查看服务状态
status:
	@echo "📋 服务状态:"
	$(DOCKER_COMPOSE) ps

# ==================== 服务访问 ====================

# 进入主容器
shell:
	@echo "🐚 进入主容器..."
	docker exec -it $(CONTAINER_NAME) /bin/bash

# 进入容器并激活conda环境（如果有的话）
shell-root:
	@echo "🔑 以root身份进入容器..."
	docker exec -it --user root $(CONTAINER_NAME) /bin/bash

# 启动Jupyter Lab
jupyter:
	@echo "📊 启动Jupyter Lab..."
	docker exec -d $(CONTAINER_NAME) jupyter lab \
		--ip=0.0.0.0 --port=8888 --no-browser --allow-root \
		--NotebookApp.token='' --NotebookApp.password=''
	@echo "✅ Jupyter Lab 已启动: http://localhost:8888"

# 启动TensorBoard
tensorboard:
	@echo "📈 启动TensorBoard..."
	docker exec -d $(CONTAINER_NAME) tensorboard \
		--logdir=$(LOG_DIR) --host=0.0.0.0 --port=6006
	@echo "✅ TensorBoard 已启动: http://localhost:6006"

# 查看日志
logs:
	@echo "📜 查看实时日志..."
	$(DOCKER_COMPOSE) logs -f

# 查看特定服务日志
logs-main:
	$(DOCKER_COMPOSE) logs -f $(PROJECT_NAME)

logs-jupyter:
	$(DOCKER_COMPOSE) logs -f jupyter

logs-tensorboard:
	$(DOCKER_COMPOSE) logs -f tensorboard

# ==================== 数据处理 ====================

# 创建必需的目录结构
setup-dirs:
	@echo "📁 创建项目目录..."
	mkdir -p $(DATA_ROOT) $(SIM_DATA_DIR) $(REAL_DATA_DIR) $(PROCESSED_DATA_DIR)
	mkdir -p $(CHECKPOINT_DIR) $(LOG_DIR) $(RESULT_DIR)
	mkdir -p $(PROCESSED_DATA_DIR)/train $(PROCESSED_DATA_DIR)/val $(PROCESSED_DATA_DIR)/test

# 数据预处理
preprocess: setup-dirs
	@echo "🔄 开始数据预处理..."
	docker exec -it $(CONTAINER_NAME) python scripts/preprocess_data.py \
		--sim_dir $(SIM_DATA_DIR) \
		--real_dir $(REAL_DATA_DIR) \
		--output_dir $(PROCESSED_DATA_DIR) \
		--total_points 120000 \
		--global_points 30000

# 检查数据
check-data:
	@echo "📊 检查数据集状态..."
	docker exec -it $(CONTAINER_NAME) python -c "\
	import os; \
	dirs = ['$(SIM_DATA_DIR)', '$(REAL_DATA_DIR)', '$(PROCESSED_DATA_DIR)']; \
	for d in dirs: \
		files = len([f for f in os.listdir(d) if f.endswith(('.npy', '.pt'))]) if os.path.exists(d) else 0; \
		print(f'{d}: {files} files')"

# ==================== 模型训练 ====================

# 开始训练
train:
	@echo "🚀 开始模型训练..."
	docker exec -it $(CONTAINER_NAME) python scripts/train.py \
		--experiment_name hierarchical_training

# 恢复训练
train-resume:
	@echo "🔄 恢复训练..."
	docker exec -it $(CONTAINER_NAME) python scripts/train.py \
		--experiment_name hierarchical_training \
		--resume

# 快速训练（用于测试）
train-debug:
	@echo "🐛 调试模式训练..."
	docker exec -it $(CONTAINER_NAME) python scripts/train.py \
		--experiment_name debug_training \
		--batch_size 2

# 监控训练进度
monitor:
	@echo "👁️ 监控训练进度..."
	@echo "📈 TensorBoard: http://localhost:6006"
	@echo "📊 检查最新日志:"
	docker exec -it $(CONTAINER_NAME) tail -f logs/*/latest.log

# ==================== 测试与推理 ====================

# 运行模型测试
test:
	@echo "🧪 运行模型测试..."
	docker exec -it $(CONTAINER_NAME) python scripts/test.py \
		--checkpoint checkpoints/hierarchical_training/best_model.pth \
		--test_data $(PROCESSED_DATA_DIR) \
		--compute_all_metrics \
		--save_visualizations

# 单文件推理示例
inference:
	@echo "🔮 运行推理示例..."
	docker exec -it $(CONTAINER_NAME) python scripts/inference.py \
		--checkpoint checkpoints/hierarchical_training/best_model.pth \
		--source $(SIM_DATA_DIR)/sample.npy \
		--reference $(REAL_DATA_DIR)/sample.npy \
		--output $(RESULT_DIR)/generated.npy \
		--visualize

# 批量推理
inference-batch:
	@echo "🔮 批量推理..."
	docker exec -it $(CONTAINER_NAME) python scripts/batch_inference.py \
		--checkpoint checkpoints/hierarchical_training/best_model.pth \
		--input_dir $(SIM_DATA_DIR) \
		--reference_dir $(REAL_DATA_DIR) \
		--output_dir $(RESULT_DIR)/batch

# 可视化结果
visualize:
	@echo "🎨 可视化结果..."
	docker exec -it $(CONTAINER_NAME) python scripts/visualize_results.py \
		--original $(SIM_DATA_DIR)/sample.npy \
		--generated $(RESULT_DIR)/generated.npy \
		--reference $(REAL_DATA_DIR)/sample.npy \
		--interactive

# ==================== 系统工具 ====================

# 检查GPU状态
check-gpu:
	@echo "🖥️ 检查GPU状态..."
	docker exec -it $(CONTAINER_NAME) nvidia-smi
	@echo ""
	docker exec -it $(CONTAINER_NAME) python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# 检查Python环境
check-env:
	@echo "🐍 检查Python环境..."
	docker exec -it $(CONTAINER_NAME) python -c "\
	import torch, numpy, open3d, matplotlib; \
	print(f'PyTorch: {torch.__version__}'); \
	print(f'NumPy: {numpy.__version__}'); \
	print(f'Open3D: {open3d.__version__}'); \
	print(f'CUDA: {torch.version.cuda}'); \
	print(f'cuDNN: {torch.backends.cudnn.version()}')"

# 性能基准测试
benchmark:
	@echo "⚡ 运行性能基准测试..."
	docker exec -it $(CONTAINER_NAME) python scripts/benchmark.py

# ==================== 清理 ====================

# 清理临时文件
clean:
	@echo "🧹 清理临时文件..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	find . -type f -name "*.log.*" -delete 2>/dev/null || true
	docker system prune -f

# 深度清理（包括Docker镜像和数据）
clean-all: clean
	@echo "🗑️ 深度清理..."
	$(DOCKER_COMPOSE) down -v --remove-orphans
	docker system prune -af --volumes
	@echo "⚠️ 警告: 这将删除所有未使用的Docker资源!"

# 清理日志文件
clean-logs:
	@echo "📜 清理日志文件..."
	find $(LOG_DIR) -name "*.log" -mtime +7 -delete 2>/dev/null || true

# 清理结果文件
clean-results:
	@echo "🗑️ 清理结果文件..."
	rm -rf $(RESULT_DIR)/*
	mkdir -p $(RESULT_DIR)

# ==================== 开发工具 ====================

# 代码格式化
format:
	@echo "🎨 代码格式化..."
	docker exec -it $(CONTAINER_NAME) black .
	docker exec -it $(CONTAINER_NAME) isort .

# 代码检查
lint:
	@echo "🔍 代码检查..."
	docker exec -it $(CONTAINER_NAME) flake8 .
	docker exec -it $(CONTAINER_NAME) mypy .

# 运行所有测试
test-all:
	@echo "🧪 运行所有测试..."
	docker exec -it $(CONTAINER_NAME) pytest tests/ -v --cov=.

# ==================== 快捷方式 ====================

# 一键启动开发环境
dev: up jupyter tensorboard
	@echo "🎯 开发环境已就绪!"

# 完整的工作流程
workflow: setup-dirs preprocess train test
	@echo "✅ 完整工作流程执行完成!"