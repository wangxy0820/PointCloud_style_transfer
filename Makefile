# 项目自动化脚本

.PHONY: help build up down restart logs shell train test clean

# 默认目标
help:
	@echo "Point Cloud Style Transfer - Docker Commands"
	@echo "==========================================="
	@echo "make build    - 构建Docker镜像"
	@echo "make up       - 启动所有服务"
	@echo "make down     - 停止所有服务"
	@echo "make restart  - 重启所有服务"
	@echo "make logs     - 查看日志"
	@echo "make shell    - 进入主容器shell"
	@echo "make train    - 开始训练"
	@echo "make test     - 运行测试"
	@echo "make clean    - 清理临时文件"

# 构建镜像
build:
	docker-compose -f docker/docker-compose.yml build

# 启动服务
up:
	docker-compose -f docker/docker-compose.yml up -d

# 停止服务
down:
	docker-compose -f docker/docker-compose.yml down

# 重启服务
restart:
	docker-compose -f docker/docker-compose.yml restart

# 查看日志
logs:
	docker-compose -f docker/docker-compose.yml logs -f

# 进入容器
shell:
	docker exec -it pointcloud-style-transfer bash

# 训练模型
train:
	docker exec -it pointcloud-style-transfer python scripts/train.py \
		--data_dir datasets/processed \
		--experiment_name docker_experiment

# 运行测试
test:
	docker exec -it pointcloud-style-transfer python scripts/test.py \
		--checkpoint checkpoints/best_model.pth \
		--test_data datasets/processed/test

# 清理
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete
