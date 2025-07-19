import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

from models.generator import CycleConsistentGenerator
from models.discriminator import HybridDiscriminator
from models.losses import StyleTransferLoss, GradientPenalty
from evaluation.metrics import PointCloudMetrics
from visualization.visualize import PointCloudVisualizer


class PointCloudStyleTransferTrainer:
    """点云风格迁移训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        
        # 内存优化设置
        self._setup_memory_optimization()
        
        # 设置日志
        self.setup_logging()
        
        # 初始化模型
        self.setup_models()
        
        # 初始化优化器
        self.setup_optimizers()
        
        # 初始化损失函数
        self.setup_losses()
        
        # 初始化评估器
        self.metrics = PointCloudMetrics()
        self.visualizer = PointCloudVisualizer()
        
        # TensorBoard记录器
        self.writer = SummaryWriter(config.log_dir)
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_score = float('inf')
        
        # 学习率调度器
        self.setup_schedulers()
        
    def _setup_memory_optimization(self):
        """设置内存优化"""
        import os
        
        # 设置CUDA内存分配策略 - 使用更激进的内存管理
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:128'
        
        if torch.cuda.is_available():
            # 启用benchmark但减少内存使用
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # 设置更保守的内存分配比例
            torch.cuda.set_per_process_memory_fraction(0.85)
            
            # 清理缓存
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # 启用自动混合精度的内存优化
            torch.cuda.amp.autocast(enabled=False)  # 暂时禁用，需要时再启用
    
    def _cleanup_memory(self):
        """清理CUDA内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
    
    def _check_memory_usage(self, stage: str = ""):
        """检查内存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            
            if allocated > 10.0:  # 如果使用超过10GB，记录警告
                self.logger.warning(f"High memory usage at {stage}: "
                                  f"Allocated: {allocated:.2f}GB, "
                                  f"Reserved: {reserved:.2f}GB, "
                                  f"Peak: {max_allocated:.2f}GB")
            
            return allocated, reserved, max_allocated
        return 0, 0, 0
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_models(self):
        """初始化模型"""
        # 生成器
        self.generator = CycleConsistentGenerator(
            input_channels=self.config.input_dim,
            feature_channels=self.config.pointnet_channels,
            style_dim=self.config.generator_dim,
            latent_dim=self.config.latent_dim,
            num_points=self.config.chunk_size
        ).to(self.device)
        
        # 判别器
        self.discriminator_real = HybridDiscriminator(
            input_channels=self.config.input_dim,
            feature_channels=self.config.pointnet_channels,
            patch_size=self.config.chunk_size // 8
        ).to(self.device)
        
        self.discriminator_sim = HybridDiscriminator(
            input_channels=self.config.input_dim,
            feature_channels=self.config.pointnet_channels,
            patch_size=self.config.chunk_size // 8
        ).to(self.device)
        
        self.logger.info(f"Models initialized on {self.device}")
        
        # 打印模型参数数量
        gen_params = sum(p.numel() for p in self.generator.parameters())
        disc_params = sum(p.numel() for p in self.discriminator_real.parameters())
        self.logger.info(f"Generator parameters: {gen_params:,}")
        self.logger.info(f"Discriminator parameters: {disc_params:,}")
        
    def setup_optimizers(self):
        """初始化优化器"""
        # 生成器优化器
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=self.config.learning_rate_g,
            betas=(self.config.beta1, self.config.beta2)
        )
        
        # 判别器优化器
        self.optimizer_D_real = optim.Adam(
            self.discriminator_real.parameters(),
            lr=self.config.learning_rate_d,
            betas=(self.config.beta1, self.config.beta2)
        )
        
        self.optimizer_D_sim = optim.Adam(
            self.discriminator_sim.parameters(),
            lr=self.config.learning_rate_d,
            betas=(self.config.beta1, self.config.beta2)
        )
        
    def setup_losses(self):
        """初始化损失函数"""
        self.criterion = StyleTransferLoss(
            lambda_recon=self.config.lambda_recon,
            lambda_adv=self.config.lambda_adv,
            lambda_cycle=self.config.lambda_cycle,
            lambda_identity=self.config.lambda_identity
        )
        
        self.gradient_penalty = GradientPenalty(lambda_gp=10.0)
        
    def setup_schedulers(self):
        """设置学习率调度器"""
        self.scheduler_G = optim.lr_scheduler.StepLR(
            self.optimizer_G, step_size=50, gamma=0.8
        )
        
        self.scheduler_D_real = optim.lr_scheduler.StepLR(
            self.optimizer_D_real, step_size=50, gamma=0.8
        )
        
        self.scheduler_D_sim = optim.lr_scheduler.StepLR(
            self.optimizer_D_sim, step_size=50, gamma=0.8
        )
        
    def train_discriminator(self, real_sim: torch.Tensor, real_real: torch.Tensor,
                          fake_real: torch.Tensor, fake_sim: torch.Tensor) -> Dict[str, float]:
        """
        训练判别器
        Args:
            real_sim: 真实仿真点云
            real_real: 真实真实点云
            fake_real: 生成的真实风格点云
            fake_sim: 生成的仿真风格点云
        Returns:
            判别器损失字典
        """
        losses = {}
        
        # 训练Real域判别器
        self.optimizer_D_real.zero_grad()
        
        # 真实数据
        real_output, real_features = self.discriminator_real(real_real)
        
        # 生成数据
        fake_output, fake_features = self.discriminator_real(fake_real.detach())
        
        # 计算损失
        d_real_loss, d_real_losses = self.criterion.discriminator_loss(real_output, fake_output)
        
        # 梯度惩罚
        gp_real = self.gradient_penalty(self.discriminator_real, real_real, fake_real)
        d_real_total = d_real_loss + gp_real
        
        d_real_total.backward()
        self.optimizer_D_real.step()
        
        losses.update({f'D_real_{k}': v.item() for k, v in d_real_losses.items()})
        losses['D_real_gp'] = gp_real.item()
        
        # 训练Sim域判别器
        self.optimizer_D_sim.zero_grad()
        
        # 真实数据
        real_output, real_features = self.discriminator_sim(real_sim)
        
        # 生成数据
        fake_output, fake_features = self.discriminator_sim(fake_sim.detach())
        
        # 计算损失
        d_sim_loss, d_sim_losses = self.criterion.discriminator_loss(real_output, fake_output)
        
        # 梯度惩罚
        gp_sim = self.gradient_penalty(self.discriminator_sim, real_sim, fake_sim)
        d_sim_total = d_sim_loss + gp_sim
        
        d_sim_total.backward()
        self.optimizer_D_sim.step()
        
        losses.update({f'D_sim_{k}': v.item() for k, v in d_sim_losses.items()})
        losses['D_sim_gp'] = gp_sim.item()
        
        return losses
    
    def train_generator(self, real_sim: torch.Tensor, real_real: torch.Tensor) -> Dict[str, float]:
        """
        训练生成器
        Args:
            real_sim: 真实仿真点云
            real_real: 真实真实点云
        Returns:
            生成器损失字典
        """
        self.optimizer_G.zero_grad()
        
        # 使用梯度累积来减少内存使用
        accumulation_steps = 2
        losses = {}
        total_loss = 0
        
        # 分步计算，避免同时存储所有中间结果
        for step in range(accumulation_steps):
            # 清理之前的缓存
            if step > 0:
                self._cleanup_memory()
            
            # 生成假数据
            fake_real, fake_sim = self.generator(real_sim, real_real)
            
            # 判别器预测（detach判别器以节省内存）
            with torch.no_grad():
                self.discriminator_real.eval()
                self.discriminator_sim.eval()
            
            fake_real_output, fake_real_features = self.discriminator_real(fake_real)
            fake_sim_output, fake_sim_features = self.discriminator_sim(fake_sim)
            
            # 对抗损失
            adv_loss_real = self.criterion.adversarial_loss(fake_real_output, True)
            adv_loss_sim = self.criterion.adversarial_loss(fake_sim_output, True)
            
            # 循环一致性（只在第一步计算）
            if step == 0:
                with torch.cuda.amp.autocast(enabled=False):  # 禁用混合精度以节省内存
                    cycled_sim = self.generator.real2sim(fake_real.detach(), real_sim)
                    cycled_real = self.generator.sim2real(fake_sim.detach(), real_real)
                
                cycle_loss_sim = self.criterion.chamfer_loss(cycled_sim, real_sim)
                cycle_loss_real = self.criterion.chamfer_loss(cycled_real, real_real)
                cycle_loss = (cycle_loss_sim + cycle_loss_real) * self.config.lambda_cycle
                
                # 立即删除不需要的变量
                del cycled_sim, cycled_real
            else:
                cycle_loss = 0
            
            # 身份损失（只在第二步计算）
            if step == 1:
                with torch.cuda.amp.autocast(enabled=False):
                    identity_real = self.generator.sim2real(real_real, real_real)
                    identity_sim = self.generator.real2sim(real_sim, real_sim)
                
                identity_loss_real = self.criterion.chamfer_loss(identity_real, real_real)
                identity_loss_sim = self.criterion.chamfer_loss(identity_sim, real_sim)
                identity_loss = (identity_loss_real + identity_loss_sim) * self.config.lambda_identity
                
                del identity_real, identity_sim
            else:
                identity_loss = 0
            
            # 计算当前步骤的损失
            step_loss = (adv_loss_real + adv_loss_sim) * self.config.lambda_adv
            if cycle_loss != 0:
                step_loss = step_loss + cycle_loss
            if identity_loss != 0:
                step_loss = step_loss + identity_loss
            
            # 梯度累积
            step_loss = step_loss / accumulation_steps
            step_loss.backward()
            
            total_loss += step_loss.item() * accumulation_steps
            
            # 记录损失
            if step == 0:
                losses['G_real_adv'] = adv_loss_real.item()
                losses['G_sim_adv'] = adv_loss_sim.item()
                if cycle_loss != 0:
                    losses['G_cycle'] = cycle_loss.item()
            if step == 1 and identity_loss != 0:
                losses['G_identity'] = identity_loss.item()
            
            # 删除中间变量
            del fake_real, fake_sim, fake_real_output, fake_sim_output
            del fake_real_features, fake_sim_features
            
        # 恢复判别器训练模式
        self.discriminator_real.train()
        self.discriminator_sim.train()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        
        self.optimizer_G.step()
        
        # 最终清理
        self._cleanup_memory()
        
        losses['G_total'] = total_loss
        
        # 返回损失和None（因为fake_real和fake_sim已经被删除）
        return losses, None, None
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        训练一个epoch
        Args:
            train_loader: 训练数据加载器
        Returns:
            平均损失字典
        """
        self.generator.train()
        self.discriminator_real.train()
        self.discriminator_sim.train()
        
        epoch_losses = {}
        num_batches = len(train_loader)
        
        with tqdm(total=num_batches, desc=f'Epoch {self.current_epoch}') as pbar:
            for batch_idx, batch in enumerate(train_loader):
                # 检查批次开始时的内存
                self._check_memory_usage(f"batch_{batch_idx}_start")
                
                real_sim = batch['sim_points'].to(self.device)
                real_real = batch['real_points'].to(self.device)
                
                # 训练生成器
                g_losses, _, _ = self.train_generator(real_sim, real_real)
                
                # 训练判别器（每隔几步训练一次）
                d_losses = {}
                if (batch_idx + 1) % self.config.discriminator_steps == 0:
                    # 重新生成数据用于判别器训练（避免过时的生成结果）
                    with torch.no_grad():
                        fake_real = self.generator.sim2real(real_sim, real_real)
                        fake_sim = self.generator.real2sim(real_real, real_sim)
                    
                    d_losses = self.train_discriminator(real_sim, real_real, fake_real, fake_sim)
                    
                    # 清理生成的数据
                    del fake_real, fake_sim
                    
                    # 合并损失
                    for k, v in d_losses.items():
                        if k not in epoch_losses:
                            epoch_losses[k] = []
                        epoch_losses[k].append(v)
                
                # 记录生成器损失
                for k, v in g_losses.items():
                    if k not in epoch_losses:
                        epoch_losses[k] = []
                    epoch_losses[k].append(v)
                
                # 清理当前批次的变量
                del real_sim, real_real
                
                # 定期清理内存
                if batch_idx % 10 == 0:
                    self._cleanup_memory()
                
                # 更新进度条
                pbar.set_postfix({'G_loss': f"{g_losses['G_total']:.4f}"})
                pbar.update(1)
                
                # 记录到TensorBoard
                if batch_idx % self.config.log_interval == 0:
                    for k, v in g_losses.items():
                        self.writer.add_scalar(f'Train/{k}', v, self.global_step)
                    
                    if d_losses:
                        for k, v in d_losses.items():
                            self.writer.add_scalar(f'Train/{k}', v, self.global_step)
                    
                    # 记录内存使用
                    allocated, reserved, max_alloc = self._check_memory_usage("logging")
                    self.writer.add_scalar('Memory/allocated_GB', allocated, self.global_step)
                    self.writer.add_scalar('Memory/reserved_GB', reserved, self.global_step)
                
                self.global_step += 1
        
        # 计算平均损失
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        # 最终清理
        self._cleanup_memory()
        
        return avg_losses
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        验证模型
        Args:
            val_loader: 验证数据加载器
        Returns:
            验证指标字典
        """
        self.generator.eval()
        
        val_metrics = {}
        all_chamfer_distances = []
        all_emd_distances = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                real_sim = batch['sim_points'].to(self.device)
                real_real = batch['real_points'].to(self.device)
                
                # 生成数据
                fake_real, fake_sim = self.generator(real_sim, real_real)
                
                # 计算指标
                # Chamfer距离
                cd_real = self.metrics.chamfer_distance(fake_real, real_real)
                cd_sim = self.metrics.chamfer_distance(fake_sim, real_sim)
                all_chamfer_distances.extend([cd_real.mean().item(), cd_sim.mean().item()])
                
                # EMD距离
                emd_real = self.metrics.earth_mover_distance(fake_real, real_real)
                emd_sim = self.metrics.earth_mover_distance(fake_sim, real_sim)
                all_emd_distances.extend([emd_real.mean().item(), emd_sim.mean().item()])
        
        val_metrics['chamfer_distance'] = np.mean(all_chamfer_distances)
        val_metrics['emd_distance'] = np.mean(all_emd_distances)
        
        return val_metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """
        保存检查点
        Args:
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_real_state_dict': self.discriminator_real.state_dict(),
            'discriminator_sim_state_dict': self.discriminator_sim.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_real_state_dict': self.optimizer_D_real.state_dict(),
            'optimizer_D_sim_state_dict': self.optimizer_D_sim.state_dict(),
            'best_score': self.best_score,
            'config': self.config
        }
        
        # 保存最新检查点
        latest_path = os.path.join(self.config.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved at epoch {self.current_epoch}")
        
        # 定期保存
        if self.current_epoch % self.config.save_interval == 0:
            epoch_path = os.path.join(self.config.checkpoint_dir, f'epoch_{self.current_epoch}.pth')
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点
        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_score = checkpoint['best_score']
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator_real.load_state_dict(checkpoint['discriminator_real_state_dict'])
        self.discriminator_sim.load_state_dict(checkpoint['discriminator_sim_state_dict'])
        
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D_real.load_state_dict(checkpoint['optimizer_D_real_state_dict'])
        self.optimizer_D_sim.load_state_dict(checkpoint['optimizer_D_sim_state_dict'])
        
        self.logger.info(f"Checkpoint loaded from epoch {self.current_epoch}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        主训练循环
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
        """
        self.logger.info("Starting training...")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # 训练
            train_losses = self.train_epoch(train_loader)
            
            # 验证
            if epoch % self.config.eval_interval == 0:
                val_metrics = self.validate(val_loader)
                
                # 记录验证结果
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f'Val/{k}', v, epoch)
                
                # 检查是否为最佳模型
                current_score = val_metrics['chamfer_distance']
                is_best = current_score < self.best_score
                if is_best:
                    self.best_score = current_score
                
                self.logger.info(f"Epoch {epoch}: Val CD = {current_score:.6f}, Best = {self.best_score:.6f}")
            else:
                is_best = False
            
            # 记录训练损失
            for k, v in train_losses.items():
                self.writer.add_scalar(f'Train_Avg/{k}', v, epoch)
            
            # 更新学习率
            self.scheduler_G.step()
            self.scheduler_D_real.step()
            self.scheduler_D_sim.step()
            
            # 保存检查点
            self.save_checkpoint(is_best)
            
            # 记录epoch时间
            epoch_time = time.time() - start_time
            self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
            
            # 可视化结果
            if epoch % (self.config.eval_interval * 2) == 0:
                self.visualize_results(val_loader, epoch)
        
        self.logger.info("Training completed!")
        self.writer.close()
    
    def visualize_results(self, val_loader: DataLoader, epoch: int):
        """
        可视化训练结果
        Args:
            val_loader: 验证数据加载器
            epoch: 当前epoch
        """
        self.generator.eval()
        
        with torch.no_grad():
            # 获取一个batch的数据
            batch = next(iter(val_loader))
            real_sim = batch['sim_points'][:self.config.vis_samples].to(self.device)
            real_real = batch['real_points'][:self.config.vis_samples].to(self.device)
            
            # 生成数据
            fake_real, fake_sim = self.generator(real_sim, real_real)
            
            # 创建可视化目录
            vis_dir = os.path.join(self.config.result_dir, f'epoch_{epoch}')
            os.makedirs(vis_dir, exist_ok=True)
            
            # 为每个样本创建可视化
            for i in range(self.config.vis_samples):
                try:
                    # 原始仿真点云
                    self.visualizer.save_point_cloud(
                        real_sim[i].cpu().numpy(),
                        os.path.join(vis_dir, f'sample_{i}_sim_original.png'),
                        title=f'Original Sim - Sample {i+1}',
                        color='sim'
                    )
                    
                    # 生成的真实风格点云
                    self.visualizer.save_point_cloud(
                        fake_real[i].cpu().numpy(),
                        os.path.join(vis_dir, f'sample_{i}_sim2real.png'),
                        title=f'Sim2Real - Sample {i+1}',
                        color='generated'
                    )
                    
                    # 真实点云参考
                    self.visualizer.save_point_cloud(
                        real_real[i].cpu().numpy(),
                        os.path.join(vis_dir, f'sample_{i}_real_reference.png'),
                        title=f'Real Reference - Sample {i+1}',
                        color='real'
                    )
                    
                    # 生成的仿真风格点云（如果有）
                    if fake_sim is not None:
                        self.visualizer.save_point_cloud(
                            fake_sim[i].cpu().numpy(),
                            os.path.join(vis_dir, f'sample_{i}_real2sim.png'),
                            title=f'Real2Sim - Sample {i+1}',
                            color='generated'
                        )
                    
                    # 创建对比可视化
                    comparison_path = os.path.join(vis_dir, f'sample_{i}_comparison.png')
                    self.visualizer.plot_style_transfer_result(
                        real_sim[i].cpu().numpy(),
                        fake_real[i].cpu().numpy(),
                        real_real[i].cpu().numpy(),
                        title=f'Style Transfer Result - Epoch {epoch}, Sample {i+1}',
                        save_path=comparison_path
                    )
                    
                except Exception as e:
                    self.logger.warning(f"Failed to visualize sample {i}: {e}")
                    continue
            
            # 记录到TensorBoard（如果可用）
            try:
                # 创建图像网格用于TensorBoard
                from torchvision.utils import make_grid
                import torch.nn.functional as F
                
                # 将点云转换为图像表示（简单的投影）
                def point_cloud_to_image(points, size=64):
                    """将点云转换为简单的投影图像"""
                    # 简单的XY平面投影
                    x, y = points[:, 0], points[:, 1]
                    
                    # 标准化到[0, size-1]
                    x_norm = ((x - x.min()) / (x.max() - x.min() + 1e-8) * (size - 1)).long()
                    y_norm = ((y - y.min()) / (y.max() - y.min() + 1e-8) * (size - 1)).long()
                    
                    # 创建图像
                    img = torch.zeros(size, size)
                    valid_mask = (x_norm >= 0) & (x_norm < size) & (y_norm >= 0) & (y_norm < size)
                    img[y_norm[valid_mask], x_norm[valid_mask]] = 1.0
                    
                    return img.unsqueeze(0)  # 添加通道维度
                
                # 转换样本为图像
                sim_images = torch.stack([point_cloud_to_image(real_sim[i].cpu()) for i in range(min(4, self.config.vis_samples))])
                fake_images = torch.stack([point_cloud_to_image(fake_real[i].cpu()) for i in range(min(4, self.config.vis_samples))])
                real_images = torch.stack([point_cloud_to_image(real_real[i].cpu()) for i in range(min(4, self.config.vis_samples))])
                
                # 创建网格并记录到TensorBoard
                sim_grid = make_grid(sim_images, nrow=2, normalize=True)
                fake_grid = make_grid(fake_images, nrow=2, normalize=True)
                real_grid = make_grid(real_images, nrow=2, normalize=True)
                
                self.writer.add_image('Samples/Original_Sim', sim_grid, epoch)
                self.writer.add_image('Samples/Generated_Real', fake_grid, epoch)
                self.writer.add_image('Samples/Reference_Real', real_grid, epoch)
                
            except Exception as e:
                self.logger.warning(f"Failed to log images to TensorBoard: {e}")
            
            self.logger.info(f"Visualization results saved to: {vis_dir}")
        
        # 恢复训练模式
        self.generator.train()
        
        return vis_dir


def create_trainer(config, resume_from=None):
    """
    创建训练器的工厂函数
    Args:
        config: 配置对象
        resume_from: 恢复训练的检查点路径
    Returns:
        训练器实例
    """
    trainer = PointCloudStyleTransferTrainer(config)
    
    if resume_from and os.path.exists(resume_from):
        trainer.load_checkpoint(resume_from)
        trainer.logger.info(f"Resumed training from {resume_from}")
    
    return trainer


def get_device_info():
    """获取设备信息"""
    if torch.cuda.is_available():
        device_info = {
            'device': 'cuda',
            'device_name': torch.cuda.get_device_name(),
            'device_count': torch.cuda.device_count(),
            'memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**3,
            'cuda_version': torch.version.cuda
        }
    else:
        device_info = {
            'device': 'cpu',
            'device_name': 'CPU',
            'device_count': 1,
            'memory_total': 0,
            'cuda_version': None
        }
    
    return device_info


def validate_config(config):
    """
    验证配置参数的合理性
    Args:
        config: 配置对象
    Returns:
        是否验证通过
    """
    errors = []
    warnings = []
    
    # 检查必要的参数
    if config.chunk_size <= 0:
        errors.append("chunk_size must be positive")
    
    if config.batch_size <= 0:
        errors.append("batch_size must be positive")
    
    if config.num_epochs <= 0:
        errors.append("num_epochs must be positive")
    
    if config.learning_rate_g <= 0 or config.learning_rate_g >= 1:
        errors.append("learning_rate_g must be in (0, 1)")
    
    if config.learning_rate_d <= 0 or config.learning_rate_d >= 1:
        errors.append("learning_rate_d must be in (0, 1)")
    
    # 检查损失权重
    if config.lambda_recon < 0:
        warnings.append("lambda_recon is negative")
    
    if config.lambda_adv < 0:
        warnings.append("lambda_adv is negative")
    
    if config.lambda_cycle < 0:
        warnings.append("lambda_cycle is negative")
    
    # 检查设备相关参数
    if config.device == 'cuda' and not torch.cuda.is_available():
        warnings.append("CUDA not available, will use CPU")
    
    # 检查内存相关参数
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        estimated_memory = config.batch_size * config.chunk_size * 4 * 3 / 1024**3  # 粗略估计
        
        if estimated_memory > gpu_memory * 0.8:
            warnings.append(f"High memory usage estimated: {estimated_memory:.1f}GB > {gpu_memory*0.8:.1f}GB")
    
    # 打印结果
    if errors:
        print("Configuration Errors:")
        for error in errors:
            print(f"  ❌ {error}")
        return False
    
    if warnings:
        print("Configuration Warnings:")
        for warning in warnings:
            print(f"  ⚠️ {warning}")
    
    return True


def setup_distributed_training(config):
    """
    设置分布式训练（预留接口）
    Args:
        config: 配置对象
    """
    # 这里可以添加分布式训练的设置
    # 例如：torch.distributed.init_process_group()
    pass


def cleanup_distributed_training():
    """清理分布式训练资源"""
    # 这里可以添加分布式训练的清理
    # 例如：torch.distributed.destroy_process_group()
    pass


if __name__ == "__main__":
    # 用于测试trainer模块
    import sys
    import os
    
    # 添加项目根目录到路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from config.config import Config
    from data.dataset import create_paired_data_loaders
    
    print("Testing PointCloudStyleTransferTrainer...")
    
    # 创建测试配置
    config = Config()
    config.batch_size = 2
    config.chunk_size = 1024
    config.num_epochs = 2
    config.log_interval = 1
    config.eval_interval = 1
    config.save_interval = 1
    
    # 验证配置
    if not validate_config(config):
        print("Configuration validation failed!")
        sys.exit(1)
    
    # 打印设备信息
    device_info = get_device_info()
    print("Device Info:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    
    # 创建模拟数据
    print("Creating mock data...")
    import torch
    import numpy as np
    
    # 创建简单的测试数据
    os.makedirs("test_data/sim", exist_ok=True)
    os.makedirs("test_data/real", exist_ok=True)
    
    for i in range(5):
        sim_points = np.random.randn(config.chunk_size, 3).astype(np.float32)
        real_points = np.random.randn(config.chunk_size, 3).astype(np.float32) * 0.5
        
        np.save(f"test_data/sim/sim_{i:03d}.npy", sim_points)
        np.save(f"test_data/real/real_{i:03d}.npy", real_points)
    
    try:
        # 测试数据加载器创建
        from data.preprocess import preprocess_dataset
        preprocess_dataset("test_data/sim", "test_data/real", "test_data/processed", 
                          config.chunk_size, "random")
        
        train_loader, val_loader, test_loader = create_paired_data_loaders(
            "test_data/processed",
            batch_size=config.batch_size,
            num_workers=0,  # 避免多进程问题
            augment_train=False
        )
        
        print(f"Data loaders created successfully!")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        # 创建训练器
        trainer = create_trainer(config)
        print("Trainer created successfully!")
        
        # 测试一个训练步骤
        print("Testing training step...")
        batch = next(iter(train_loader))
        
        # 模拟训练一个epoch（只处理一个batch）
        losses = trainer.train_epoch(train_loader)
        print(f"Training step completed. Losses: {losses}")
        
        # 测试验证
        print("Testing validation...")
        metrics = trainer.validate(val_loader)
        print(f"Validation completed. Metrics: {metrics}")
        
        # 测试可视化
        print("Testing visualization...")
        vis_dir = trainer.visualize_results(val_loader, 0)
        print(f"Visualization completed. Saved to: {vis_dir}")
        
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理测试数据
        import shutil
        if os.path.exists("test_data"):
            shutil.rmtree("test_data")
        print("Test data cleaned up.")