"""
训练器模块
支持Teacher Forcing和Free Running策略
支持混合精度训练（AMP）和分布式训练（DDP）
支持学习率调度和损失可视化
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
try:
    from torch.amp import autocast, GradScaler  # PyTorch 2.0+
except ImportError:
    from torch.cuda.amp import autocast, GradScaler  # 旧版本兼容
from typing import Dict, Optional
import time
import os
import json
import hashlib
from tqdm import tqdm
from src.utils.metrics import compute_bleu
from src.decoding.decoder_strategy import GreedyDecoder

# 全局变量：matplotlib是否可用，是否使用交互式后端
HAS_MATPLOTLIB = False
MATPLOTLIB_INTERACTIVE = False

try:
    import matplotlib
    # 尝试使用交互式后端（优先顺序：TkAgg > Qt5Agg > QtAgg > GTK3Agg）
    # 如果都不可用，回退到Agg（非交互式）
    import os
    interactive_backends = ['TkAgg', 'Qt5Agg', 'QtAgg', 'GTK3Agg']
    backend_set = False
    selected_backend = None
    for backend in interactive_backends:
        try:
            matplotlib.use(backend)
            backend_set = True
            selected_backend = backend
            break
        except Exception as e:
            continue
    if not backend_set:
        matplotlib.use('Agg')  # 非交互式后端（服务器环境）
        selected_backend = 'Agg'
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
    # 设置交互模式（显示窗口）- 仅当使用交互式后端时
    if selected_backend != 'Agg':
        plt.ion()  # 开启交互模式
        MATPLOTLIB_INTERACTIVE = True
        # 只在主进程中打印提示
        if os.environ.get('LOCAL_RANK', '0') == '0':
            print(f"[INFO] 使用交互式后端（{selected_backend}），图表将显示在窗口中")
    else:
        MATPLOTLIB_INTERACTIVE = False
        # 只在主进程中打印提示
        if os.environ.get('LOCAL_RANK', '0') == '0':
            print("[INFO] 使用非交互式后端（Agg），图表将保存到文件，不会显示窗口")
            print("[INFO] 如需显示窗口，请安装GUI后端：sudo apt-get install python3-tk 或 pip install PyQt5")
except (ImportError, ModuleNotFoundError):
    HAS_MATPLOTLIB = False
    MATPLOTLIB_INTERACTIVE = False
    # 只在主进程中打印警告，避免分布式训练时重复输出
    import os
    if os.environ.get('LOCAL_RANK', '0') == '0':
        print("[WARNING] matplotlib未安装，将跳过损失可视化")


class Trainer:
    """训练器类"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 valid_loader: DataLoader,
                 optimizer: optim.Optimizer,
                 criterion: nn.Module,
                 device: torch.device,
                 config: Dict,
                 checkpoint_dir: str = 'checkpoints'):
        """
        初始化训练器
        
        Args:
            model: Seq2Seq模型
            train_loader: 训练数据加载器
            valid_loader: 验证数据加载器
            optimizer: 优化器
            criterion: 损失函数
            device: 设备
            config: 配置字典
            checkpoint_dir: 检查点保存目录
        """
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        
        # 检测模型类型
        model_type = config.get('model', {}).get('type', 'rnn').lower()
        self.is_transformer = (model_type == 'transformer')
        
        # 创建检查点目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 训练状态
        self.current_epoch = 0
        self.best_valid_loss = float('inf')
        self.patience_counter = 0
        
        # 梯度裁剪
        self.grad_clip = config['training'].get('grad_clip', 5.0)
        
        # 日志间隔
        self.log_interval = config['training'].get('log_interval', 100)
        
        # 评估间隔（每多少个batch评估一次）
        self.eval_interval = config['training'].get('eval_interval', 200)
        
        # 画图更新间隔（每多少个batch更新一次图表）
        self.plot_interval = config['training'].get('plot_interval', 200)
        
        # 全局batch计数器（用于跨epoch的batch编号）
        self.global_batch_idx = 0
        
        # Early Stopping
        self.early_stopping_patience = config['training'].get('early_stopping_patience', 5)
        
        # 混合精度训练
        self.use_amp = config['training'].get('use_amp', False)
        if self.use_amp:
            try:
                self.scaler = GradScaler('cuda')  # PyTorch 2.0+
            except TypeError:
                self.scaler = GradScaler()  # 旧版本兼容
        else:
            self.scaler = None
        
        # 分布式训练
        self.is_distributed = hasattr(model, 'module')  # 检查是否使用DDP
        self.is_main_process = True  # 是否为主进程（用于保存模型）
        
        # 如果是分布式训练，检查是否为主进程
        if self.is_distributed:
            import torch.distributed as dist
            self.is_main_process = (dist.get_rank() == 0)
        
        # 学习率调度器
        self.lr_scheduler_type = config['training'].get('lr_scheduler', 'none')
        self.lr_scheduler_obj = None
        if self.lr_scheduler_type != 'none':
            self._setup_lr_scheduler(config)
        
        # 损失历史记录（用于可视化）
        # 格式：[(batch_idx, epoch, value), ...]
        self.train_losses = []
        self.valid_losses = []
        self.train_ppls = []
        self.valid_ppls = []
        self.valid_bleus = []  # BLEU分数历史
        
        # 可视化文件路径
        self.plot_path = os.path.join(checkpoint_dir, 'training_curve.png')
        self.plot_dir = os.path.join(checkpoint_dir, 'plots')
        os.makedirs(self.plot_dir, exist_ok=True)  # 创建图像序列目录
        
        # 初始化matplotlib figure（复用同一个窗口）
        self.fig = None
        self.axes = None
        if HAS_MATPLOTLIB and self.is_main_process:
            try:
                self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
                self.fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
                if MATPLOTLIB_INTERACTIVE:
                    plt.ion()
                    plt.show(block=False)
            except Exception as e:
                print(f"[WARNING] 初始化绘图窗口失败: {e}")
                self.fig = None
                self.axes = None
        
        # 解码器（用于计算BLEU）
        self.decoder = None
        
        # 配置哈希（用于检查checkpoint兼容性）
        self.config_hash = self._compute_config_hash(config)
    
    def set_vocabs(self, src_vocab, tgt_vocab):
        """设置词汇表（用于BLEU计算）"""
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
    
    def _setup_lr_scheduler(self, config):
        """设置学习率调度器"""
        scheduler_type = config['training'].get('lr_scheduler', 'none')
        step_size = config['training'].get('lr_step_size', 3)
        gamma = config['training'].get('lr_gamma', 0.5)
        
        if scheduler_type == 'step':
            self.lr_scheduler_obj = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_type == 'cosine':
            self.lr_scheduler_obj = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config['training'].get('num_epochs', 10)
            )
        elif scheduler_type == 'exponential':
            self.lr_scheduler_obj = optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=gamma
            )
        elif scheduler_type == 'cosine_warm_restarts':
            self.lr_scheduler_obj = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=config['training'].get('T_0', 20), T_mult=config['training'].get('T_mult', 1)
            )
        else:
            raise ValueError(f"不支持的学习率调度器类型: {scheduler_type}")
    
    def _compute_config_hash(self, config):
        """计算配置哈希，用于检查checkpoint兼容性"""
        # 提取关键配置（影响模型结构的配置）
        key_config = {
            'model': config.get('model', {}),
            'data': {
                'max_vocab_size': config.get('data', {}).get('max_vocab_size'),
                'source_lang': config.get('data', {}).get('source_lang'),
                'target_lang': config.get('data', {}).get('target_lang'),
            }
        }
        config_str = json.dumps(key_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _check_checkpoint_compatibility(self, checkpoint_path):
        """检查checkpoint是否与当前配置兼容"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            checkpoint_config = checkpoint.get('config', {})
            checkpoint_hash = self._compute_config_hash(checkpoint_config)
            return checkpoint_hash == self.config_hash, checkpoint
        except Exception as e:
            return False, None
    
    def _plot_training_curve(self):
        """绘制训练曲线（横坐标为batch）"""
        if not HAS_MATPLOTLIB or not self.is_main_process:
            return
        
        try:
            if not self.train_losses and not self.valid_losses:
                return
            
            # 如果没有figure，创建新的
            if self.fig is None or self.axes is None:
                self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
                self.fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
                if MATPLOTLIB_INTERACTIVE:
                    plt.ion()
                    plt.show(block=False)
            
            # 提取batch编号和loss/ppl值
            def extract_data(data_list):
                if not data_list:
                    return [], []
                # 检查格式
                if isinstance(data_list[0], (list, tuple)) and len(data_list[0]) >= 2:
                    # 新格式：(batch_idx, epoch, value)
                    batches = [item[0] for item in data_list]
                    values = [item[2] if len(item) > 2 else item[1] for item in data_list]
                else:
                    # 旧格式：value（兼容旧checkpoint）
                    batches = list(range(1, len(data_list) + 1))
                    values = data_list
                return batches, values
            
            train_batches, train_losses = extract_data(self.train_losses)
            valid_batches, valid_losses = extract_data(self.valid_losses)
            train_ppl_batches, train_ppls = extract_data(self.train_ppls)
            valid_ppl_batches, valid_ppls = extract_data(self.valid_ppls)
            valid_bleu_batches, valid_bleus = extract_data(self.valid_bleus)
            
            # Debug: Print data counts
            if self.global_batch_idx <= 30 or (self.global_batch_idx % 50 == 0):
                print(f"[DEBUG] Plot data: Train Losses={len(train_batches)}, Valid Losses={len(valid_batches)}, Valid BLEUs={len(valid_bleu_batches)}")
            
            # 清空所有子图
            for ax in self.axes.flat:
                ax.clear()
            
            # 1. 损失图（左上）
            ax = self.axes[0, 0]
            if train_batches and train_losses:
                ax.plot(train_batches, train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.7)
            if valid_batches and valid_losses:
                ax.plot(valid_batches, valid_losses, 'r-', label='Valid Loss', linewidth=2, alpha=0.7)
            ax.set_xlabel('Batch')
            ax.set_ylabel('Loss')
            ax.set_title('Training and Validation Loss')
            if train_batches or valid_batches:
                ax.legend()
            ax.grid(True, alpha=0.3)
            # 自动调整x轴和y轴范围
            if train_batches or valid_batches:
                all_batches = (train_batches or []) + (valid_batches or [])
                all_losses = (train_losses or []) + (valid_losses or [])
                if all_batches:
                    ax.set_xlim([0, max(all_batches) * 1.05])
                if all_losses:
                    ax.set_ylim([0, max(all_losses) * 1.1])
            
            # 2. 困惑度图（右上）
            ax = self.axes[0, 1]
            if train_ppl_batches and train_ppls:
                ax.plot(train_ppl_batches, train_ppls, 'b-', label='Train PPL', linewidth=2, alpha=0.7)
            if valid_ppl_batches and valid_ppls:
                ax.plot(valid_ppl_batches, valid_ppls, 'r-', label='Valid PPL', linewidth=2, alpha=0.7)
            ax.set_xlabel('Batch')
            ax.set_ylabel('Perplexity')
            ax.set_title('Training and Validation Perplexity')
            if train_ppl_batches or valid_ppl_batches:
                ax.legend()
            ax.grid(True, alpha=0.3)
            # 自动调整x轴和y轴范围
            if train_ppl_batches or valid_ppl_batches:
                all_batches = (train_ppl_batches or []) + (valid_ppl_batches or [])
                all_ppls = (train_ppls or []) + (valid_ppls or [])
                if all_batches:
                    ax.set_xlim([0, max(all_batches) * 1.05])
                if all_ppls:
                    ax.set_ylim([0, max(all_ppls) * 1.1])
            # 自动调整x轴范围
            if train_ppl_batches or valid_ppl_batches:
                all_batches = (train_ppl_batches or []) + (valid_ppl_batches or [])
                if all_batches:
                    ax.set_xlim([0, max(all_batches) * 1.05])
            
            # 3. BLEU Score Plot (bottom left)
            ax = self.axes[1, 0]
            if valid_bleu_batches and valid_bleus:
                # Filter non-zero values (if all are 0, show message)
                non_zero_bleus = [b for b in valid_bleus if b > 0]
                if non_zero_bleus:
                    # Plot only non-zero BLEU values
                    non_zero_batches = [valid_bleu_batches[i] for i, b in enumerate(valid_bleus) if b > 0]
                    ax.plot(non_zero_batches, non_zero_bleus, 'g-', label='Valid BLEU', linewidth=2, alpha=0.7, marker='o', markersize=3)
                elif len(valid_bleus) > 0:
                    # If all are 0, show message
                    ax.text(0.5, 0.5, f'BLEU Score: 0\n(Evaluated {len(valid_bleus)} times)\nModel still training...', 
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=11, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
                # Plot all BLEU values (including 0) as reference line
                ax.plot(valid_bleu_batches, valid_bleus, 'g--', alpha=0.3, linewidth=1, label='BLEU (all)')
            else:
                # If no BLEU data yet, show message
                ax.text(0.5, 0.5, f'BLEU Score\nWaiting for evaluation...\n(Eval every {self.eval_interval} batches)', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=11, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            ax.set_xlabel('Batch')
            ax.set_ylabel('BLEU Score')
            ax.set_title('Validation BLEU Score')
            if valid_bleu_batches:
                ax.legend()
            ax.grid(True, alpha=0.3)
            if valid_bleu_batches and valid_bleus:
                non_zero_bleus = [b for b in valid_bleus if b > 0]
                if non_zero_bleus:
                    ax.set_ylim([0, max(100, max(non_zero_bleus) * 1.1)])
                else:
                    ax.set_ylim([0, 100])  # 默认范围
            else:
                ax.set_ylim([0, 100])  # 默认范围
            
            # 4. Summary Metrics (bottom right) - Show latest metrics
            ax = self.axes[1, 1]
            ax.axis('off')  # Turn off axes
            info_text = "Latest Metrics:\n\n"
            if train_losses:
                info_text += f"Train Loss: {train_losses[-1]:.4f}\n"
            else:
                info_text += f"Train Loss: N/A\n"
            if valid_losses:
                info_text += f"Valid Loss: {valid_losses[-1]:.4f}\n"
            else:
                info_text += f"Valid Loss: N/A (waiting...)\n"
            if train_ppls:
                info_text += f"Train PPL: {train_ppls[-1]:.2f}\n"
            else:
                info_text += f"Train PPL: N/A\n"
            if valid_ppls:
                info_text += f"Valid PPL: {valid_ppls[-1]:.2f}\n"
            else:
                info_text += f"Valid PPL: N/A (waiting...)\n"
            if valid_bleus:
                bleu_val = valid_bleus[-1]
                if bleu_val > 0:
                    info_text += f"Valid BLEU: {bleu_val:.2f}\n"
                else:
                    info_text += f"Valid BLEU: 0.00 (training...)\n"
            else:
                info_text += f"Valid BLEU: N/A (waiting...)\n"
            info_text += f"\nBatch: {self.global_batch_idx}\n"
            info_text += f"Epoch: {self.current_epoch + 1}\n"
            info_text += f"\nEval Freq: every {self.eval_interval} batches\n"
            info_text += f"Plot Freq: every {self.plot_interval} batches"
            ax.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center', 
                   family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            self.fig.tight_layout()
            
            # 保存图片到主文件
            self.fig.savefig(self.plot_path, dpi=150, bbox_inches='tight')
            
            # 保存图像序列（带batch编号）
            try:
                # 确保目录存在
                os.makedirs(self.plot_dir, exist_ok=True)
                plot_seq_path = os.path.join(self.plot_dir, f'training_curve_batch_{self.global_batch_idx:06d}.png')
                self.fig.savefig(plot_seq_path, dpi=150, bbox_inches='tight')
            except Exception as e:
                print(f"[WARNING] 保存图像序列失败: {e}")
            
            # 显示窗口（交互式模式）- 如果使用交互式后端
            if MATPLOTLIB_INTERACTIVE:
                # 强制更新窗口 - 使用多种方法确保更新
                try:
                    # 方法1: 使用draw()强制重绘
                    self.fig.canvas.draw()
                    # 方法2: 刷新事件队列
                    self.fig.canvas.flush_events()
                    # 方法3: 确保窗口可见
                    if hasattr(self.fig.canvas, 'manager') and self.fig.canvas.manager:
                        try:
                            self.fig.canvas.manager.show()
                        except:
                            pass
                    # 方法4: 短暂暂停，给GUI时间更新
                    plt.pause(0.1)  # 增加暂停时间，确保窗口更新
                except Exception as e:
                    # 如果更新失败，尝试备用方法
                    try:
                        self.fig.canvas.draw_idle()
                        plt.pause(0.1)
                    except:
                        pass
        except Exception as e:
            print(f"[WARNING] 绘制训练曲线失败: {e}")
    
    def train_epoch(self, teacher_forcing_ratio: float = 0.5) -> float:
        """
        训练一个epoch
        
        Args:
            teacher_forcing_ratio: Teacher Forcing比例
            
        Returns:
            平均训练损失
        """
        self.model.train()
        epoch_loss = 0
        epoch_batch_count = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1} [Train]')
        
        for batch_idx, (src, src_lengths, tgt, tgt_lengths) in enumerate(pbar):
            # 移至设备
            src = src.to(self.device, non_blocking=True)
            src_lengths = src_lengths.to(self.device, non_blocking=True)
            tgt = tgt.to(self.device, non_blocking=True)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            # 混合精度训练
            if self.use_amp:
                try:
                    autocast_context = autocast('cuda')  # PyTorch 2.0+
                except TypeError:
                    autocast_context = autocast()  # 旧版本兼容
                with autocast_context:
                    # 根据模型类型调用不同的forward方法
                    if self.is_transformer:
                        # Transformer模型：需要创建掩码
                        src_mask = (src != 0).float()  # [batch_size, src_len]
                        tgt_mask = (tgt != 0).float()  # [batch_size, tgt_len]
                        # outputs: [batch_size, tgt_len-1, vocab_size]
                        outputs = self.model(src, tgt, src_mask, tgt_mask, teacher_forcing_ratio)
                    else:
                        # RNN模型
                        outputs, _ = self.model(src, src_lengths, tgt, teacher_forcing_ratio)
                    
                    # 计算损失
                    outputs = outputs.contiguous().view(-1, outputs.size(-1))
                    tgt = tgt[:, 1:].contiguous().view(-1)
                    
                    loss = self.criterion(outputs, tgt)
                
                # 反向传播（使用scaler）
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                # 更新参数
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 根据模型类型调用不同的forward方法
                if self.is_transformer:
                    # Transformer模型：需要创建掩码
                    src_mask = (src != 0).float()  # [batch_size, src_len]
                    tgt_mask = (tgt != 0).float()  # [batch_size, tgt_len]
                    # outputs: [batch_size, tgt_len-1, vocab_size]
                    outputs = self.model(src, tgt, src_mask, tgt_mask, teacher_forcing_ratio)
                else:
                    # RNN模型
                    outputs, _ = self.model(src, src_lengths, tgt, teacher_forcing_ratio)
                
                # 计算损失
                # tgt[:, 1:]: [batch_size, tgt_len-1] (去掉<sos>)
                # outputs: [batch_size, tgt_len-1, vocab_size]
                outputs = outputs.contiguous().view(-1, outputs.size(-1))
                tgt = tgt[:, 1:].contiguous().view(-1)
                
                loss = self.criterion(outputs, tgt)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                # 更新参数
                self.optimizer.step()

            # 更新学习率
            if self.lr_scheduler_obj:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.lr_scheduler_obj.step()
                new_lr = self.optimizer.param_groups[0]['lr']
                # if current_lr != new_lr:
                #     print(f'\t学习率更新: {current_lr:.6f} → {new_lr:.6f}')
            
            # 累计损失
            epoch_loss += loss.item()
            epoch_batch_count += 1
            
            # 更新进度条
            pbar.set_postfix({'loss': loss.item()})
            
            # 更新全局batch编号
            self.global_batch_idx += 1
            
            # 只在每plot_interval个batch记录一次训练损失（用于绘图）
            # 跳过前几个batch的异常大loss
            skip_initial_batches = 3  # 跳过前3个batch
            if self.global_batch_idx % self.plot_interval == 0 and self.global_batch_idx > skip_initial_batches:
                batch_loss = loss.item()
                batch_ppl = torch.exp(torch.tensor(batch_loss)).item()
                self.train_losses.append((self.global_batch_idx, self.current_epoch, batch_loss))
                self.train_ppls.append((self.global_batch_idx, self.current_epoch, batch_ppl))
            
            # 每隔eval_interval个batch进行一次评估
            if self.global_batch_idx % self.eval_interval == 0:
                # 快速评估
                self.model.eval()
                eval_loss = 0
                eval_batches = 0
                max_eval_batches = 10  # 只评估前10个batch以节省时间
                
                with torch.no_grad():
                    for eval_src, eval_src_lengths, eval_tgt, eval_tgt_lengths in self.valid_loader:
                        if eval_batches >= max_eval_batches:
                            break
                        eval_src = eval_src.to(self.device, non_blocking=True)
                        eval_src_lengths = eval_src_lengths.to(self.device, non_blocking=True)
                        eval_tgt = eval_tgt.to(self.device, non_blocking=True)
                        
                        if self.use_amp:
                            try:
                                autocast_context = autocast('cuda')
                            except TypeError:
                                autocast_context = autocast()
                            with autocast_context:
                                if self.is_transformer:
                                    eval_src_mask = (eval_src != 0).float()
                                    eval_tgt_mask = (eval_tgt != 0).float()
                                    eval_outputs = self.model(eval_src, eval_tgt, eval_src_mask, eval_tgt_mask, 0.0)
                                else:
                                    eval_outputs, _ = self.model(eval_src, eval_src_lengths, eval_tgt, 0.0)
                                eval_outputs = eval_outputs.contiguous().view(-1, eval_outputs.size(-1))
                                eval_tgt_flat = eval_tgt[:, 1:].contiguous().view(-1)
                                eval_batch_loss = self.criterion(eval_outputs, eval_tgt_flat)
                        else:
                            if self.is_transformer:
                                eval_src_mask = (eval_src != 0).float()
                                eval_tgt_mask = (eval_tgt != 0).float()
                                eval_outputs = self.model(eval_src, eval_tgt, eval_src_mask, eval_tgt_mask, 0.0)
                            else:
                                eval_outputs, _ = self.model(eval_src, eval_src_lengths, eval_tgt, 0.0)
                            eval_outputs = eval_outputs.contiguous().view(-1, eval_outputs.size(-1))
                            eval_tgt_flat = eval_tgt[:, 1:].contiguous().view(-1)
                            eval_batch_loss = self.criterion(eval_outputs, eval_tgt_flat)
                        
                        eval_loss += eval_batch_loss.item()
                        eval_batches += 1
                
                avg_eval_loss = eval_loss / eval_batches if eval_batches > 0 else 0
                eval_ppl = torch.exp(torch.tensor(avg_eval_loss)).item()
                
                # 计算BLEU分数（仅在eval_interval时计算，避免太频繁）
                bleu_score = 0.0
                if self.src_vocab is not None and self.tgt_vocab is not None:
                    try:
                        # 初始化解码器（如果还没有）
                        if self.decoder is None:
                            model_for_decode = self.model.module if self.is_distributed else self.model
                            self.decoder = GreedyDecoder(
                                model=model_for_decode,
                                sos_idx=self.tgt_vocab.SOS_IDX,
                                eos_idx=self.tgt_vocab.EOS_IDX,
                                max_length=self.config['decoding'].get('max_length', 100)
                            )
                        
                        # 计算BLEU（只评估前5个batch以节省时间）
                        predictions = []
                        references = []
                        bleu_batches = 0
                        for eval_src, eval_src_lengths, eval_tgt, eval_tgt_lengths in self.valid_loader:
                            if bleu_batches >= 5:  # 只评估前5个batch
                                break
                            
                            batch_size = eval_src.size(0)
                            eval_src = eval_src.to(self.device)
                            eval_src_lengths = eval_src_lengths.to(self.device)
                            
                            # 对batch中的每个样本分别解码
                            for i in range(batch_size):
                                try:
                                    # 提取单个样本的实际长度
                                    actual_src_len = eval_src_lengths[i].item()
                                    # 只提取实际长度的部分，避免padding导致的维度不匹配
                                    src_single = eval_src[i:i+1, :actual_src_len].to(self.device)  # [1, actual_src_len]
                                    src_lengths_single = torch.tensor([actual_src_len], device=self.device)  # [1]
                                    tgt_single = eval_tgt[i:i+1]  # [1, tgt_len]
                                    
                                    # 翻译（单个样本）
                                    # 现在src_single的长度维度与实际长度匹配，encoder输出也会匹配
                                    decoded_indices, _ = self.decoder.decode(src_single, src_lengths_single)
                                    
                                    # 解码为文本
                                    if isinstance(decoded_indices, list) and len(decoded_indices) > 0:
                                        decoded_tokens = self.tgt_vocab.decode(decoded_indices, skip_special=True)
                                        prediction = ' '.join(decoded_tokens)
                                    else:
                                        prediction = ""
                                    
                                    # 参考翻译
                                    ref_tokens = self.tgt_vocab.decode(tgt_single[0].tolist(), skip_special=True)
                                    reference = ' '.join(ref_tokens)
                                    
                                    predictions.append(prediction)
                                    references.append(reference)
                                except Exception as e:
                                    if self.is_main_process:
                                        print(f"[WARNING] BLEU decode failed for sample {i}: {e}")
                                        import traceback
                                        traceback.print_exc()
                                    continue
                            
                            bleu_batches += 1
                            if len(predictions) >= 20:  # At least 20 samples for BLEU calculation
                                break
                        
                        if predictions and references:
                            bleu_score = compute_bleu(predictions, references)
                            if self.is_main_process:
                                print(f"[DEBUG] BLEU calculation: {len(predictions)} samples, score={bleu_score:.2f}")
                        else:
                            if self.is_main_process:
                                print(f"[DEBUG] BLEU calculation: predictions or references are empty")
                    except Exception as e:
                        # BLEU计算失败不影响训练
                        if self.is_main_process:
                            print(f"[WARNING] BLEU计算失败: {e}")
                            import traceback
                            traceback.print_exc()
                        bleu_score = 0.0
                
                # 记录评估结果（格式：batch_idx, epoch, value）
                self.valid_losses.append((self.global_batch_idx, self.current_epoch, avg_eval_loss))
                self.valid_ppls.append((self.global_batch_idx, self.current_epoch, eval_ppl))
                # 记录BLEU分数（即使为0也记录，方便调试和绘图）
                self.valid_bleus.append((self.global_batch_idx, self.current_epoch, bleu_score))
                if self.is_main_process:
                    print(f"[DEBUG] Recorded eval results: Valid Loss={avg_eval_loss:.4f}, Valid PPL={eval_ppl:.2f}, BLEU={bleu_score:.2f} (total {len(self.valid_losses)} valid points)")
                
                # 打印评估结果
                if self.is_main_process:
                    bleu_str = f' | Valid BLEU: {bleu_score:.2f}' if bleu_score > 0 else ''
                    print(f'\n[Eval @ Batch {self.global_batch_idx} (Epoch {self.current_epoch + 1})] '
                          f'Valid Loss: {avg_eval_loss:.4f} | Valid PPL: {eval_ppl:.2f}{bleu_str}')
                
                # 恢复训练模式
                self.model.train()
            
            # 每隔plot_interval个batch更新一次图表
            if self.global_batch_idx % self.plot_interval == 0:
                if self.is_main_process:
                    # Add debug info
                    if self.global_batch_idx <= 50:  # Print debug info for first 50 batches
                        print(f"[DEBUG] Plot update @ Batch {self.global_batch_idx}, Train Losses: {len(self.train_losses)}, Valid Losses: {len(self.valid_losses)}, Valid BLEUs: {len(self.valid_bleus)}")
                    self._plot_training_curve()
        
        # Note: Training loss is now recorded only every plot_interval batches
        # Calculate epoch average for logging purposes
        avg_train_loss = epoch_loss / len(self.train_loader)
        
        return avg_train_loss
    
    def evaluate(self, teacher_forcing_ratio: float = 0.0) -> float:
        """
        评估模型
        
        Args:
            teacher_forcing_ratio: Teacher Forcing比例（评估时通常为0）
            
        Returns:
            平均验证损失
        """
        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(self.valid_loader, desc=f'Epoch {self.current_epoch + 1} [Valid]')
            
            for src, src_lengths, tgt, tgt_lengths in pbar:
                # 移至设备
                src = src.to(self.device, non_blocking=True)
                src_lengths = src_lengths.to(self.device, non_blocking=True)
                tgt = tgt.to(self.device, non_blocking=True)
                
                # 前向传播（评估时也使用AMP以保持一致性）
                if self.use_amp:
                    try:
                        autocast_context = autocast('cuda')  # PyTorch 2.0+
                    except TypeError:
                        autocast_context = autocast()  # 旧版本兼容
                    with autocast_context:
                        if self.is_transformer:
                            src_mask = (src != 0).float()
                            tgt_mask = (tgt != 0).float()
                            outputs = self.model(src, tgt, src_mask, tgt_mask, teacher_forcing_ratio)
                        else:
                            outputs, _ = self.model(src, src_lengths, tgt, teacher_forcing_ratio)
                        outputs = outputs.contiguous().view(-1, outputs.size(-1))
                        tgt = tgt[:, 1:].contiguous().view(-1)
                        loss = self.criterion(outputs, tgt)
                else:
                    if self.is_transformer:
                        src_mask = (src != 0).float()
                        tgt_mask = (tgt != 0).float()
                        outputs = self.model(src, tgt, src_mask, tgt_mask, teacher_forcing_ratio)
                    else:
                        outputs, _ = self.model(src, src_lengths, tgt, teacher_forcing_ratio)
                    outputs = outputs.contiguous().view(-1, outputs.size(-1))
                    tgt = tgt[:, 1:].contiguous().view(-1)
                    loss = self.criterion(outputs, tgt)
                
                epoch_loss += loss.item()
                
                # 更新进度条
                pbar.set_postfix({'loss': loss.item()})
        
        return epoch_loss / len(self.valid_loader)
    
    def train(self, num_epochs: int, teacher_forcing_ratio: float = 0.5, start_epoch: int = 0):
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数
            teacher_forcing_ratio: Teacher Forcing比例
            start_epoch: 起始epoch（用于从checkpoint恢复）
        """
        if start_epoch == 0:
            print(f"开始训练，共 {num_epochs} 个epoch")
        else:
            print(f"从epoch {start_epoch + 1}继续训练，共 {num_epochs} 个epoch")
        print(f"Teacher Forcing Ratio: {teacher_forcing_ratio}")
        if self.lr_scheduler_obj:
            print(f"学习率调度器: {self.lr_scheduler_type}")
        
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            
            # 分布式训练：设置sampler的epoch（确保每个epoch数据分布不同）
            if self.is_distributed and hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            if self.is_distributed and hasattr(self.valid_loader, 'sampler') and hasattr(self.valid_loader.sampler, 'set_epoch'):
                self.valid_loader.sampler.set_epoch(epoch)
            
            # 训练
            start_time = time.time()
            train_loss = self.train_epoch(teacher_forcing_ratio)
            
            # 验证（epoch结束时）
            valid_loss = self.evaluate()
            
            # 计算困惑度
            train_ppl = torch.exp(torch.tensor(train_loss)).item()
            valid_ppl = torch.exp(torch.tensor(valid_loss)).item()
            
            # 记录epoch结束时的验证结果（格式：batch_idx, epoch, value）
            # 注意：训练损失已经在train_epoch中记录
            self.valid_losses.append((self.global_batch_idx, self.current_epoch, valid_loss))
            self.valid_ppls.append((self.global_batch_idx, self.current_epoch, valid_ppl))
            
            end_time = time.time()
            epoch_time = end_time - start_time
            
            # 打印信息
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'\nEpoch: {epoch + 1:02}/{num_epochs} | Batch: {self.global_batch_idx} | Time: {epoch_time:.2f}s | LR: {current_lr:.6f}')
            print(f'\tTrain Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}')
            print(f'\tValid Loss: {valid_loss:.4f} | Valid PPL: {valid_ppl:.2f}')
            
            # Epoch结束时也更新图表
            if self.is_main_process:
                self._plot_training_curve()
            
            # 保存最佳模型（只在主进程保存）
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self.patience_counter = 0
                if self.is_main_process:
                    self.save_checkpoint('best_model.pt')
                    print(f'\t✓ 保存最佳模型 (Valid Loss: {valid_loss:.4f})')
            else:
                self.patience_counter += 1
            
            # 定期保存检查点（只在主进程保存）
            if (epoch + 1) % 2 == 0 and self.is_main_process:
                checkpoint_name = f'checkpoint_epoch_{epoch + 1}.pt'
                self.save_checkpoint(checkpoint_name)
                print(f'\t✓ 保存检查点: {checkpoint_name}')
            
            # Early Stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                break
        
        # 分布式训练：确保所有进程同步后再退出（避免NCCL超时）
        if self.is_distributed:
            import torch.distributed as dist
            dist.barrier()  # 等待所有进程完成
        
        print('\n训练完成！')
        if self.is_main_process and HAS_MATPLOTLIB:
            print(f'训练曲线已保存到: {self.plot_path}')
    
    def save_checkpoint(self, filename: str):
        """
        保存检查点
        
        Args:
            filename: 文件名
        """
        # 如果是DDP模型，保存model.module而不是model
        model_state_dict = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_valid_loss': self.best_valid_loss,
            'config': self.config,
            'config_hash': self.config_hash,  # 保存配置哈希
            'train_losses': self.train_losses,  # 保存损失历史
            'valid_losses': self.valid_losses,
            'train_ppls': self.train_ppls,
            'valid_ppls': self.valid_ppls,
            'global_batch_idx': self.global_batch_idx,  # 保存全局batch编号
        }
        
        # 保存学习率调度器状态
        if self.lr_scheduler_obj:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler_obj.state_dict()
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        try:
            torch.save(checkpoint, filepath)
            if self.is_main_process:
                file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                print(f'[DEBUG] Checkpoint saved: {filename} ({file_size:.1f} MB)')
        except Exception as e:
            if self.is_main_process:
                print(f'[ERROR] Failed to save checkpoint {filename}: {e}')
            raise
    
    def load_checkpoint(self, filename: str, load_loss_history: bool = True):
        """
        加载检查点
        
        Args:
            filename: 文件名
            load_loss_history: 是否加载损失历史
        """
        filepath = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 检查配置兼容性
        checkpoint_hash = checkpoint.get('config_hash', '')
        if checkpoint_hash and checkpoint_hash != self.config_hash:
            raise ValueError(
                f"Checkpoint配置不匹配！\n"
                f"当前配置哈希: {self.config_hash}\n"
                f"Checkpoint配置哈希: {checkpoint_hash}\n"
                f"请使用匹配的配置或从头开始训练"
            )
        
        # 加载模型状态（处理DDP包装的前缀问题）
        model_state_dict = checkpoint['model_state_dict']
        
        # 如果当前模型是DDP包装的，但checkpoint中没有"module."前缀，需要添加
        if self.is_distributed:
            # 检查checkpoint的key是否有"module."前缀
            has_module_prefix = any(k.startswith('module.') for k in model_state_dict.keys())
            # 检查当前模型是否需要"module."前缀
            current_model = self.model.module if hasattr(self.model, 'module') else self.model
            current_keys = set(current_model.state_dict().keys())
            
            if not has_module_prefix:
                # checkpoint没有"module."前缀，但当前模型是DDP，需要添加前缀
                model_state_dict = {f'module.{k}': v for k, v in model_state_dict.items()}
            elif has_module_prefix and not any(k.startswith('module.') for k in current_keys):
                # checkpoint有"module."前缀，但当前模型不是DDP，需要移除前缀
                model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items() if k.startswith('module.')}
        
        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_valid_loss = checkpoint['best_valid_loss']
        
        # 加载全局batch编号
        self.global_batch_idx = checkpoint.get('global_batch_idx', 0)
        
        # 加载损失历史
        if load_loss_history:
            self.train_losses = checkpoint.get('train_losses', [])
            self.valid_losses = checkpoint.get('valid_losses', [])
            self.train_ppls = checkpoint.get('train_ppls', [])
            self.valid_ppls = checkpoint.get('valid_ppls', [])
        
        # 加载学习率调度器状态
        if self.lr_scheduler_obj and 'lr_scheduler_state_dict' in checkpoint:
            self.lr_scheduler_obj.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        print(f'检查点已加载: {filepath}')
        print(f'Epoch: {self.current_epoch + 1}, Best Valid Loss: {self.best_valid_loss:.4f}')
        
        return checkpoint

