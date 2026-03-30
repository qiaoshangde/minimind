# 导入操作系统相关功能模块
import os
# 导入Python运行时环境相关功能
import sys

# 设置当前包的名称，用于相对导入
__package__ = "trainer"
# 将当前文件所在目录的父目录添加到系统路径中，以便能够导入其他模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入命令行参数解析模块
import argparse
# 导入时间处理模块
import time
# 导入警告处理模块，用于忽略警告信息
import warnings
# 导入PyTorch深度学习框架
import torch
# 导入PyTorch分布式通信模块
import torch.distributed as dist
# 导入上下文管理器，用于创建空上下文
from contextlib import nullcontext
# 从torch中导入优化器和神经网络模块
from torch import optim, nn
# 导入PyTorch分布式数据并行模块
from torch.nn.parallel import DistributedDataParallel
# 从torch.utils.data导入数据加载器和分布式采样器
from torch.utils.data import DataLoader, DistributedSampler
# 导入MiniMind模型配置类
from model.model_minimind import MiniMindConfig
# 导入监督微调数据集类
from dataset.lm_dataset import SFTDataset
# 导入LoRA相关的保存和应用函数
from model.model_lora import save_lora, apply_lora
# 导入训练工具函数：学习率调整、日志记录、主进程判断、检查点保存、分布式初始化、随机种子设置、模型初始化等
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

# 忽略所有警告信息，保持输出整洁
warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None):
    """
    训练一个epoch的函数
    
    参数:
        epoch: 当前epoch索引
        loader: 数据加载器，提供批次数据
        iters: 当前epoch的总迭代步数
        lora_params: LoRA参数列表，用于梯度裁剪
        start_step: 起始步数，用于断点续训
        wandb: wandb日志记录对象，可选
    """
    # 记录epoch开始时间，用于计算训练耗时
    start_time = time.time()
    # 记录最后一步的索引，用于后续处理未完成的梯度累积
    last_step = start_step
    # 遍历数据加载器，step从start_step+1开始计数
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 将输入token ids移动到指定设备（GPU/CPU）
        input_ids = input_ids.to(args.device)
        # 将标签token ids移动到指定设备
        labels = labels.to(args.device)
        # 更新最后一步索引
        last_step = step
        # 根据当前进度计算学习率（使用余弦退火或其他策略）
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 更新优化器中所有参数组的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 在混合精度上下文中进行前向传播
        with autocast_ctx:
            # 模型前向传播，输入input_ids和labels
            res = model(input_ids, labels=labels)
            # 计算总损失 = 主损失 + 辅助损失（如MoE的负载均衡损失）
            loss = res.loss + res.aux_loss
            # 除以梯度累积步数，实现小batch模拟大batch的效果
            loss = loss / args.accumulation_steps

        # 反向传播，使用梯度缩放器处理混合精度
        scaler.scale(loss).backward()

        # 每accumulation_steps步或最后一步时更新参数
        if step % args.accumulation_steps == 0:
            # 对优化器进行梯度反缩放，用于梯度裁剪
            scaler.unscale_(optimizer)
            # 对LoRA参数进行梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            # 执行优化器步骤（更新参数）
            scaler.step(optimizer)
            # 更新梯度缩放器
            scaler.update()
            # 清空梯度，set_to_none=True更高效
            optimizer.zero_grad(set_to_none=True)

        # 达到日志打印间隔或最后一个step时打印训练信息
        if step % args.log_interval == 0 or step == iters:
            # 计算已用时间
            spend_time = time.time() - start_time
            # 获取当前损失（还原梯度累积前的值）
            current_loss = loss.item() * args.accumulation_steps
            # 获取辅助损失值，如果不存在则为0
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            # 计算主损失（总损失 - 辅助损失）
            current_logits_loss = current_loss - current_aux_loss
            # 获取当前学习率
            current_lr = optimizer.param_groups[-1]['lr']
            # 估算剩余时间（分钟）
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            # 打印训练日志
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            # 如果启用了wandb，记录训练指标
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # 达到保存间隔或最后一个step时，且为主进程时保存模型
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            # 切换到评估模式，确保保存时不影响模型状态
            model.eval()
            # 构建LoRA权重保存路径
            lora_save_path = f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}.pth'
            # 只保存LoRA权重（不保存完整模型）
            save_lora(model, lora_save_path)
            # 保存完整检查点（包括模型、优化器、缩放器等状态）
            lm_checkpoint(lm_config, weight=args.lora_name, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            # 切换回训练模式
            model.train()

        # 释放不再需要的张量，节省显存
        del input_ids, labels, res, loss

    # 处理epoch结束时可能未完成的梯度累积步骤
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        # 反缩放梯度
        scaler.unscale_(optimizer)
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
        # 更新参数
        scaler.step(optimizer)
        # 更新缩放器
        scaler.update()
        # 清空梯度
        optimizer.zero_grad(set_to_none=True)

# 主程序入口
if __name__ == "__main__":
    # 创建命令行参数解析器，描述为"MiniMind LoRA Fine-tuning"
    parser = argparse.ArgumentParser(description="MiniMind LoRA Fine-tuning")
    # 添加各种命令行参数，每个参数都有默认值和帮助说明
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--lora_name", type=str, default="lora_medical", help="LoRA权重名称(如lora_identity/lora_medical等)")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/lora_medical.jsonl", help="LoRA训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练，默认full_sft")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-LoRA", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    # 解析命令行参数
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 初始化分布式训练环境，返回本地进程rank
    local_rank = init_distributed_mode()
    # 如果分布式已初始化，设置设备为当前进程的GPU
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 设置随机种子，确保结果可复现，不同进程使用不同种子避免数据重复
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    # 创建保存目录（如果不存在）
    os.makedirs(args.save_dir, exist_ok=True)
    # 创建MiniMind模型配置对象
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 如果需要续训，加载已有的检查点数据
    ckp_data = lm_checkpoint(lm_config, weight=args.lora_name, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    # 获取设备类型（cuda或cpu）
    device_type = "cuda" if "cuda" in args.device else "cpu"
    # 根据参数选择混合精度类型
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # 如果是CPU则使用空上下文，否则使用cuda的自动混合精度上下文
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    # 初始化wandb日志记录对象
    wandb = None
    # 如果启用了wandb且当前是主进程，则初始化wandb
    if args.use_wandb and is_main_process():
        import swanlab as wandb  # 注意：这里导入的是swanlab，可能是个笔误，实际应该是wandb
        # 获取续训时的wandb运行ID
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        # 如果有ID则设置resume模式
        resume = 'must' if wandb_id else None
        # 构建运行名称
        wandb_run_name = f"MiniMind-LoRA-{args.lora_name}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        # 初始化wandb运行
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、应用LoRA、冻结非LoRA参数 ==========
    # 初始化模型和分词器
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # 在模型上应用LoRA适配器
    apply_lora(model)
    
    # 统计总参数量
    total_params = sum(p.numel() for p in model.parameters())
    # 统计LoRA参数量（参数名中包含'lora'）
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    # 打印统计信息
    Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M")
    Logger(f"LoRA 参数量: {lora_params_count / 1e6:.3f} M")
    Logger(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")
    
    # 冻结非LoRA参数，收集需要训练的LoRA参数
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            # LoRA参数需要梯度
            param.requires_grad = True
            # 添加到LoRA参数列表
            lora_params.append(param)
        else:
            # 非LoRA参数冻结，不更新
            param.requires_grad = False
    
    # ========== 6. 定义数据和优化器 ==========
    # 创建SFT数据集实例
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 如果分布式训练，使用分布式采样器
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # 创建梯度缩放器，用于float16混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # 创建AdamW优化器，只优化LoRA参数
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
    
    # ========== 7. 从ckp恢复状态 ==========
    # 初始化起始epoch和step
    start_epoch, start_step = 0, 0
    # 如果有检查点数据，恢复训练状态
    if ckp_data:
        # 加载模型权重（strict=False允许部分加载）
        model.load_state_dict(ckp_data['model'], strict=False)
        # 加载优化器状态
        optimizer.load_state_dict(ckp_data['optimizer'])
        # 加载梯度缩放器状态
        scaler.load_state_dict(ckp_data['scaler'])
        # 恢复epoch和step
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 8. 编译和分布式包装 ==========
    # 如果启用torch.compile，编译模型以加速训练
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    # 如果分布式训练，包装模型为DistributedDataParallel
    if dist.is_initialized():
        # 指定不需要同步的缓冲区（位置编码相关）
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        # 创建分布式数据并行模型
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 9. 开始训练 ==========
    # 循环训练每个epoch
    for epoch in range(start_epoch, args.epochs):
        # 设置分布式采样器的epoch，确保数据shuffle正确
        train_sampler and train_sampler.set_epoch(epoch)
        # 设置随机种子并创建数据索引的随机排列
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 如果是续训且是起始epoch，需要跳过的步数
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # 创建跳过指定步数的批采样器
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # 创建数据加载器
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        # 如果有跳过的步数，打印提示信息
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            # 训练epoch，传入起始步数
            train_epoch(epoch, loader, len(loader) + skip, lora_params, start_step, wandb)
        else:
            # 正常训练epoch
            train_epoch(epoch, loader, len(loader), lora_params, 0, wandb)
    
    # ========== 10. 清理分布进程 ==========
    # 如果分布式初始化了，销毁进程组
    if dist.is_initialized(): dist.destroy_process_group()