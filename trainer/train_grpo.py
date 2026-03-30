import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import math
import re
import gc
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel
from trainer.rollout_engine import create_rollout_engine, compute_per_token_logps

warnings.filterwarnings('ignore')

#检查文本中重复内容的函数，计算重复惩罚值。通过提取文本中的n-gram（默认n=3），计算总的n-gram数量和唯一n-gram数量之间的差异来衡量文本的重复程度。
def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0

# 输入数据流:
# ┌─────────────────┐
# │   prompts (B)   │ ──┐
# └─────────────────┘   │
#                        ├─→ 遍历每个prompt和它的num_generations个响应
# ┌─────────────────┐   │
# │ responses (B×G) │ ──┘
# └─────────────────┘

# 对每个响应:
# 1. 解析prompt → messages
#    "prompt" ──正则提取──→ [{"role": "user", "content": "..."}]

# 2. 响应分析
#    response ──长度检查──→ +0.5 或 -0.5
#           │
#           └──包含</think>? ──Yes──→ 分割思考/答案
#                                          │
#                                          ├─思考长度检查 → +1.0 或 -0.5
#                                          ├─标签数量检查 → +0.25 或 -0.25
#                                          └─更新answer
#           │
#           └──重复检查 ──→ -rep_penalty(answer)

# 3. 奖励模型评分
#    messages + answer ──reward_model.get_score()──→ score

# 4. 累加所有奖励项
#    最终rewards[response_idx] = 长度奖励 + 思考奖励 + 标签奖励 - 重复惩罚 + 模型评分

# 输出:
# rewards = [R0, R1, R2, R3, R4, R5]  # shape: [B × num_generations]



def calculate_rewards(prompts, responses, reward_model):
    #prompts: list[str], responses: list[str], reward_model: LMForRewardModel
    # responses的形状是[B*num_gen]，其中B是batch size，num_gen是每个prompt生成的回复数量。reward_model是一个用于计算奖励的模型实例。
    # 返回值: rewards，形状为 [B * num_generations] 的张量

    # # 初始化奖励张量，全为0
    # 长度 = B * num_generations
    # 后续将累加各种奖励和惩罚项

    rewards = torch.zeros(len(responses), device=args.device)

    with torch.no_grad():
        # 禁用梯度计算，因为奖励计算不需要反向传播
        # 这样可以节省内存和计算资源

        reward_model_scores = []
        # 存储奖励模型返回的分数，稍后统一转为张量
        # 这样可以批量操作，提高效率
        batch_size = len(prompts)
         # 获取原始prompt数量，即B

        for i in range(batch_size):
            # 遍历每个原始prompt
            # i: 当前prompt的索引

            for j in range(args.num_generations):
                # 遍历每个prompt生成的多个响应
                # j: 当前响应的索引（相对于当前prompt）
                # 计算在responses列表中的全局索引
                response_idx = i * args.num_generations + j
                #取出当前响应和对应的prompt
                response = responses[response_idx]
                prompt = prompts[i]
                # ========== 第一部分：解析prompt中的消息格式 ==========
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                # 正则表达式模式，用于解析特殊的消息格式
                # <|im_start|>role content<|im_end|>
                # role: system/user/assistant
                # 这个格式常用于对话系统（如ChatML格式）

                matches = re.findall(pattern, prompt, re.DOTALL)
                # 查找prompt中所有符合模式的消息
                # re.DOTALL: 使.匹配包括换行符在内的所有字符
                # matches: 列表，每个元素是(role, content)元组
                # 例如：[('user', 'What is the capital of France?')]

                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                # 构建消息列表，用于传递给奖励模型
                # 去除content前后的空白字符
                # 例如：[{"role": "user", "content": "What is the capital of France?"}]


                # 初始时将完整响应作为答案
                # 后续如果发现包含思考标签，会提取真正的答案部分
                answer = response
                
                # ========== 第二部分：基于响应长度的奖励 ==========
                # 响应长度奖励/惩罚
                # 如果响应长度在20-800字符之间，奖励+0.5
                # 否则惩罚-0.5
                # 目的：鼓励生成适中长度的响应，避免过短或过长
                rewards[response_idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5

                # ========== 第三部分：处理思考标签（</think>） ==========
                if '</think>' in response:
                    # 如果响应中包含思考标签（用于思维链推理）
                    # 格式：思考内容</think>答案内容

                    # 按第一个</think>标签分割
                    # thinking_content: 标签之前的内容（思考过程）
                    # answer_content: 标签之后的内容（最终答案）
                    # split的第二个参数1表示只分割第一次出现
                    thinking_content, answer_content = response.split('</think>', 1)

                    # 思考内容长度奖励/惩罚
                    # 鼓励思考内容在20-300字符之间
                    # 如果符合条件奖励+1.0，否则惩罚-0.5
                    rewards[response_idx] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                   
                    # 思考标签数量奖励/惩罚
                    # 期望只有一个</think>标签（符合标准格式）
                    # 如果恰好一个奖励+0.25，否则惩罚-0.25
                    rewards[response_idx] += 0.25 if response.count('</think>') == 1 else -0.25

                    # 更新answer为分割后的答案部分（去除空白）
                    # 后续只会用这部分计算奖励模型分数
                    answer = answer_content.strip()
                 # ========== 第四部分：重复惩罚 ==========    
                rewards[response_idx] -= rep_penalty(answer)
                # 应用重复惩罚
                # rep_penalty函数会检查答案中的重复内容
                # 惩罚值通常在0-1之间，从奖励中减去
                # 目的：鼓励多样性，避免重复词语或句子
                 # ========== 第五部分：奖励模型评分 ==========
                score = reward_model.get_score(messages, answer)
                # 调用奖励模型获取评分
                # messages: 对话历史（从prompt解析得到）
                # answer: 当前响应（可能是提取后的答案）
                # 返回一个标量分数，通常范围在0-1或-1到1之间
                # 例如：0.85（表示高质量回答）
                reward_model_scores.append(score)
        # ========== 第六部分：合并奖励模型分数 ==========
        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        # 将列表转换为PyTorch张量，移到指定设备
        # shape: [B * num_generations]


        # 将奖励模型分数加到总奖励上
        # 最终rewards包含了：
        # - 长度奖励/惩罚
        # - 思考标签相关奖励/惩罚
        # - 重复惩罚（负值）
        # - 奖励模型评分
        rewards += reward_model_scores
    # 返回计算好的奖励张量
    # shape: [B * num_generations]  
    return rewards


def grpo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model, start_step=0, wandb=None, use_sglang=False):
    # 训练一个epoch的GRPO
    #参数解析：
    # epoch: 当前训练的轮数
    # loader: 数据加载器，提供训练数据
    # iters: 当前epoch的总迭代次数  
    # rollout_engine: Rollout引擎实例，用于生成文本和计算log概率
    # ref_model: Reference模型，用于计算KL散度
    # reward_model: Reward模型，用于计算奖励
    # start_step: 当前epoch开始时已经完成的迭代步数（用于续训）
    # wandb: wandb实例，用于日志记录    
    # use_sglang: 是否使用SGLang作为rollout引擎
    # 返回值：无，函数内部执行训练逻辑并更新模型参数  


    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch['prompt']  # list[str], length B
        #进行tokenize并生成输入id和attention mask，注意这里使用了padding和padding_side="left"来适配不同长度的prompt
        #自动padding到当前batch中最长的prompt长度，并且将padding放在输入的左侧（即输入的右侧是有效token），
        #这样在生成时就可以直接在输入的右侧接续生成新的token，并且生成的token位置对应的attention mask为1，
        #padding使用的是tokenizer的pad_token_id，如果tokenizer没有定义pad_token_id，则默认为0。
        #不需要特殊特殊token（如[CLS]）
        # padding位置对应的attention mask为0，方便模型区分有效输入和padding。
        #返回张量格式，并将其移动到指定设备上
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)
        


        #如果设置了max_seq_len，则对输入进行截断，保留输入的右侧部分（即最新的token），确保输入长度不超过max_seq_len。
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]



        # 调用rollout引擎生成文本，并计算生成文本的log概率。
        # rollout引擎会根据输入的prompt_ids和attention_mask生成指定数量的文本（num_generations），也就是生成num_generations个不同的回复，
        # 每个回复的最大长度为max_new_tokens。
        # 每个文本的最大长度为max_new_tokens，生成过程中使用指定的temperature（越高，随机性越大）进行采样。
        # 输出是一个RolloutResult对象，包含生成的文本的token id（output_ids）、生成回复的token id（completion_ids）、
        # 每个token的log概率（per_token_logps）以及生成的文本字符串（completions）。
        rollout_result = rollout_engine.rollout(
            prompt_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            num_generations=args.num_generations,
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )

        #outputs是生成的完整文本的token id，包含了输入的prompt和生成的回复（（prompt + completion）的token id）；
        # completion_ids是生成回复的token id，不包含输入的prompt部分；
        # completions是生成回复的字符串列表；
        # old_per_token_logps是生成文本中每个token的log概率，用于后续计算损失。

        outputs = rollout_result.output_ids
        completion_ids = rollout_result.completion_ids
        completions = rollout_result.completions
        old_per_token_logps = rollout_result.per_token_logps.to(args.device)

        #如果是分布式训练，则获取模型的原始模块（即去掉DistributedDataParallel包装）。
        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model

        #  自动混合精度上下文（如果启用）
        with autocast_ctx:
            #如果使用SGLang引擎或者模型使用了MoE架构，则需要通过模型的前向函数计算logits和aux_loss（如果是MoE），
            # 否则直接使用rollout引擎返回的per_token_logps。
            if use_sglang or lm_config.use_moe:
                # 由于SGLang引擎直接返回生成文本的token id和log概率，而不是通过模型的generate函数生成，
                # 因此需要手动调用模型的前向函数来计算logits和aux_loss（如果是MoE架构）。
                
                res = model_unwrapped(outputs)
                # # MoE模型的辅助损失（用于平衡专家负载）
                aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
                # 去掉最后一个位置的logits（因为没有对应的下一个token）
                logits = res.logits[:, :-1, :]
                # 计算每个token的对数概率，因为logits的shape是[B*num_gen, seq_len-1, vocab_size]，而outputs[:, 1:]的shape是[B*num_gen, seq_len-1]，
                # gather操作提取实际生成token的log概率,因为logits对应的是每个位置预测下一个token的概率分布，而outputs[:, 1:]是实际生成的token id，
                # 所以通过gather提取出生成token对应的log概率。
                # 最后只取生成部分的log概率
                per_token_logps = F.log_softmax(logits, dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1)[:, -completion_ids.size(1):]
            else:
                aux_loss = torch.tensor(0.0, device=args.device)
                per_token_logps = rollout_result.per_token_logps
        



        with torch.no_grad():
            # 计算参考模型的每个token的log概率，用于后续计算KL散度。这里使用了与生成模型相同的输入（outputs）和生成回复的长度（completion_ids.size(1)），
            # 以确保计算的log概率与生成模型的输出对齐。
            # 参考模型的log概率用于衡量生成模型与参考模型之间的差异，进而计算KL散度作为损失的一部分，以引导生成模型不偏离参考模型。
            ref_per_token_logps = compute_per_token_logps(ref_model, outputs, completion_ids.size(1))
        #计算每个生成响应的奖励值，奖励值是根据生成的文本内容以及参考模型的评分综合计算得出的。
        rewards = calculate_rewards(prompts, completions, reward_model).to(args.device)  # [B*num_gen]

        # 训练调试输出：在debug模式下，每隔一定的step打印一次当前的prompt、生成的回复以及对应的奖励值，方便观察训练过程中的生成质量和奖励变化。
        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-'*100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i]) # 打印原始prompt
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    Logger(f"{'=' * 28} [DEBUG] gen[{j}] RESPONSE_BEGIN {'=' * 28}")
                    Logger(completions[idx]) # 打印生成的回复
                    Logger(f"{'=' * 29} [DEBUG] gen[{j}] RESPONSE_END {'=' * 29}")
                    Logger(f"[DEBUG] gen[{j}] reward={rewards[idx].item():.4f}")  # 打印奖励值
                Logger('='*100)
        # 将奖励重新组织为[B, num_generations]的形状，每个prompt对应num_generations个奖励，然后计算每个prompt的奖励的均值和标准差，
        # 并将其扩展回[B*num_gen]的形状，以便后续计算优势函数。
        grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
        #计算每个prompt对应的平均奖励，然后重复以匹配原始形状[B*num_gen]，这样每个生成的回复都对应其所属prompt的平均奖励。
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
        std_r = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
        # 计算优势函数（GRPO的核心）
        # 标准化奖励：相对于同一prompt的其他生成来说的相对优势
        advantages = (rewards - mean_r) / (std_r + 1e-4)  # [B*num_gen]


        # 标记每个位置是否为EOS token，eos_token是生成文本中的特殊token，表示文本的结束。
        # 通过比较completion_ids与tokenizer的eos_token_id来创建一个布尔张量is_eos，
        # 其形状为[B*num_gen, R]，其中R是生成回复，每个位置为True表示对应的token是EOS token。然后通过is_eos.any(dim=1)找到每个生成回复中是否存在EOS token，
        is_eos = completion_ids == tokenizer.eos_token_id  # [B*num_gen, R]

        # 初始化为最大长度（默认没有EOS）然后
        # 找到每个生成回复中第一个EOS token的位置，如果存在EOS token，
        # 则记录其位置索引；如果不存在EOS token，则将位置索引设置为生成回复的长度（即completion_ids.size(1)），
        # 这样后续计算损失时就可以正确地处理没有EOS token的情况。
        # completion_mask是一个整数张量，形状为[B*num_gen, R]，
        # 其中每个位置为1表示对应的token是有效的（即在EOS token之前），
        # 为0表示对应的token是无效的（即在EOS token之后）。这个mask用于在计算损失时只考虑有效的生成部分。
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)

        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]

        #在eos_idx之前的位置都是有效的生成token，因此completion_mask在这些位置为1，在eos_idx及之后的位置为0。
        # 这个mask将用于后续计算损失时只考虑有效的生成部分，避免对EOS token之后的无效部分进行计算。
        completion_mask = (torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)).int()  # [B*num_gen, R]


        #KL散度的近似：参考模型和当前模型log概率的差值，
        kl_div = ref_per_token_logps - per_token_logps
        # 更精确的KL散度计算：exp(Δ) - Δ - 1
        # 当Δ=0时，KL≈0
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B*num_gen, R]

        #重要性采样权重：当前模型和旧模型log概率的差值的指数，即生成文本在当前模型下的概率与在旧模型下的概率之比。
        #重要性采样比率：π_θ / π_old = exp(logπ_θ - logπ_old)，这里的logπ_θ是当前模型的log概率，logπ_old是旧模型的log概率。
        ratio = torch.exp(per_token_logps - old_per_token_logps)  # [B*num_gen, R]

        #如果使用CISPO loss，则直接使用clamped_ratio乘以优势函数，并减去KL惩罚项；
        # 如果使用GRPO loss，则先计算PPO的clip版本的损失，然后取两者的最小值，并减去KL惩罚项。
        if args.loss_type == "cispo":
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            #公式：per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) - args.beta * per_token_kl)
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
        else:
            # PPO的clip版本损失：ratio * advantages 和 clip(ratio, 1-epsilon, 1+epsilon) * advantages 取最小值，防止过大更新，同时减去KL惩罚项。
            #限制比率在[1-ε, 1+ε]范围内
            #公式：per_token_loss1 = ratio * advantages.unsqueeze(1)
            #      per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
        
        # 计算策略损失：
        # 1. 应用mask（只计算有效token）
        # 2. 对每个序列取平均
        # 3. 对所有序列取平均
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        # 总损失，除以梯度累积步数
        loss = (policy_loss + aux_loss) / args.accumulation_steps  # scalar
        loss.backward()


        # 每隔一定的step进行一次优化器更新（根据梯度累积步数）。
        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)# 梯度裁剪，防止梯度爆炸
            optimizer.step()# 更新模型参数
            scheduler.step()# 更新学习率调度器
            optimizer.zero_grad()# 清零梯度
            if is_main_process() and step % args.save_interval == 0: rollout_engine.update_policy(model)# 更新rollout引擎中的策略模型


        # 每隔一定的step打印一次日志，包括当前的奖励、KL散度、优势函数的均值和标准差、策略损失、平均生成长度以及当前学习率等指标，方便监控训练过程。
        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            kl_ref_val = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / completion_mask.sum().item()
            advantages_mean_val = advantages.mean().item()
            advantages_std_val = advantages.std().item()
            current_lr = optimizer.param_groups[0]['lr']

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Reward: {avg_reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, '
                   f'Adv Std: {advantages_std_val:.4f}, Adv Mean: {advantages_mean_val:.4f}, '
                   f'Actor Loss: {policy_loss_val:.4f}, Avg Response Len: {avg_len_val:.2f}, Learning Rate: {current_lr:.8f}')

            if wandb and is_main_process():
                wandb.log({
                    "reward": avg_reward_val,
                    "kl_ref": kl_ref_val,
                    "advantages_std": advantages_std_val,
                    "advantages_mean": advantages_mean_val,
                    "policy_loss": policy_loss_val,
                    "avg_response_len": avg_len_val,
                    "learning_rate": current_lr
                })
        # 每隔一定的step保存一次模型权重和训练状态，便于后续恢复训练或进行评估。保存的内容包括模型权重、优化器状态、学习率调度器状态、当前epoch和step等信息。
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict
    # 处理最后一个batch如果未达到累积步数的情况，确保最后的优化器更新和模型保存。
    # 如果当前step大于start_step，并且当前step不是累积步数的整数倍（即还有未更新的梯度），则进行一次优化器更新和模型保存，确保训练状态的完整性。
    # 这段代码主要是为了处理在训练过程中，如果最后一个batch的step没有达到梯度累积步数的整数倍，
    # 导致最后的优化器更新和模型保存没有执行的情况。
    # 通过这个条件判断，可以确保即使在训练结束时还有未更新的梯度，
    # 也能进行一次优化器更新和模型保存，保证训练状态的完整性和可恢复性。
    if step > start_step and step % args.accumulation_steps != 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if is_main_process() and step % args.save_interval == 0: rollout_engine.update_policy(model)

        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind GRPO (Group Relative Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl", help="RLAIF数据路径")
    parser.add_argument("--num_generations", type=int, default=6, help="每个prompt生成的样本数")
    parser.add_argument("--beta", type=float, default=0.1, help="KL惩罚系数")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="loss类型")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="epsilon上界")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--rollout_engine", type=str, default="sglang", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8996", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_grpo", help="SGLang共享存储路径")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型和数据 ==========
    base_weight = args.from_weight
    # Policy模型
    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    # Reference模型
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    # Reward模型
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    # Rollout引擎（可插拔替换，只负责 policy 推理）
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    # 数据和优化器
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len, thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(model)
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    if is_main_process(): rollout_engine.update_policy(model)
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            grpo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model, start_step, wandb, use_sglang = (args.rollout_engine == "sglang"))
        else:
            grpo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model, 0, wandb, use_sglang = (args.rollout_engine == "sglang"))
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()