import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


class SequenceBCDataset(Dataset):
    """
    将 Panda Lift BC 数据构造成固定长度 observation 序列。

    原始 observations: (N, obs_dim)
    原始 actions:      (N, action_dim)
    返回 obs_seq:      (seq_len, obs_dim)
    返回 target_action:(action_dim,)
    """

    def __init__(self, data_path, seq_len):
        # seq_len 表示模型一次看到多少个连续 observation。
        # 例如 seq_len=16 时，用过去 16 帧状态预测最后一帧对应的专家动作。
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")

        # 数据文件来自 collect_panda_lift_bc_data_with_metrics.py。
        # Transformer 训练只需要样本级的 observations / actions；
        # episode 级指标会保留在 npz 里，但这里不参与训练。
        data = np.load(data_path)
        self.observations = data["observations"].astype(np.float32)
        self.actions = data["actions"].astype(np.float32)
        self.seq_len = int(seq_len)

        # 做一些基础检查，尽早暴露数据格式问题。
        if self.observations.ndim != 2:
            raise ValueError(
                f"observations must be 2D [N, obs_dim], got {self.observations.shape}"
            )
        if self.actions.ndim != 2:
            raise ValueError(f"actions must be 2D [N, action_dim], got {self.actions.shape}")
        if len(self.observations) != len(self.actions):
            raise ValueError(
                "observations and actions must have the same first dimension, "
                f"got {len(self.observations)} and {len(self.actions)}"
            )

        # 如果数据里有 episode_ids，就必须按 episode 构造序列。
        # 这样不会把上一个 episode 的结尾和下一个 episode 的开头拼在一起。
        self.episode_ids = None
        if "episode_ids" in data.files:
            self.episode_ids = data["episode_ids"]
            if len(self.episode_ids) != len(self.observations):
                raise ValueError(
                    "episode_ids must have the same length as observations, "
                    f"got {len(self.episode_ids)} and {len(self.observations)}"
                )

        # sequence_slices 里只保存索引范围，不复制真实数据；
        # __getitem__ 时再按索引切片，内存更省。
        self.sequence_slices = self._build_sequence_slices()
        if not self.sequence_slices:
            raise ValueError(
                "No valid sequences were created. "
                f"Try a smaller seq_len than {self.seq_len}."
            )

        print(f"Loaded observations: {self.observations.shape}")
        print(f"Loaded actions: {self.actions.shape}")
        print(f"seq_len: {self.seq_len}")
        print(f"Sequence samples: {len(self.sequence_slices)}")

    def _build_episode_ranges(self):
        # 老版本 BC 数据可能没有 episode_ids。
        # 没有时退化成把整段数据当作一条连续轨迹。
        if self.episode_ids is None:
            return [(0, len(self.observations))]

        # episode_ids 是按采集顺序保存的，例如：
        # [0, 0, 0, ..., 1, 1, 1, ..., 2, 2, 2]
        # 这里把它转换成每个连续 episode chunk 的 [start, end) 范围。
        ranges = []
        start = 0
        for idx in range(1, len(self.episode_ids)):
            if self.episode_ids[idx] != self.episode_ids[idx - 1]:
                ranges.append((start, idx))
                start = idx
        ranges.append((start, len(self.episode_ids)))

        unique_episode_count = len(np.unique(self.episode_ids))
        print("Found episode_ids in data.")
        print(f"Unique episode ids: {unique_episode_count}")
        print(f"Contiguous episode chunks: {len(ranges)}")
        return ranges

    def _build_sequence_slices(self):
        """
        每个样本使用 [t - seq_len + 1, ..., t] 的 obs 预测 action[t]。

        如果有 episode_ids，只在同一个连续 episode chunk 内滑窗，保证不会跨 episode。
        """
        sequence_slices = []
        skipped_short_episodes = 0

        for start, end in self._build_episode_ranges():
            episode_len = end - start
            if episode_len < self.seq_len:
                skipped_short_episodes += 1
                continue

            for target_idx in range(start + self.seq_len - 1, end):
                seq_start = target_idx - self.seq_len + 1
                seq_end = target_idx + 1
                sequence_slices.append((seq_start, seq_end, target_idx))

        if skipped_short_episodes > 0:
            print(
                f"Skipped {skipped_short_episodes} episode chunks shorter than "
                f"seq_len={self.seq_len}."
            )

        return sequence_slices

    def __len__(self):
        # Dataset 的长度是可训练序列样本数，不是原始 step 数。
        return len(self.sequence_slices)

    def __getitem__(self, idx):
        # DataLoader 会调用这里，取出一个形状为 [seq_len, obs_dim] 的序列，
        # 以及这个序列最后一个时间步对应的专家 action。
        seq_start, seq_end, target_idx = self.sequence_slices[idx]
        obs_seq = self.observations[seq_start:seq_end]
        target_action = self.actions[target_idx]
        return torch.from_numpy(obs_seq), torch.from_numpy(target_action)


class TransformerBCPolicy(nn.Module):
    """
    Transformer behavior cloning policy.

    输入 observation 序列，输出序列最后一个时间步对应的 action。
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        seq_len,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()

        # PyTorch Multi-Head Attention 要求 d_model 能被 head 数整除。
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")

        # 把原始 observation vector 投影到 Transformer 使用的 hidden size。
        self.obs_embedding = nn.Linear(obs_dim, d_model)

        # 可学习的位置编码，让模型知道序列中每一帧的先后顺序。
        # 形状是 [1, seq_len, d_model]，第一个维度用于自动 broadcast 到 batch。
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))

        # batch_first=True 表示输入输出都使用 [batch, seq_len, d_model]。
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # 最后只取序列末尾 hidden state，再映射成当前动作。
        self.action_head = nn.Linear(d_model, action_dim)

        # 位置编码用较小随机值初始化，避免一开始扰动过大。
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

    def forward(self, obs_seq):
        # obs_seq: [batch_size, seq_len, obs_dim]
        # hidden: [batch_size, seq_len, d_model]
        hidden = self.obs_embedding(obs_seq)

        # 给每个时间步加上对应的位置向量。
        hidden = hidden + self.pos_embedding[:, : hidden.size(1), :]

        # TransformerEncoder 会在序列内部做 self-attention，
        # 让每个时间步可以利用前后文信息。
        hidden = self.transformer_encoder(hidden)

        # Behavior Cloning 的标签是最后一个时间步的 action，
        # 所以这里只取最后一个 hidden state 做预测。
        last_hidden = hidden[:, -1, :]
        return self.action_head(last_hidden)


def train_transformer_bc(
    data_path="data/panda_lift_bc_data_with_metrics_10ep.npz",
    save_path="models/bc/panda_lift_transformer_bc_policy.pt",
    seq_len=16,
    d_model=128,
    nhead=4,
    num_layers=2,
    dim_feedforward=256,
    dropout=0.1,
    epochs=50,
    batch_size=128,
    lr=1e-3,
):
    # 固定随机种子，让 train/val 划分和初始化更容易复现实验。
    torch.manual_seed(0)
    np.random.seed(0)

    # 1. 加载原始 BC 数据，并构造序列样本。
    dataset = SequenceBCDataset(data_path=data_path, seq_len=seq_len)

    if len(dataset) < 2:
        raise ValueError("Need at least 2 sequence samples for a train/val split.")

    # 自动从数据形状读取 observation 维度和 action 维度。
    obs_dim = dataset.observations.shape[1]
    action_dim = dataset.actions.shape[1]

    print(f"obs_dim: {obs_dim}")
    print(f"action_dim: {action_dim}")

    # 2. 按 90% / 10% 划分训练集和验证集。
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    if train_size == 0 or val_size == 0:
        val_size = 1
        train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(0),
    )

    # 训练集打乱顺序；验证集只评估，不需要打乱。
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # 3. 优先使用 GPU，没有 GPU 就使用 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 4. 创建 Transformer BC policy。
    policy = TransformerBCPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        seq_len=seq_len,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device)

    # Adam 负责更新参数；MSELoss 衡量预测动作和专家动作的均方误差。
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # 只保存验证集 loss 最低的 checkpoint。
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # ===== 训练阶段 =====
        policy.train()
        train_loss_sum = 0.0

        for obs_seq_batch, action_batch in train_loader:
            # batch 数据和模型必须放在同一个 device 上。
            obs_seq_batch = obs_seq_batch.to(device)
            action_batch = action_batch.to(device)

            # 前向传播：用 observation 序列预测最后一步动作。
            pred_action = policy(obs_seq_batch)
            loss = loss_fn(pred_action, action_batch)

            # 标准反向传播三步：清梯度、算梯度、更新参数。
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss.item() 是 batch 平均 loss，乘以 batch 大小后累计，
            # 最后除以样本总数得到 epoch 平均 loss。
            train_loss_sum += loss.item() * obs_seq_batch.size(0)

        train_loss = train_loss_sum / len(train_dataset)

        # ===== 验证阶段 =====
        policy.eval()
        val_loss_sum = 0.0

        # 验证时不需要梯度，可以节省显存和计算。
        with torch.no_grad():
            for obs_seq_batch, action_batch in val_loader:
                obs_seq_batch = obs_seq_batch.to(device)
                action_batch = action_batch.to(device)

                pred_action = policy(obs_seq_batch)
                loss = loss_fn(pred_action, action_batch)

                val_loss_sum += loss.item() * obs_seq_batch.size(0)

        val_loss = val_loss_sum / len(val_dataset)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f}"
        )

        # 当前验证集表现更好时，覆盖保存最佳 checkpoint。
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            torch.save(
                {
                    # 保存 state_dict 比保存整个模型对象更轻量、更稳定。
                    "model_state_dict": policy.state_dict(),
                    # 下面这些维度和超参数用于之后重建同结构模型。
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "seq_len": seq_len,
                    "d_model": d_model,
                    "nhead": nhead,
                    "num_layers": num_layers,
                    "dim_feedforward": dim_feedforward,
                    "dropout": dropout,
                    "best_val_loss": best_val_loss,
                },
                save_path,
            )

    print("\n===== Transformer BC training finished =====")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Saved policy to: {save_path}")


def parse_args():
    # 命令行参数都有默认值，所以直接运行
    # python src/train_panda_lift_transformer_bc.py
    # 就会使用题目要求的数据路径和模型保存路径。
    parser = argparse.ArgumentParser(
        description="Train Panda Lift Transformer BC policy."
    )
    parser.add_argument(
        "--data",
        default="data/panda_lift_bc_data_with_metrics_10ep.npz",
        help="Path to the BC training .npz data file.",
    )
    parser.add_argument(
        "--output",
        default="models/bc/panda_lift_transformer_bc_policy.pt",
        help="Path to save the trained Transformer BC policy checkpoint.",
    )
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Training data path: {args.data}")
    print(f"Model save path: {args.output}")

    train_transformer_bc(
        data_path=args.data,
        save_path=args.output,
        seq_len=args.seq_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
