import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


class BCDataset(Dataset):
    """
    读取 BC 数据集。

    observations: (N, obs_dim)
    actions:      (N, action_dim)
    """

    def __init__(self, data_path):
        # 读取 collect_panda_lift_bc_data.py 保存出来的 .npz 数据文件。
        data = np.load(data_path)

        # observations 是神经网络输入，actions 是专家策略给出的目标动作。
        # 统一转成 float32，方便后面转换成 PyTorch tensor 进行训练。
        self.observations = data["observations"].astype(np.float32)
        self.actions = data["actions"].astype(np.float32)

        print(f"Loaded observations: {self.observations.shape}")
        print(f"Loaded actions: {self.actions.shape}")

    def __len__(self):
        # 返回数据集中一共有多少条 observation-action 样本。
        return len(self.observations)

    def __getitem__(self, idx):
        # 根据索引 idx 取出一条训练样本：
        # obs 是当前状态，action 是专家在这个状态下执行的动作。
        obs = self.observations[idx]
        action = self.actions[idx]
        return obs, action


class MLPPolicy(nn.Module):
    """
    MLP policy:
    输入 obs vector，输出 action vector。
    """

    def __init__(self, obs_dim, action_dim):
        super().__init__()

        # 一个简单的 MLP 策略网络：
        # 输入 observation vector，经过两层隐藏层，输出 action vector。
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            # 最后一层输出动作维度，和环境 action_dim 保持一致。
            nn.Linear(256, action_dim),
            # robosuite 的动作通常限制在 [-1, 1]，Tanh 正好把输出压到这个范围。
            nn.Tanh(),
        )

    def forward(self, obs):
        # 前向传播：输入一批 obs，输出对应预测动作。
        return self.net(obs)


def train_bc(
    data_path="data/panda_lift_bc_data.npz",
    save_path="models/bc/panda_lift_bc_policy.pt",
    epochs=50,
    batch_size=256,
    lr=1e-3,
):
    # 1. 加载行为克隆数据集。
    dataset = BCDataset(data_path)

    # 自动从数据形状里读取 observation 维度和 action 维度，
    # 这样后面创建 MLP 时不用手动填写。
    obs_dim = dataset.observations.shape[1]
    action_dim = dataset.actions.shape[1]

    print(f"obs_dim: {obs_dim}")
    print(f"action_dim: {action_dim}")

    # 2. 按 9:1 把数据集划分成训练集和验证集。
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        # 固定随机种子，让每次划分结果一致，方便复现实验。
        generator=torch.Generator().manual_seed(0),
    )

    # DataLoader 会自动调用 Dataset.__getitem__，
    # 每次取出 batch_size 条样本组成一个 batch。
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        # 训练集每个 epoch 都打乱顺序，避免模型依赖固定数据顺序。
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        # 验证集只用于评估，不需要打乱。
        shuffle=False,
    )

    # 3. 优先使用 GPU，没有 GPU 就使用 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建策略网络，并移动到对应设备上。
    policy = MLPPolicy(obs_dim, action_dim).to(device)

    # Adam 负责根据梯度更新网络参数；MSELoss 用来比较预测动作和专家动作的差距。
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # 记录目前为止最好的验证集 loss，只保存验证效果最好的模型。
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # ===== 训练阶段 =====
        policy.train()
        train_loss_sum = 0.0

        for obs_batch, action_batch in train_loader:
            # batch 数据和模型必须在同一个设备上。
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)

            # 用当前策略网络预测动作。
            pred_action = policy(obs_batch)

            # 计算预测动作和专家动作之间的均方误差。
            loss = loss_fn(pred_action, action_batch)

            # 标准反向传播三步：
            # 1. 清空旧梯度；2. 计算新梯度；3. 更新网络参数。
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss.item() 是当前 batch 的平均 loss。
            # 乘以 batch 样本数后累加，最后再除以总样本数，得到整个训练集平均 loss。
            train_loss_sum += loss.item() * obs_batch.size(0)

        train_loss = train_loss_sum / len(train_dataset)

        # ===== 验证阶段 =====
        policy.eval()
        val_loss_sum = 0.0

        # 验证时不更新参数，因此不需要计算梯度，可以节省显存和时间。
        with torch.no_grad():
            for obs_batch, action_batch in val_loader:
                obs_batch = obs_batch.to(device)
                action_batch = action_batch.to(device)

                pred_action = policy(obs_batch)
                loss = loss_fn(pred_action, action_batch)

                val_loss_sum += loss.item() * obs_batch.size(0)

        val_loss = val_loss_sum / len(val_dataset)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f}"
        )

        # 如果当前模型在验证集上更好，就保存 checkpoint。
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            # 确保保存模型的目录存在。
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            torch.save(
                {
                    # 只保存模型参数，不保存整个 Python 对象，更轻量也更常见。
                    "model_state_dict": policy.state_dict(),
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "best_val_loss": best_val_loss,
                },
                save_path,
            )

    print("\n===== BC training finished =====")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Saved policy to: {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Panda Lift BC policy.")
    parser.add_argument(
        "--data",
        default="data/panda_lift_bc_data.npz",
        help="Path to the BC training .npz data file.",
    )
    parser.add_argument(
        "--output",
        default="models/bc/panda_lift_bc_policy.pt",
        help="Path to save the trained BC policy checkpoint.",
    )
    args = parser.parse_args()

    print(f"Training data path: {args.data}")
    print(f"Model save path: {args.output}")

    train_bc(data_path=args.data, save_path=args.output)
