import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
from main import DoomEnv

# 定義 DQN 模型
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(np.prod(input_shape), 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = x.float() / 255  # 標準化輸入
        x = x.view(x.size(0), -1)  # 展平圖像
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 簡單的記憶回放
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)

# 訓練函數
def train(model, optimizer, criterion, replay_buffer, batch_size):
    if len(replay_buffer) < batch_size:
        return

    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.tensor(state, dtype=torch.float32)
    next_state = torch.tensor(next_state, dtype=torch.float32)
    action = torch.tensor(action, dtype=torch.int64)
    reward = torch.tensor(reward, dtype=torch.float32)
    done = torch.tensor(done, dtype=torch.float32)

    q_values = model(state)
    next_q_values = model(next_state)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + 0.99 * next_q_value * (1 - done)

    loss = criterion(q_value, expected_q_value)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 主函數
def main():
    env = DoomEnv()  # 假設您已經創建了 DoomEnv 類
    model = DQN(env.observation_space.shape, env.action_space.n)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    save_path = './model'
    replay_buffer = ReplayBuffer(10000)

    num_episodes = 1000
    batch_size = 32
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        print("model exit")
    else:
        print("model dosen't exit")
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).max(1)[1].item()
            next_state, reward, done, _ = env.step(action)

            replay_buffer.push(state, action, reward, next_state, done)

            train(model, optimizer, criterion, replay_buffer, batch_size)

            state = next_state
            total_reward += reward

            if done:
                break

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

        if episode%100 == 0:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
                torch.save(model.state_dict(),save_path)
                model.load_state_dict(torch.load(save_path))
            else:
                torch.save(model.state_dict(),save_path)
                model.load_state_dict(torch.load(save_path))
                print("model has been save")
                
    env.close()


if __name__ == "__main__":
    main()
