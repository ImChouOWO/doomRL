import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
from main import DoomEnv
import matplotlib.pyplot as plt
import statistics


# 定義 DQN 模型
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(16000, 128)
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
gradients = []


def train(model, optimizer, criterion, replay_buffer, batch_size,device):
    
    if len(replay_buffer) < batch_size:
        return

    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.tensor(state, dtype=torch.float32).to(device)
    next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
    action = torch.tensor(action, dtype=torch.int64).to(device)
    reward = torch.tensor(reward, dtype=torch.float32).to(device)
    done = torch.tensor(done, dtype=torch.float32).to(device)

    q_values = model(state)
    next_q_values = model(next_state)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + 0.99 * next_q_value * (1 - done)

    loss = criterion(q_value, expected_q_value)
    optimizer.zero_grad()
    loss.backward()

    gradients.append(model.fc1.weight.grad.norm().item())


    optimizer.step()


def draw_chart(data,epoch,num):
    save_path =f'./img/reward/reward_trend_with_{epoch*num}_epoch.png'
    plt.plot(data)
    plt.title('Reward Trend')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.savefig(save_path)

def draw_gradient(data,epoch,num):
    save_path =f'./img/gradient/gradent_with_{epoch*num}_epoch.png'
    plt.plot(data)
    plt.title("Gradient Norm During Training")
    plt.xlabel("Training Steps")
    plt.ylabel("Gradient Norm")
    plt.savefig(save_path)



# 主函數
def main():
    global gradients
    # apply to apple silicon
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"MPS available :{torch.backends.mps.is_available()}")

    # apply to Nvidia
    # device = torch.device("CUDA" if torch.backends.mps.is_available() else "cpu")


    env = DoomEnv() 
    model = DQN(env.observation_space.shape, env.action_space.n).to(device)

    learning_rate = 0.01
    weight_decary  = 1e-5

    optimizer = optim.Adam(model.parameters(),lr = learning_rate)
    criterion = nn.MSELoss()
    save_path = './model/model.pth'
    
    replay_buffer = ReplayBuffer(10000)
    reward_list = []
    num_episodes = 10000
    batch_size = 32
    tmp_score = []

    now_round =3

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        print("Model loaded")
    else:
        print("Model does not exist")
    gradients = []
    tmp = 0
    for episode in range(num_episodes):
        
        state = env.reset()
        total_reward = 0

        while True:
            # 選擇動作
            action = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)).max(1)[1].item()
            next_state, reward, done, _ = env.step(action)

            replay_buffer.push(state, action, reward, next_state, done)

            train(model, optimizer, criterion, replay_buffer, batch_size,device)

            state = next_state
            total_reward += reward

            
            if done:
                break
        tmp_score.append(total_reward)    
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
        
        if episode % 100 == 0:
            
            reward_list.append(statistics.mean(tmp_score))
            tmp_score = []
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print("Model has been saved")

        if episode %1000 == 0:
            tmp +=1
            rollback_save_path = f'./model/rollback/model_epoch_{tmp*1000}.pth'
            torch.save(model.state_dict(), rollback_save_path)

    draw_chart(reward_list,num_episodes,now_round)
    draw_gradient(gradients,num_episodes,now_round)        
    env.close()


if __name__ == "__main__":
    main()
