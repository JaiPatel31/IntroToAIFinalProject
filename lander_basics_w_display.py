import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt #for displaying frames


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)

# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buffer)

# Change render_mode to "rgb_array" to allow frame capture
env = gym.make("LunarLander-v3", render_mode="rgb_array")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

q_net = QNetwork(state_dim, action_dim).to(device)
target_net = QNetwork(state_dim, action_dim).to(device)
target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
buffer = ReplayBuffer()

gamma = 0.99
batch_size = 64
epsilon_start, epsilon_end, epsilon_decay = 1.0, 0.05, 50000
global_step = 0
target_update_freq = 1000

def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = q_net(s)
        return int(torch.argmax(q_values, dim=1).item())

num_episodes = 1000
episode_rewards = []

for ep in range(num_episodes):
    state, info = env.reset()
    done = False
    ep_reward = 0

    # Initialize a list to store frames for visualization for this episode
    current_episode_frames = []
    # Decide if we should record frames for this episode (e.g., only for the first episode)
    record_frames = (ep == 0) # Only record and display frames for the very first episode

    while not done:
        # Render and collect frames if recording is enabled for this episode
        if record_frames:
            frame = env.render()
            current_episode_frames.append(frame)

        epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                  np.exp(-1.0 * global_step / epsilon_decay)

        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Optional: modify reward or state here for experiments
        buffer.push(state, action, reward, next_state, done)
        state = next_state
        ep_reward += reward
        global_step += 1

        # Update step
        if len(buffer) >= batch_size:
            s, a, r, s2, d = buffer.sample(batch_size)
            s  = torch.tensor(s,  dtype=torch.float32, device=device)
            a  = torch.tensor(a,  dtype=torch.int64,   device=device)
            r  = torch.tensor(r,  dtype=torch.float32, device=device)
            s2 = torch.tensor(s2, dtype=torch.float32, device=device)
            d  = torch.tensor(d,  dtype=torch.float32, device=device)

            q_values = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                max_next_q = target_net(s2).max(1)[0]
                target = r + gamma * max_next_q * (1 - d)

            loss = nn.MSELoss()(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % target_update_freq == 0:
                target_net.load_state_dict(q_net.state_dict())

    episode_rewards.append(ep_reward)
    print(f"Episode {ep}, reward: {ep_reward:.1f}, epsilon: {epsilon:.3f}")

    # Display collected frames for the recorded episode
    if record_frames and len(current_episode_frames) > 0:
        print(f"\nDisplaying frames from Episode {ep}:")
        num_frames_to_display = 5
        indices = np.linspace(0, len(current_episode_frames) - 1, num_frames_to_display, dtype=int)

        plt.figure(figsize=(15, 5))
        for i, idx in enumerate(indices):
            plt.subplot(1, num_frames_to_display, i + 1)
            plt.imshow(current_episode_frames[idx])
            plt.title(f'Frame {idx+1}')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
        print("\n") # Add a newline for better separation

env.close()
