import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# --------------------
# Hyperparameters
# --------------------
ENV_ID = "LunarLander-v3"

GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64

REPLAY_CAPACITY = 100_000
NUM_EPISODES = 1000

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 50_000

TARGET_UPDATE_FREQ = 1000  # steps
EVAL_EPISODES = 100

# Flag for reward shaping experiment
USE_SHAPED_REWARD = False  # baseline: False; experiment: True


# --------------------
# DQN Network
# --------------------
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


# --------------------
# Replay Buffer
# --------------------
class ReplayBuffer:
    def __init__(self, capacity=REPLAY_CAPACITY):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buffer)


# --------------------
# Epsilon-greedy Policy
# --------------------
def compute_epsilon(global_step: int) -> float:
    return EPS_END + (EPS_START - EPS_END) * np.exp(-1.0 * global_step / EPS_DECAY_STEPS)


def select_action(q_net, state, epsilon, env, device):
    if random.random() < epsilon:
        return env.action_space.sample()
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = q_net(s)
        return int(torch.argmax(q_values, dim=1).item())


# --------------------
# Reward Shaping (Experiment)
# --------------------
def shape_reward(state, action, reward, done):
    """
    Example reward modification experiment:
      - Extra penalty for being far from center (x-position).
      - Extra penalty for firing engines (fuel cost).
    """
    # state[0] = horizontal position (x), state[1] = vertical position (y)
    x_pos = state[0]
    distance_from_center = abs(x_pos)

    shaped_reward = reward

    # Penalty for distance from center (encourages hovering near x=0)
    shaped_reward -= 0.3 * distance_from_center

    # Simple fuel cost: penalize firing main/side engines
    # Lander actions: 0 = do nothing, 1 = fire left engine,
    # 2 = fire main engine, 3 = fire right engine
    if action in [1, 2, 3]:
        shaped_reward -= 0.05  # small constant fuel penalty

    return shaped_reward


# --------------------
# Training Loop
# --------------------
def train_dqn():
    # Training env (no human render)
    env = gym.make(ENV_ID)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    buffer = ReplayBuffer()

    global_step = 0
    episode_rewards = []

    for ep in range(NUM_EPISODES):
        state, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            epsilon = compute_epsilon(global_step)
            action = select_action(q_net, state, epsilon, env, device)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Optionally modify reward for the experiment
            if USE_SHAPED_REWARD:
                reward_to_store = shape_reward(state, action, reward, done)
            else:
                reward_to_store = reward

            buffer.push(state, action, reward_to_store, next_state, done)
            state = next_state
            ep_reward += reward
            global_step += 1

            # Gradient update step
            if len(buffer) >= BATCH_SIZE:
                s, a, r, s2, d = buffer.sample(BATCH_SIZE)
                s  = torch.tensor(s,  dtype=torch.float32, device=device)
                a  = torch.tensor(a,  dtype=torch.int64,   device=device)
                r  = torch.tensor(r,  dtype=torch.float32, device=device)
                s2 = torch.tensor(s2, dtype=torch.float32, device=device)
                d  = torch.tensor(d,  dtype=torch.float32, device=device)

                q_values = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    max_next_q = target_net(s2).max(1)[0]
                    target = r + GAMMA * max_next_q * (1 - d)

                loss = nn.MSELoss()(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Target network update
                if global_step % TARGET_UPDATE_FREQ == 0:
                    target_net.load_state_dict(q_net.state_dict())

        episode_rewards.append(ep_reward)
        print(f"Episode {ep+1}/{NUM_EPISODES}, reward: {ep_reward:.1f}, epsilon: {epsilon:.3f}")

    env.close()
    return q_net, episode_rewards, device


# --------------------
# Plot Learning Curves
# --------------------
def plot_learning_curve(rewards, window=100, title_suffix="(baseline)"):
    episodes = np.arange(len(rewards)) + 1
    plt.figure()
    plt.plot(episodes, rewards, label="Episode reward")

    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(
            np.arange(window, len(rewards) + 1),
            moving_avg,
            label=f"{window}-episode moving average"
        )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"LunarLander DQN Training {title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.show()


# --------------------
# Evaluation (ε ≈ 0)
# --------------------
def evaluate_policy(q_net, device, render=False, num_episodes=EVAL_EPISODES):
    env = gym.make(ENV_ID, render_mode="human" if render else None)
    rewards = []
    successes = 0

    for ep in range(num_episodes):
        state, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            # Pure exploitation (epsilon ~ 0)
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = q_net(s)
                action = int(torch.argmax(q_values, dim=1).item())

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            ep_reward += reward

        rewards.append(ep_reward)

        # Approximate "success": reward ≥ 200
        if ep_reward >= 200.0:
            successes += 1

        print(f"[EVAL] Episode {ep+1}/{num_episodes}, reward: {ep_reward:.1f}")

    env.close()

    avg_reward = np.mean(rewards)
    success_rate = successes / num_episodes * 100.0
    print(f"\nEvaluation over {num_episodes} episodes:")
    print(f"  Average reward: {avg_reward:.2f}")
    print(f"  Approx. success rate: {success_rate:.1f}% "
          f"(episodes with reward ≥ 200)")
    return avg_reward, success_rate


# --------------------
# Visual Demo Episode (nice for video)
# --------------------
def run_visual_episode(q_net, device):
    env = gym.make(ENV_ID, render_mode="human")
    state, info = env.reset()
    done = False
    ep_reward = 0.0

    while not done:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = q_net(s)
            action = int(torch.argmax(q_values, dim=1).item())

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state
        ep_reward += reward

    print(f"[VISUAL] Episode reward: {ep_reward:.1f}")
    env.close()


# --------------------
# Main
# --------------------
if __name__ == "__main__":
    print(f"Training DQN on {ENV_ID} | Reward shaping: {USE_SHAPED_REWARD}")
    q_net, episode_rewards, device = train_dqn()

    # Plot learning curve
    title = "(Reward Shaping)" if USE_SHAPED_REWARD else "(Baseline)"
    plot_learning_curve(episode_rewards, window=100, title_suffix=title)

    # Evaluation: pure exploitation, no learning
    avg_reward, success_rate = evaluate_policy(q_net, device, render=False)

    # Optional: run 1–2 visual episodes for your demo video
    # run_visual_episode(q_net, device)
