import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np

class ActorCritic(nn.Module):
    # 모델 구현
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.action_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)
    # 순전파 구현
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.action_head(x)
        value = self.value_head(x)
        return logits, value
    # 순전파
    def get_action_value(self, state):
        # 상태 -> 행동 확률, 상태 가치, 행동 샘플 및 log_prob 반환
        logits, value = self.forward(state)
        action_probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()          # action_probs의 확률에 대한 확률로 샘플링 
        return action, dist.log_prob(action), value, action_probs
    # log_prob = log를 씌워서 음수로 최대
    def get_logprob_value(self, state, action):
        logits, value = self.forward(state)
        action_probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        return dist.log_prob(action), value

class PPO:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=3e-4, clip_epsilon=0.2, k_epochs=4, rollout_steps=2048):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.k_epochs = k_epochs
        self.rollout_steps = rollout_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ac = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
        # 롤아웃 저장용 버퍼
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.done_flags = []
        self.values = []
    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, logprob, value, _ = self.ac.get_action_value(state_t)
        self.states.append(state)
        self.actions.append(action.item())
        self.logprobs.append(logprob.item())
        self.values.append(value.item())
        return action.item()
    def store_reward(self, reward, done):
        self.rewards.append(reward)
        self.done_flags.append(done)
    def compute_returns_advantages(self, next_state):
        # GAE 등 사용 가능하지만 여기서는 단순 n-step Return
        # next_value가 done이면 0, 아니면 value(s')
        if self.done_flags[-1]:
            next_value = 0
        else:
            state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, next_value_t = self.ac.forward(state_t)
            next_value = next_value_t.item()
        returns = []
        G = next_value
        # done 역순으로 r을 계산 처음은 0
        for r, done, val in reversed(list(zip(self.rewards, self.done_flags, self.values))):
            if done:
                G = r
            else:
                G = r + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns)
        values = np.array(self.values)
        advantages = returns - values
        # 정규화(Optional)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
        return returns, advantages
    def update(self, next_state):
        returns, advantages = self.compute_returns_advantages(next_state)
        states_t = torch.FloatTensor(self.states).to(self.device)
        actions_t = torch.LongTensor(self.actions).to(self.device)
        old_logprobs_t = torch.FloatTensor(self.logprobs).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        # 여러 epoch 업데이트
        for _ in range(self.k_epochs):
            logprobs, values_t = self.ac.get_logprob_value(states_t, actions_t)
            ratio = torch.exp(logprobs - old_logprobs_t)
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_t
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values_t.squeeze(), returns_t)
            loss = actor_loss + 0.5 * critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # 버퍼 정리
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.done_flags = []
        self.values = []

class TradingEnv:
    def __init__(self,):
        
max_episodes=300
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = PPO(state_dim, action_dim)
reward_history = []
state,_ = env.reset()
episode_rewards = 0
for ep in range(max_episodes):
    for step in range(agent.rollout_steps):
        action = agent.select_action(state)
        next_state, reward, done, info, _ = env.step(action)
        agent.store_reward(reward, done)
        state = next_state
        episode_rewards += reward
        if done:
            reward_history.append(episode_rewards)
            episode_rewards = 0
            state,_ = env.reset()
    # 수집한 rollout_steps만큼의 데이터로 업데이트
    agent.update(next_state)
    if (ep+1) % 20 == 0:
        avg_reward = np.mean(reward_history[-20:]) if len(reward_history)>20 else np.mean(reward_history)
        print(f"Episode {ep+1}, Avg Reward(last 20): {avg_reward:.2f}")

env.close()
