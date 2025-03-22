import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ddpg import DDPG  # Import base DDPG agent

class Hybrid_DDPG_PPO:
    def __init__(self, state_dim, action_dim, lr_actor=1e-3, lr_critic=1e-3, gamma=0.99, tau=0.005, epsilon=0.2):
        self.ddpg = DDPG(state_dim, action_dim, lr_actor, lr_critic, gamma, tau)
        self.epsilon = epsilon  # PPO Clipping Factor
        self.gamma = gamma
        
        self.ppo_critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.critic_optimizer = optim.Adam(self.ppo_critic.parameters(), lr=lr_critic)
    
    def select_action(self, state):
        return self.ddpg.select_action(state)  # Use DDPG actor
    
    def update(self, replay_buffer, batch_size=64, ppo_ratio=0.5):
        ddpg_loss = self.ddpg.update(replay_buffer, batch_size)  # Train DDPG
        
        # PPO Update
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        values = self.ppo_critic(states)
        next_values = self.ppo_critic(next_states).detach()
        
        advantages = rewards + (1 - dones) * self.gamma * next_values - values
        old_probs = torch.exp(-advantages)  # Estimate old probability ratios
        
        # PPO Clipped Loss
        ratio = old_probs / (old_probs.detach() + 1e-10)
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        ppo_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        self.critic_optimizer.zero_grad()
        ppo_loss.backward()
        self.critic_optimizer.step()
        
        # Hybrid Loss Combination
        hybrid_loss = ppo_ratio * ppo_loss + (1 - ppo_ratio) * ddpg_loss
        return hybrid_loss

    def save(self, filename):
        torch.save(self.ddpg.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.ddpg.critic.state_dict(), filename + "_critic.pth")
    
    def load(self, filename):
        self.ddpg.actor.load_state_dict(torch.load(filename + "_actor.pth"))
        self.ddpg.critic.load_state_dict(torch.load(filename + "_critic.pth"))
