import numpy as np
import gym
import random
import torch
from gym import spaces

class RIS_MISO_Env(gym.Env):
    def __init__(self, ris=True, random_ris=False):
        super(RIS_MISO_Env, self).__init__()
        
        self.ris = ris  # Enable RIS
        self.random_ris = random_ris  # Use random RIS phase shifts
        self.state_dim = 10  # Example: SNR, user position, RIS parameters
        self.action_dim = 5  # Example: RIS phase shift adjustments
        
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
    
    def reset(self):
        self.state = np.random.uniform(-1, 1, self.state_dim)
        return self.state
    
    def step(self, action):
        if self.random_ris:
            action = np.random.uniform(-1, 1, self.action_dim)
        
        reward = self._calculate_reward(action)
        next_state = self.state + action * 0.1  # Simulated transition
        done = np.random.rand() < 0.1  # Stop with small probability
        
        info = {
            "spectral_efficiency": np.random.uniform(2, 10),
            "energy_efficiency": np.random.uniform(0.5, 2.0),
            "sinr": np.random.uniform(10, 30)
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, action):
        return np.sum(action) / self.action_dim  # Example reward function

# Meta-Learning Wrapper
def meta_wrapper(env, task):
    if task == "high_SNR":
        env.state[0] = np.random.uniform(20, 30)  # Adjust SNR level
    elif task == "low_SNR":
        env.state[0] = np.random.uniform(5, 15)
    elif task == "user_movement":
        env.state[1] = np.random.uniform(0, 100)  # Adjust user position
    return env


# Test with Meta-Learning adaptation
env = meta_wrapper(env, "high_SNR")  
state = env.reset()
print("State after High SNR Adaptation:", state)


# Test with Meta-Learning adaptation
env = meta_wrapper(env, "high_SNR")  
state = env.reset()
print("State after High SNR Adaptation:", state)
