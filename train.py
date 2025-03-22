import torch
import numpy as np
from env import RIS_MISO_Env, meta_wrapper
from hybrid_ddpg_ppo import Hybrid_DDPG_PPO
from replay_buffer import ReplayBuffer

# Hyperparameters
STATE_DIM = 10  # Adjust according to your environment
ACTION_DIM = 5  # Adjust according to RIS phase shifts
EPISODES = 500
BATCH_SIZE = 64
BUFFER_SIZE = 100000

# Initialize Environment
env = RIS_MISO_Env(ris=True)
env = meta_wrapper(env, "high_SNR")  # Example adaptation

# Initialize Agent
agent = Hybrid_DDPG_PPO(STATE_DIM, ACTION_DIM)
replay_buffer = ReplayBuffer(BUFFER_SIZE)

# Training Loop
for episode in range(EPISODES):
    state = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)
        
        state = next_state
        episode_reward += reward
        
        if len(replay_buffer) > BATCH_SIZE:
            agent.update(replay_buffer, BATCH_SIZE)
    
    print(f"Episode {episode + 1}: Reward = {episode_reward}")
    
    # Save model every 50 episodes
    if (episode + 1) % 50 == 0:
        agent.save("models/hybrid_drl")

print("Training Completed!")
