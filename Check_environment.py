from environment import RIS_MISO_Env, meta_wrapper

# Test standard environment
env = RIS_MISO_Env(ris=True)  
state = env.reset()
print("Initial State:", state)

# Test with Meta-Learning adaptation
env = meta_wrapper(env, "high_SNR")  
state = env.reset()
print("State after High SNR Adaptation:", state)

# Step through the environment
action = env.action_space.sample()
next_state, reward, done, info = env.step(action)

print("Next State:", next_state)
print("Reward:", reward)
print("Info:", info)
