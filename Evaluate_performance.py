import torch
import numpy as np
import matplotlib.pyplot as plt
from env import RIS_MISO_Env
from hybrid_drl import HybridDRL

# Load trained agent
state_dim = 10  # Update based on your environment
action_dim = 5  # Update based on RIS elements
agent = HybridDRL(state_dim, action_dim)
agent.actor.load_state_dict(torch.load("hybrid_drl_model.pth"))

def evaluate_model(env, agent, num_episodes=100):
    total_se, total_ee, total_sinr = [], [], []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_se, episode_ee, episode_sinr = [], [], []

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            episode_se.append(info["spectral_efficiency"])
            episode_ee.append(info["energy_efficiency"])
            episode_sinr.append(info["sinr"])
            
            state = next_state

        total_se.append(np.mean(episode_se))
        total_ee.append(np.mean(episode_ee))
        total_sinr.append(np.mean(episode_sinr))
    
    return np.mean(total_se), np.mean(total_ee), np.mean(total_sinr)

# Baseline Evaluations
env_baselines = {
    "Without RIS": RIS_MISO_Env(ris=False),
    "Random RIS": RIS_MISO_Env(random_ris=True),
    "DDPG Only": RIS_MISO_Env(),
    "Hybrid DRL": RIS_MISO_Env(),
    "Hybrid DRL + Meta-Learning": RIS_MISO_Env()
}

results = {}
for method, env in env_baselines.items():
    print(f"Evaluating: {method}")
    results[method] = evaluate_model(env, agent)
    
# Extract results for plotting
methods = list(results.keys())
y_se = [results[m][0] for m in methods]
y_ee = [results[m][1] for m in methods]

def plot_performance(methods, y_se, y_ee):
    plt.figure(figsize=(10,5))
    plt.plot(methods, y_se, marker='o', label="Spectral Efficiency (bps/Hz)")
    plt.plot(methods, y_ee, marker='s', label="Energy Efficiency (bits/Joule)")
    plt.xlabel("Method")
    plt.ylabel("Performance")
    plt.legend()
    plt.title("Performance Comparison")
    plt.xticks(rotation=15)
    plt.show()

plot_performance(methods, y_se, y_ee)

# Save Results
np.save("evaluation_results.npy", results)
print("Evaluation Completed and Results Saved!")
