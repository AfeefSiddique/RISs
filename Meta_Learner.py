import torch
import torch.optim as optim
import random

class MetaLearner:
    def __init__(self, agent, meta_lr=0.001, num_tasks=5):
        self.agent = agent
        self.meta_lr = meta_lr
        self.num_tasks = num_tasks
        self.optimizer = optim.Adam(self.agent.actor.parameters(), lr=meta_lr)

    def sample_tasks(self):
        """Generate different task variations (e.g., different user locations, SNR levels)"""
        tasks = []
        for _ in range(self.num_tasks):
            task = {"snr": random.uniform(10, 30), "user_position": random.uniform(0, 100)}
            tasks.append(task)
        return tasks

    def update(self):
        """Meta-update using collected experience"""
        self.optimizer.zero_grad()

        for param in self.agent.actor.parameters():
            param.grad = torch.zeros_like(param)

        # Sum gradients over tasks
        for _ in range(self.num_tasks):
            task_loss = self.compute_task_loss()
            task_loss.backward()

        self.optimizer.step()

    def compute_task_loss(self):
        """Compute task-specific loss (e.g., maximize received signal strength)"""
        dummy_state = torch.FloatTensor([0.5] * self.agent.actor.fc1.in_features)
        dummy_action = self.agent.actor(dummy_state)
        dummy_q_value = self.agent.critic(dummy_state.unsqueeze(0), dummy_action.unsqueeze(0))
        return -dummy_q_value.mean()
