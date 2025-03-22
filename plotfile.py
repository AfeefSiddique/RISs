import numpy as np
import matplotlib.pyplot as plt
import os

# Define the path to the .npy file
file_path = r'D:\RIS-MISO-Deep-Reinforcement-Learning-main\Learning Curves\sum_rate_ris\result.npy'

# Load the .npy file
try:
    print("Loading data from:", file_path)
    data = np.load(file_path)
except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")

# Plot the data if loaded successfully
if 'data' in locals():  # Check if data was loaded
    plt.figure(figsize=(10, 6))
    plt.plot(data, marker='o', linestyle='-', color='b', label='Data')
    plt.title('Data from result.npy')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.show()
