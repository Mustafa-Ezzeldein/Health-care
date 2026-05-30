import numpy as np
import random
import os

class PatientRLEnvironment:
    """
    Reinforcement Learning Environment (RL Env):
    Simulates the interaction of the doctor/system's decision with the patient's condition.
    States: 0 (Low Risk) - 1 (Medium Risk) - 2 (High Risk)
    Actions: 0 (Routine Check) - 1 (Alert Doctor) - 2 (Emergency Call)
    """
    def __init__(self):
        self.n_states = 3
        self.n_actions = 3
        
    def step(self, state, action):
        reward = 0
        
        # Reward Engineering
        if state == 0: # Healthy Patient (Low Risk)
            if action == 0: reward = 10     # Correct: Save hospital resources
            elif action == 1: reward = -5   # Error: Unnecessarily disturb the doctor
            elif action == 2: reward = -20  # Disaster: False emergency alarm
                
        elif state == 1: # Patient at Medium Risk
            if action == 0: reward = -10    # Medical negligence
            elif action == 1: reward = 15   # Optimal action: Call doctor for checkup
            elif action == 2: reward = -5   # Slightly overreacted response
                
        elif state == 2: # Patient in Critical Condition
            if action == 0: reward = -50    # Medical crime (Neglecting critical case)
            elif action == 1: reward = 5    # Slow response
            elif action == 2: reward = 20   # Immediate ICU (Optimal action)
                
        # Simulate state evolution after decision
        next_state = random.choice([0, 1, 2])
        return next_state, reward

def train_qlearning_agent():
    print("🚀 Initializing Reinforcement Learning (Q-Learning) for Optimal Medical Decisions...")
    
    env = PatientRLEnvironment()
    # Create Q-Table (Rows: States, Columns: Actions)
    q_table = np.zeros((env.n_states, env.n_actions))
    
    # Reinforcement Learning Hyperparameters
    alpha = 0.1       # Learning Rate
    gamma = 0.9       # Discount Factor
    epsilon = 0.2     # Exploration Rate
    epochs = 10000    # Number of scenarios to simulate system interaction with different patients
    
    for i in range(epochs):
        state = random.choice([0, 1, 2])
        
        # Action Selection (Epsilon-Greedy Strategy)
        if random.uniform(0, 1) < epsilon:
            action = random.choice([0, 1, 2]) # Try a random decision for learning
        else:
            action = np.argmax(q_table[state]) # Exploit the best known decision
            
        # Execute action and receive reward/penalty
        next_state, reward = env.step(state, action)
        
        # Update mathematical decision value (Q-Learning Formula)
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

    print("\n" + "="*45)
    print("        🤖 AI Recommended Treatment Policy 🤖")
    print("        (Learned via Penalties & Rewards)   ")
    print("="*45)
    
    states_desc = ["🟢 Low Risk   ", "🟡 Medium Risk", "🔴 High Risk  "]
    actions_desc = ["Routine Check ✅", "Alert Doctor ⚠️", "Emergency Call 🚑🚨"]
    
    for s in range(env.n_states):
        best_action = np.argmax(q_table[s])
        print(f"If Patient State is [{states_desc[s]}] => Optimal AI Action: {actions_desc[best_action]}")
    print("="*45)
    
    os.makedirs('models', exist_ok=True)
    np.save('models/q_table.npy', q_table)
    print("\n✅ RL Q-Table Policy saved successfully to 'models/q_table.npy'.")

if __name__ == '__main__':
    train_qlearning_agent()
