# Reinforcement Learning Algorithms
A Python package including the implementation of various Reinforcement Learning algorithms.
Currently, there are two main packages, "tabular" and "function_approximation".
In the "tabular.environments" and "function_approximation.environments" subpackages, we have provided some utility functions for plotting the results of each run of the algorithm on the FrozenLake environment from the Gymnasium Library.
In the "tabular.rl-algorithms" and "function_approximation.rl_algorithms" subpackages, we provide different Reinforcement Learning algorithms. Currently, there are implementations for several tabular and function approximation RL algorithms available.
You can find different exploration algorithms under the "tabular.rl-algorithms.exploration" and "function_approximation.rl_algorithms.exploration" subpackage, including:
1. Greedy
2. Epsilon Greedy
3. Softmax

Also, you can find various RL algorithms under the "tabular.rl-algorithms.agents" and "function_approximation.rl_algorithms.agents" subpackage, including:
- Tabular methods:
  1. Policy Iteration
  2. Value Iteration
  3. Real-time Dynamic Programming (RTDP)
  4. On-policy Monte Carlo
  5. Sarsa
  6. Expected Sarsa
  7. Q-learning
  8. Double Q-learning
  9. N-step Sarsa
  10. N-step Expected Sarsa
  11. Tree Backup
  12. Dyna-Q
  13. Dyna-Q+
  14. Prioritized Sweeping
- Function approximation methods:
  - Value-based:
    1. Dueling Double DQN
    2. PER Dueling Double DQN
    3. PER Dueling Double Deep Sarsa
    4. PER Dueling Double Deep N-Step Sarsa
    5. PER Dueling Double Deep N-Step Tree Backup
  - Policy-based:
    1. REINFORCE
    2. REINFORCE with baseline / Vanilla Policy Gradient (VPG)
- Advanced function approximation methods:
  - Advanced policy gradient methods:
    1. Actor-Critic
    2. N-step Actor-Critic
    3. Actor-Critic with Generalized Advantage Estimation (GAE) (n-step lambda-return)
    4. Asynchronous Advantage Actor-Critic (A3C)
    5. N-step Asynchronous Advantage Actor-Critic (A3C)
    6. Asynchronous Advantage Actor-Critic (A3C) with Generalized Advantage Estimation (GAE) (n-step lambda-return)
    7. Advantage Actor-Critic (A2C)
    8. N-step Advantage Actor-Critic (A2C)
    9. Advantage Actor-Critic (A2C) with Generalized Advantage Estimation (GAE) (n-step lambda-return)
    10. Proximal Policy Optimization (PPO)
    11. N-step Proximal Policy Optimization (PPO)
    12. Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE) (n-step lambda-return)
  
You can run and evaluate different algorithms on the FrozenLake environment using the "run_tabular.py" and "run_function_approximation.py" modules on the root directory.
