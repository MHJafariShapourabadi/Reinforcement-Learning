# Reinforcement Learning Algorithms
A Python package including the implementation of various Reinforcement Learning algorithms.
Currently, there are two main packages, "rl-algorithms" and "environments".
In the "environment" package, we have provided some utility functions for plotting the results of each run of the algorithm on the FrozenLake environment from the Gymnasium Library.
In the "rl-algorithms" package, we provide different Reinforcement Learning algorithms. Currently, there are implementations for several Tabular RL algorithms in the "tabular" subpackage.
You can find different exploration algorithms under the "rl-algorithms.tabular.exploration" subpackage, including:
1. Greedy
2. Epsilon Greedy
3. Softmax
Also, you can find various RL algorithms under the "rl-algorithms.tabular.agents" subpackage, including:
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

You can run and evaluate different algorithms on the FrozenLake environment using the "run.py" module on the root directory.
