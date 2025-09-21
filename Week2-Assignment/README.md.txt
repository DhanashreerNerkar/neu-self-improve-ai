**Multi-Armed Bandit Framework**
A Python implementation of multi-armed bandit algorithms reproducing 
Figure 2.2 from Sutton & Barto's "Reinforcement Learning: An Introduction" 
with additional gradient policy learning methods.

Overview:
----------
This framework implements and compares different action selection strategies for the multi-armed bandit problem:
1. Epsilon-Greedy Methods: Balance exploration vs exploitation using Epsilon probability
2. Gradient Bandit Methods: Use softmax action selection with gradient-based learning (REINFORCE)
The implementation reproduces the classic 10-armed testbed experiment and extends it with gradient policy learning algorithms.

What's Inside:
---------------
1. MDP Framework: Built from scratch without external RL libraries
2. Multiple Algorithms: Epsilon-greedy (Epsilon = 0, 0.01, 0.1) and Gradient bandits with/without baseline (alpha = 0.1, 0.4)
3. Evaluation: Average reward tracking | Optimal action percentage | Statistical significance through 2000 independent runs
4. Visualization: Reproduces Figure 2.2 style plots from the book

Requirements:
--------------
requirements.txt
numpy>=1.19.0
matplotlib>=3.3.0
tqdm>=4.50.0
pandas>=1.1.0

Installation:
--------------
1. Clone the repository:
git clone https://github.com/yourusername/bandit-framework.git
cd bandit-framework

2. Install dependencies:
pip install -r requirements.txt

3. Usage:
Run the complete experiment in Jupyter Notebook or Python script

Architecture:
-------------
Components
1. MultiArmedBandit:
Simulates k-armed bandit with Gaussian rewards
True action values: q*(a) ~ N(0, 1)
Rewards: R ~ N(q*(a), 1)

2. Epsilon-Greedy Agent:
Epsilon-greedy action selection
Sample-average or constant step-size updates
Optimistic initial values support

3. GradientBandit Agent:
Softmax action selection: pie(a) = exp(H(a)) / sum exp(H(b))
Gradient ascent on preferences
Optional baseline for variance reduction
Update rule: H(a) <- H(a) + alpha(R - R')(1 - pie(a))

4. run_experiment
Manages multiple independent runs
Aggregates statistics across bandits
Handles reproducibility with seeding

Experiements:
-------------
The main experiment compares 7 different methods:
Method      Parameters              Description
ε-greedy    ε=0.1                   10% exploration
ε-greedy    ε=0.01                  1% exploration
ε-greedy    ε=0                     Pure exploitation (greedy)
Gradient    α=0.1, no baseline      Slow learning without baseline
Gradient    α=0.4, no baseline      Fast learning without baseline
Gradient    α=0.1, with baseline    Slow learning with variance reduction
Gradient    α=0.4, with baseline    Fast learning with variance reduction

Expected Results:
------------------
Performance at step 1000:
Best overall: Gradient alpha=0.4 with baseline (~85% optimal)
Best Epsilon-greedy: Epsilon=0.1 (~80% optimal)
Worst: Epsilon=0 greedy (~40% optimal)

Key Insights:
--------------
Baseline significantly improves gradient methods
Higher learning rates (alpha=0.4) work well with baseline
Pure greedy (Epsilon=0) gets stuck on suboptimal actions
Small exploration (Epsilon=0.01) improves slowly

Visualization
--------------
The framework generates two plots:
Average Reward: Shows learning progress over time
% Optimal Action: Measures how often the best arm is selected

Theory/Background:
------------------
Multi-Armed Bandit Problem:
Goal: Maximize expected reward over time
Challenge: Exploration vs exploitation trade-off
Applications: A/B testing, clinical trials, recommendation systems

Algorithm               Pros                            Cons
------------------------------------------------------------------------------------
ε-greedy                Simple, proven effective        Fixed exploration rate
Gradient                Natural action probabilities    Requires tuning, can be unstable
Gradient + Baseline     Reduced variance, stable        More complex

References
--------------
Sutton & Barto: Reinforcement Learning: An Introduction, Chapter 2

