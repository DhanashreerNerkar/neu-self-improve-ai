A PyTorch implementation combining A-PO (Policy Optimization via Optimal Advantage Regression)* and PAG (Multi-Turn Reinforced LLM Self-Correction) for mathematical reasoning tasks.

Overview
This project implements two cutting-edge reinforcement learning techniques for improving LLM reasoning:
A-PO*: Efficiently trains models by pre-computing optimal value functions offline, requiring only single generations per prompt during online training
PAG: Enables self-correction through role-switching between policy (answer generation) and verifier (answer checking)

Based on research papers:
"Accelerating RL for LLM Reasoning with Optimal Advantage Regression" (2025)
"PAG: Multi-Turn Reinforced LLM Self-Correction with Policy as Generative Verifier" (2025)

Key Features
Two-Stage Training:
Stage 1: Offline V* estimation using reference policy
Stage 2: Multi-turn RL with selective revision mechanism

Comprehensive Metrics Tracking:
Training loss
Policy rewards (Turn 1 vs Turn 3)
Verifier accuracy
Answer change ratio (model collapse detection)
Revision rate
V* distribution


Visualization Dashboard: 6 comprehensive plots showing training dynamics

Installation
bash# Install dependencies
pip install transformers==4.41.0
pip install datasets accelerate
pip install torch
pip install matplotlib numpy

Usage
Quick Start
python# 1. Load the notebook
jupyter notebook AstarPO_PAG_implementation.ipynb

# 2. Run cells sequentially
# The notebook automatically:
#    - Downloads/creates training data
#    - Loads the model
#    - Trains using A*-PO + PAG
#    - Generates visualizations
Training Own Model
# Modify in the notebook:
model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # Better for math

# Adjust hyperparameters:
trainer = AStarPO_PAG_Trainer_WithMetrics(
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    train_problems=train_problems[:50],  # Increase dataset size
    beta1=0.5,      # V* estimation temperature
    beta2=0.001,    # KL regularization
    n_samples_v_star=8  # More samples = better V*
)

# Train
trained_model = trainer.train(n_epochs=10, batch_size=4)
```

## Project Structure
```
├── Cell 1: Dependencies & GPU Check
├── Cell 2: MATH Dataset Loading
│   ├── Downloads from HuggingFace
│   └── Falls back to synthetic problems
├── Cell 3: Model Loading
│   ├── GPT-2 (demo)
│   └── Can switch to Qwen2.5-1.5B
├── Cell 4: A*-PO Core Implementation
│   ├── V* estimation
│   └── Advantage calculation
├── Cell 5: PAG Multi-Turn Training
│   ├── Turn 1: Initial attempt
│   ├── Turn 2: Self-verification
│   └── Turn 3: Selective revision
└── Cell 6: Enhanced Training + Visualization
    ├── Metric tracking
    └── 6 visualization plots
```

## Metrics Tracked

1. **Training Loss**: Overall optimization progress
2. **Acc@t1**: Direct generation accuracy
3. **Acc@final**: Self-correction accuracy
4. **Verifier Accuracy**: How well the model detects errors
5. **Answer Change Ratio**: Model collapse indicator
6. **Revision Rate**: Frequency of selective revision
7. **V* Distribution**: Quality of optimal value estimates

## Current Limitations

### Using GPT-2 (Demo Model)
```
Expected Results:
- V* ≈ 0.05 (GPT-2 can't do math)
- Acc@t1 = 0.0 (No correct answers)
- Verifier Acc = High (Can detect errors)
- Answer Change Ratio = Low (Potential collapse)
```

### Recommended: Qwen2.5-1.5B-Instruct
```
Expected Results:
- V* ≈ 0.45 (45% base accuracy)
- Acc@t1 = 0.35 → 0.45 (Improving)
- Acc@final = 0.50 → 0.65 (Self-correction works!)
- Answer Change Ratio = Healthy (>0.5)

Visualization Plots
The training generates 6 comprehensive plots:

Training Loss - Optimization convergence
Policy Rewards - Shows if self-correction helps
Verifier Accuracy - Error detection capability
Answer Change Ratio - Model collapse detection (key PAG metric)
Revision Rate - Selective revision frequency
V Distribution* - Optimal value quality

🔧 Troubleshooting
Memory Issues
python# Use 8-bit quantization for Qwen
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    load_in_8bit=True,
    device_map="auto"
)

Low Accuracy
GPT-2: Expected (not trained for math)
Qwen: Increase n_epochs and n_samples_v_star

Model Collapse
Check Answer Change Ratio in plots
Should be >0.5 for healthy training
If low: Increase revision penalty in loss

Key Concepts
A*-PO (Two-Stage Training)

Stage 1 (Offline): Estimate V* for all problems

python   V*(x) = β₁ * ln(E[exp(r(x,y)/β₁)])

Stage 2 (Online): Train using optimal advantages

python   Loss = (A* - improvement_bonus)²
   where A* = r - V*
```

PAG (Multi-Turn Framework)
```
Turn 1: Generate initial attempt
Turn 2: Self-verify correctness
Turn 3: Revise only if verified as wrong (selective revision)
```

## Sample Results (GPT-2)
```
V* Estimation:
- Problem: "What is 19 - 8?"
- GPT-2 generates nonsense → V* = 0.0

Training:
Epoch 1: Loss=0.288, Acc@t1=0.0, Verifier=100%
Epoch 5: Loss=0.273, Acc@t1=0.0, Verifier=100%

Warning: Low answer change ratio (model collapse)
Future Improvements

Use proper math model: Qwen2.5-1.5B-Instruct
Increase dataset: Use full MATH dataset
Add evaluation: Separate test set metrics
Optimize hyperparameters: Grid search β₁, β₂
Implement GRPO comparison: Benchmark against baseline

References
A*-PO Paper: arXiv:2505.20686v1
PAG Paper: arXiv:2506.10406v1
Original implementations: Links in papers

Contributing
This is a research implementation. Improvements welcome:

Better answer extraction
More robust training
Additional baselines

License
Educational/Research purposes. See original papers for usage rights.

Built with: PyTorch, Transformers, Matplotlib
Status: Demo/Research Implementation
GPU Required: Yes (T4 minimum, better with A100)
