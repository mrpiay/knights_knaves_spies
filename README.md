# Knights, Knaves & Spies: From Logic Solver to Machine Learning

> *"On the island of knights and knaves, knights always tell the truth, knaves always lie — but a spy can do either."*

---

## Table of Contents

- [The Puzzle](#the-puzzle)
- [Project Overview](#project-overview)
- [Files](#files)
- [How It Works](#how-it-works)
  - [1. Universal Solver](#1-universal-solver)
  - [2. Puzzle Dataset](#2-puzzle-dataset)
  - [3. Correctness Checker](#3-correctness-checker)
  - [4. MLP Classifier](#4-mlp-classifier)
- [Running the Code](#running-the-code)
- [ML Concepts Covered](#ml-concepts-covered)
- [Results](#results)
- [MLP Tutorial](#mlp-tutorial)
- [References & Credits](#references--credits)

---

## The Puzzle

Knights and Knaves is a classic logic puzzle genre introduced by philosopher **Raymond Smullyan**. The standard version has two character types:

- **Knight** — always tells the truth
- **Knave** — always lies

This project uses an extended three-character variant that adds:

- **Spy** — can say anything, truth or lie

You are given **three people (A, B, C)** — one Knight, one Knave, one Spy — and each makes a statement. Your task is to deduce **who is who** from pure logical deduction.

**Example puzzle:**

```
A: "I am a Knight"
B: "I am not a Spy"
C: "I am a Knave"
```

> Answer: A=Knave, B=Knight, C=Spy

---

## Project Overview

This project approaches the puzzle from multiple angles, treating it as a vehicle for exploring **logic, constraint satisfaction, and machine learning**:

```
Knights, Knaves & Spies Puzzles
        │
        ├── Logic / Constraint Satisfaction
        │       └── Universal Solver  (brute-force over 6 role permutations)
        │
        ├── Dataset Engineering
        │       └── 274 puzzles → structured JSON with verified solutions
        │
        └── Classical ML
                └── MLP Classifier  (one-hot encoding → 6-class prediction)
```

The core ML question is: **can a model learn to solve these puzzles without being told the rules?**

---

## Files

| File | Description |
|------|-------------|
| `data/knights_knaves_puzzles.json` | 274 unique puzzles with verified solutions |
| `solver/knights_knaves_solver.py` | Universal logic solver (constraint satisfaction) |
| `solver/checker.py` | Validates solver against all 274 puzzles |
| `ml/mlp_classifier.py` | MLP neural net that learns to solve puzzles from examples |

---

## How It Works

### 1. Universal Solver

**File:** `solver/knights_knaves_solver.py`

The solver works by **brute-force constraint satisfaction** over all 6 possible role permutations of (Knight, Knave, Spy) assigned to (A, B, C). For each candidate assignment it checks whether every speaker's statement is consistent with their role:

- **Knight** → statement must be **true**
- **Knave** → statement must be **false**
- **Spy** → always consistent (no constraint)

The solver supports all statement forms found in the dataset:

```
"I am a Knight / Knave / Spy"
"I am not a Knight / Knave / Spy"
"X is a Knight / Knave / Spy"
"X is not a Knight / Knave / Spy"
"If you asked me, I would say that X is the spy"
```

The trickiest case is the **conditional statement**. A Knave saying *"If you asked me, I would say X is the spy"* is consistent only if X **is** the spy — the double-negation of the Knave's lying nature cancels out, producing the same constraint as for a Knight.

**Validated against all 274 puzzles: 274/274 correct.**

---

### 2. Puzzle Dataset

**File:** `data/knights_knaves_puzzles.json`

The dataset contains 274 unique Knight/Knave/Spy puzzles with exactly one valid solution each. It is structured as a JSON array:

```json
[
  {
    "statements": {
      "A": "I am a Knight",
      "B": "I am not a Spy",
      "C": "I am a Knave"
    },
    "solution": {
      "A": "Knave",
      "B": "Knight",
      "C": "Spy"
    }
  },
  ...
]
```

**Dataset statistics:**

| Property | Value |
|----------|-------|
| Total puzzles | 274 |
| Unique statements | 22 |
| Feature dimensions | 66 (3 × 22 one-hot) |
| Output classes | 6 (role permutations) |
| Class balance | Roughly uniform (35–62 per class) |

---

### 3. Correctness Checker

**File:** `solver/checker.py`

Runs the solver against every puzzle in the JSON dataset and prints a detailed puzzle-by-puzzle report:

```
Puzzle #  1  ✅ PASS
  A: I am a Knight
  B: I am not a Spy
  C: I am a Knave
  Expected : A=Knave   B=Knight  C=Spy
```

The terminal stays open at the end so you can scroll through results at your own pace.

---

### 4. MLP Classifier

**File:** `ml/mlp_classifier.py`

A standard **Multi-Layer Perceptron** trained to predict the correct role assignment directly from the puzzle's statements.

**Feature encoding:**

Each puzzle is encoded as a **66-dimensional binary vector** — 3 speakers, each one-hot encoded over 22 possible statements:

```
[  speaker A (22 bits)  |  speaker B (22 bits)  |  speaker C (22 bits)  ]
```

**Architecture:**

```
Input (66) → Dense (64) → ReLU → Dense (32) → ReLU → Output (6) → Softmax
```

**Training:** scikit-learn `MLPClassifier`, cross-entropy loss, Adam optimizer, 80/20 train-test split.

This model learns purely from **correlations in the data** — it does not know the rules. It discovers that certain statement patterns reliably predict certain outcomes.

> For a full line-by-line explanation of the MLP code, including theory and exercises, see **[docs/MLP_TUTORIAL.md](docs/MLP_TUTORIAL.md)**.

---

## Running the Code

```bash
git clone https://github.com/mrpiay/knights_knaves_spies.git
cd knights_knaves_spies
pip install -r requirements.txt
```

**Run the solver checker** (puzzle-by-puzzle output):
```bash
python solver/checker.py
```

**Run the MLP classifier:**
```bash
python ml/mlp_classifier.py
```

Run both commands from the repo root. Each script resolves its own paths automatically.

---

## ML Concepts Covered

| Concept | Where |
|---------|-------|
| Constraint satisfaction | `knights_knaves_solver.py` |
| One-hot feature encoding | `mlp_classifier.py` |
| Multi-class classification | `mlp_classifier.py` |
| Cross-entropy loss | `mlp_classifier.py` |
| Train/test split & stratification | `mlp_classifier.py` |
| Classification report & confusion analysis | `mlp_classifier.py` |

---

## Results

### MLP Classifier

| Metric | Value |
|--------|-------|
| Test accuracy | **96.4%** |
| Misclassified | 2 / 55 |
| Both errors | Spy ↔ Knight confusion |

The 2 errors both involve swapping Spy and Knight — the two roles most likely to make truthful-sounding statements.

---

## MLP Tutorial

For a detailed, concept-by-concept walkthrough of the MLP classifier — covering one-hot encoding, forward pass, backpropagation, hyperparameters, and exercises — see the dedicated tutorial:

**[docs/MLP_TUTORIAL.md](docs/MLP_TUTORIAL.md)**

---

## References & Credits

**Puzzle logic & theory:**
- Wikipedia — [Knights and Knaves](https://en.wikipedia.org/wiki/Knights_and_Knaves)
- Smullyan, R. M. (1978). *What Is the Name of This Book?* Prentice-Hall.

**Puzzle dataset (274 unique Knight/Knave/Spy puzzles):**
- Credit: **Mark Newheiser** — [http://newheiser.googlepages.com/knightsandknaves](http://newheiser.googlepages.com/knightsandknaves)
- Please do not link the original puzzle file directly; credit the author if you use or build on this dataset.

**Libraries used:**
- [NumPy](https://numpy.org/) — numerical computing
- [scikit-learn](https://scikit-learn.org/) — MLP classifier, evaluation utilities
