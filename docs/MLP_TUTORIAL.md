# MLP Tutorial — Application for Solving the Knights, Knaves & Spies Logic Puzzles

> A comprehensive guide to Multi-Layer Perceptrons through the lens of a logic puzzle classification problem.

---

## Table of Contents

1. [What is an MLP?](#1-what-is-an-mlp)
2. [The Challenge](#2-the-challenge)
3. [Why MLP fits this problem](#3-why-mlp-fits-this-problem)
4. [Core Concepts Before the Code](#4-core-concepts-before-the-code)
   - 4.1 [Features and Labels](#41-features-and-labels)
   - 4.2 [One-Hot Encoding](#42-one-hot-encoding)
   - 4.3 [Forward Pass](#43-forward-pass)
   - 4.4 [Loss Function](#44-loss-function)
   - 4.5 [Backpropagation](#45-backpropagation)
   - 4.6 [Training Loop](#46-training-loop)
   - 4.7 [Train / Test Split](#47-train--test-split)
5. [The Code — Line by Line](#5-the-code--line-by-line)
   - 5.1 [Imports](#51-imports)
   - 5.2 [Loading the Data](#52-loading-the-data)
   - 5.3 [Building the Vocabulary](#53-building-the-vocabulary)
   - 5.4 [Defining the Labels](#54-defining-the-labels)
   - 5.5 [Feature Encoding Function](#55-feature-encoding-function)
   - 5.6 [Building X and y](#56-building-x-and-y)
   - 5.7 [Train / Test Split](#57-train--test-split)
   - 5.8 [Defining the MLP](#58-defining-the-mlp)
   - 5.9 [Training](#59-training)
   - 5.10 [Evaluation](#510-evaluation)
   - 5.11 [Inspecting Mistakes](#511-inspecting-mistakes)
6. [Reading the Output](#6-reading-the-output)
7. [Hyperparameter Guide](#7-hyperparameter-guide)
8. [Exercises](#8-exercises)
9. [Glossary](#9-glossary)

---

## 1. What is an MLP?

A **Multi-Layer Perceptron (MLP)** is the foundational building block of deep learning. It is a function that maps an input vector to an output vector by passing numbers through a series of **layers**, each one transforming the data.

```
Input layer     Hidden layer 1    Hidden layer 2    Output layer
  (66 nodes)      (64 nodes)         (32 nodes)       (6 nodes)

  x₁ ──┐
  x₂ ──┼──► [W₁, b₁] ──► ReLU ──► [W₂, b₂] ──► ReLU ──► [W₃, b₃] ──► Softmax ──► ŷ
  ...  ─┘
  x₆₆─┘
```

Each arrow represents a **weighted connection**. Training means finding the weights that make the network's output match the correct labels.

There are three kinds of layers:

| Layer type | Role |
|------------|------|
| **Input** | Receives the raw feature vector — no computation |
| **Hidden** | Learns internal representations — this is where the "thinking" happens |
| **Output** | Produces the final prediction — one value per class |

Between every pair of layers sits an **activation function** that introduces non-linearity, allowing the network to learn patterns that a simple straight line cannot capture.

---

## 2. The Challenge

We have **274 logic puzzles**. In each one:

- Three people (A, B, C) each make a statement
- Exactly one is a **Knight** (always truthful), one a **Knave** (always lies), one a **Spy** (can do either)
- The task is to determine who has which role

**Example:**

```
A: "I am a Knight"
B: "I am not a Spy"
C: "I am a Knave"

Answer: A = Knave, B = Knight, C = Spy
```

From a machine learning perspective:
- **Input**: 3 statements (one per person)
- **Output**: 1 of 6 possible role assignments

The MLP must learn — purely from the 274 labelled examples — which patterns of statements correspond to which assignments.

---

## 3. Why MLP fits this problem

| Property of our problem | Why MLP handles it |
|-------------------------|-------------------|
| Fixed-size input (always 3 statements) | MLP needs fixed-size input vectors — ✅ |
| Small dataset (274 puzzles) | A shallow MLP generalises well without overfitting on small data |
| Discrete categories (6 classes) | Multi-class classification is what MLP + Softmax was designed for |
| Non-linear decision boundary | Statements interact in complex ways; ReLU hidden layers capture this |
| Structured vocabulary (22 known statements) | One-hot encoding converts text to numbers cleanly |

---

## 4. Core Concepts Before the Code

### 4.1 Features and Labels

Machine learning requires data in two parts:

- **X** — the **features matrix**, shape `[n_samples, n_features]`. Each row is one puzzle, each column is one measurable attribute.
- **y** — the **labels vector**, shape `[n_samples]`. Each entry is the correct class index for that puzzle.

In our case:
- `X` has shape `[274, 66]` — 274 puzzles, 66 features each
- `y` has shape `[274]` — values 0–5, one per possible role assignment

### 4.2 One-Hot Encoding

Computers need numbers, not words. **One-hot encoding** converts a categorical value into a binary vector where exactly one position is 1 and the rest are 0.

With 22 possible statements, the statement *"I am a Knight"* becomes:

```
Position:  0   1   2   3   4  ...  21
Value:    [0,  0,  0,  1,  0, ..., 0]
                         ▲
                  index of "I am a Knight" in sorted vocab
```

We do this **three times** — once per speaker — and concatenate the results:

```
Full feature vector (66 values):
[ A's 22 bits | B's 22 bits | C's 22 bits ]
```

Why one-hot and not just the index number (0, 1, 2…)?  
Because numbers imply order. Statement 5 is not "greater than" statement 2 — they are just different. One-hot treats all statements as equally distant from each other.

### 4.3 Forward Pass

When a puzzle's feature vector `x` enters the network, it flows **forward** through each layer:

```
Layer 1 (hidden, 64 nodes):
  z₁ = x · W₁ + b₁          (linear transformation)
  a₁ = ReLU(z₁)              (activation: max(0, z))

Layer 2 (hidden, 32 nodes):
  z₂ = a₁ · W₂ + b₂
  a₂ = ReLU(z₂)

Output layer (6 nodes):
  z₃ = a₂ · W₃ + b₃
  ŷ  = Softmax(z₃)           (converts scores to probabilities)
```

**ReLU** (Rectified Linear Unit): `f(x) = max(0, x)`. It is zero for negative inputs and linear for positive ones. Simple, fast, effective.

**Softmax**: converts a vector of raw scores into a probability distribution that sums to 1.

```
Softmax([2.1, 0.3, -1.2, 0.8, 1.5, -0.4])
     → [0.45, 0.08, 0.02, 0.13, 0.25, 0.04]   ← sums to 1.0
```

The network predicts the class with the **highest probability**.

### 4.4 Loss Function

The **loss** measures how wrong the prediction is. We use **cross-entropy loss**:

```
Loss = -log( P(correct class) )
```

If the network assigns probability 0.95 to the correct class → loss = -log(0.95) ≈ 0.05 (good)  
If the network assigns probability 0.05 to the correct class → loss = -log(0.05) ≈ 3.0 (bad)

The goal of training is to **minimise this loss**.

### 4.5 Backpropagation

After the forward pass computes the loss, **backpropagation** computes how much each weight contributed to that loss. It uses the **chain rule** from calculus to propagate gradients backwards through the network:

```
∂Loss/∂W₃  →  update W₃
∂Loss/∂W₂  →  update W₂
∂Loss/∂W₁  →  update W₁
```

Each weight is then nudged in the direction that **reduces the loss**:

```
W ← W - learning_rate × ∂Loss/∂W
```

This is **gradient descent**. Repeat thousands of times → the network gradually improves.

### 4.6 Training Loop

One full pass through all training data is called an **epoch**. Training runs for many epochs:

```
for epoch in range(max_iter):
    for each batch of training examples:
        1. forward pass → compute predictions
        2. compute loss
        3. backward pass → compute gradients
        4. update weights
```

scikit-learn handles all of this automatically inside `model.fit()`.

### 4.7 Train / Test Split

We never evaluate the model on data it was trained on — that would be like letting a student mark their own exam. Instead we split the data:

- **Training set (80%)** — used to fit the model weights
- **Test set (20%)** — held out, used only for final evaluation

```
274 puzzles total
    └── 219 training puzzles  (model learns from these)
    └──  55 test puzzles      (model never sees these during training)
```

**Stratification** ensures each class is proportionally represented in both splits, which is important with only 274 examples.

---

## 5. The Code — Line by Line

### 5.1 Imports

```python
import json
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
```

| Import | Purpose |
|--------|---------|
| `json` | Standard library — reads the `.json` puzzle file |
| `numpy as np` | Numerical arrays — all feature vectors are numpy arrays |
| `MLPClassifier` | scikit-learn's MLP implementation — handles forward pass, backprop, weight updates |
| `train_test_split` | Splits data into training and test sets with a single function call |
| `classification_report` | Prints per-class precision, recall and F1 score |

---

### 5.2 Loading the Data

```python
with open('knights_knaves_puzzles.json') as f:
    puzzles = json.load(f)
```

`json.load(f)` reads the file and converts it to a Python list of dictionaries. Each element looks like:

```python
{
    "statements": {"A": "I am a Knight", "B": "I am not a Spy", "C": "I am a Knave"},
    "solution":   {"A": "Knave", "B": "Knight", "C": "Spy"}
}
```

After this line, `puzzles` is a list of 274 such dicts.

---

### 5.3 Building the Vocabulary

```python
PEOPLE = ['A', 'B', 'C']
ROLES  = ['Knight', 'Knave', 'Spy']

vocab    = sorted(set(s for p in puzzles for s in p['statements'].values()))
stmt2idx = {s: i for i, s in enumerate(vocab)}
```

**Line by line:**

```python
PEOPLE = ['A', 'B', 'C']   # the three speakers — used to iterate in a consistent order
ROLES  = ['Knight', 'Knave', 'Spy']   # the three possible roles
```

```python
vocab = sorted(set(s for p in puzzles for s in p['statements'].values()))
```

This is a **generator expression** inside `set()` inside `sorted()`. Unpacked:

```python
# Step 1: collect every statement from every puzzle
all_statements = []
for p in puzzles:                          # for each puzzle
    for s in p['statements'].values():     # for each of A's, B's, C's statements
        all_statements.append(s)

# Step 2: deduplicate with set()
unique_statements = set(all_statements)    # 22 unique statements

# Step 3: sort for a stable, reproducible order
vocab = sorted(unique_statements)          # list of 22 statements, alphabetically ordered
```

`sorted()` is critical here: Python sets have no guaranteed order. If we skip `sorted()`, the vocabulary index could change between runs, making results irreproducible.

```python
stmt2idx = {s: i for i, s in enumerate(vocab)}
```

Creates a **lookup dictionary** mapping each statement string to its integer index:

```python
{
    'A is a Knave':    0,
    'A is a Knight':   1,
    'A is a Spy':      2,
    'A is not a Knave': 3,
    ...
    'If you asked me, I would say that C is the spy': 21
}
```

`enumerate(vocab)` yields `(0, stmt_0), (1, stmt_1), ...` — the dict comprehension reverses it to `{stmt: index}`.

---

### 5.4 Defining the Labels

```python
role_assignments = [
    ('Knight', 'Knave', 'Spy'),
    ('Knight', 'Spy',   'Knave'),
    ('Knave',  'Knight','Spy'),
    ('Knave',  'Spy',   'Knight'),
    ('Spy',    'Knight','Knave'),
    ('Spy',    'Knave', 'Knight'),
]
```

There are exactly **3! = 6** ways to assign three distinct roles to three people. This list enumerates all of them. The **position** in this list becomes the **class label** (0 through 5).

```
Index 0 → A=Knight, B=Knave,  C=Spy
Index 1 → A=Knight, B=Spy,    C=Knave
Index 2 → A=Knave,  B=Knight, C=Spy
...
Index 5 → A=Spy,    B=Knave,  C=Knight
```

```python
label2name = {i: f"A={r[0][0]} B={r[1][0]} C={r[2][0]}"
              for i, r in enumerate(role_assignments)}
```

Creates short human-readable names for the classification report:

```python
{0: "A=K B=K C=S",   # K = Knight, K = Knave, S = Spy (first letter only)
 1: "A=K B=S C=K",
 ...}
```

`r[0][0]` means: take role tuple `r`, get the first role string `r[0]` (e.g. `"Knight"`), take its first character `[0]` → `"K"`.

```python
assign2label = {r: i for i, r in enumerate(role_assignments)}
```

The **reverse** mapping — given a tuple like `('Knave', 'Knight', 'Spy')`, look up its integer label. Used during encoding.

---

### 5.5 Feature Encoding Function

```python
def encode(puzzle):
    x = np.zeros(3 * len(vocab), dtype=np.float32)
    for j, person in enumerate(PEOPLE):
        x[j * len(vocab) + stmt2idx[puzzle['statements'][person]]] = 1.0
    y = assign2label[tuple(puzzle['solution'][p] for p in PEOPLE)]
    return x, y
```

This is the heart of the data pipeline. Let's walk through it step by step.

```python
x = np.zeros(3 * len(vocab), dtype=np.float32)
```

Creates a vector of **66 zeros**: 3 speakers × 22 statements. This is the blank template we'll fill in.

```python
for j, person in enumerate(PEOPLE):
```

Iterates: `j=0, person='A'` → `j=1, person='B'` → `j=2, person='C'`

```python
    x[j * len(vocab) + stmt2idx[puzzle['statements'][person]]] = 1.0
```

Let's break the index calculation apart:

```
j * len(vocab)                    → offset for this speaker's block
                                     j=0 (A): offset = 0
                                     j=1 (B): offset = 22
                                     j=2 (C): offset = 44

stmt2idx[puzzle['statements'][person]]  → index of this person's statement within vocab
                                           e.g. "I am a Knight" → 16

Final index for A saying "I am a Knight":  0 + 16 = 16   → x[16]  = 1.0
Final index for B saying "I am not a Spy": 22 + 18 = 40  → x[40]  = 1.0
Final index for C saying "I am a Knave":   44 + 15 = 59  → x[59]  = 1.0
```

The resulting vector has **exactly 3 ones** and 63 zeros — one per speaker.

```python
    y = assign2label[tuple(puzzle['solution'][p] for p in PEOPLE)]
```

Constructs the label integer:

```python
# puzzle['solution'] = {"A": "Knave", "B": "Knight", "C": "Spy"}
tuple(puzzle['solution'][p] for p in PEOPLE)   # → ('Knave', 'Knight', 'Spy')
assign2label[('Knave', 'Knight', 'Spy')]        # → 2
```

So for this puzzle `y = 2`.

---

### 5.6 Building X and y

```python
X, y = zip(*[encode(p) for p in puzzles])
X, y = np.array(X), np.array(y)
```

```python
[encode(p) for p in puzzles]
```

Applies `encode()` to all 274 puzzles, producing a list of `(x_vector, label)` tuples:

```python
[(array([0,0,...,1,...,0]), 2),
 (array([0,0,...,0,...,1]), 0),
 ...]
```

```python
zip(*[...])
```

The `*` **unpacks** the list, and `zip` **transposes** it — turning a list of `(x, y)` pairs into two separate sequences: all x vectors and all y labels.

```python
X, y = np.array(X), np.array(y)
```

Converts the sequences to proper numpy arrays:
- `X.shape` → `(274, 66)` — 274 rows, 66 columns
- `y.shape` → `(274,)` — 274 integer labels

---

### 5.7 Train / Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `test_size=0.2` | 20% | Reserve 55 puzzles for testing; train on 219 |
| `stratify=y` | labels | Ensure each of the 6 classes appears in the same proportion in both splits |
| `random_state=42` | seed | Makes the split reproducible — same result every run |

Without `stratify`, random chance could put all examples of one class into the test set, making evaluation misleading.

---

### 5.8 Defining the MLP

```python
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    max_iter=500,
    random_state=42,
    verbose=False,
)
```

Each parameter controls a specific aspect of the architecture or training:

**`hidden_layer_sizes=(64, 32)`**

Defines the network's depth and width. A tuple of integers — each value is the number of neurons in that hidden layer.

```
Input (66) → Hidden₁ (64) → Hidden₂ (32) → Output (6)
```

- `(64, 32)` → 2 hidden layers, getting narrower → the network compresses information
- `(128,)` → 1 wider hidden layer
- `(64, 64, 64)` → 3 equal-width layers → more depth, same width

The output layer size (6) is automatically inferred from the number of unique classes in `y`.

**`activation='relu'`**

The non-linearity applied after each hidden layer. ReLU (`max(0, x)`) is the default choice for most problems — it is fast, doesn't suffer from the vanishing gradient problem, and works well in practice.

Other options: `'tanh'`, `'logistic'` (sigmoid). ReLU is almost always the right choice.

**`max_iter=500`**

Maximum number of training epochs. Training stops early if the loss converges. 500 is generous for a dataset this size.

**`random_state=42`**

Seeds the random number generator for weight initialisation. Without this, results vary between runs. Any integer works — 42 is just a convention.

**`verbose=False`**

Suppresses per-epoch training output. Set to `True` to watch loss decrease during training.

---

### 5.9 Training

```python
model.fit(X_train, y_train)
```

This single line does everything:

1. Initialises weights randomly (using `random_state=42`)
2. Runs up to 500 epochs of forward pass + backpropagation + weight updates
3. Uses **Adam optimiser** (scikit-learn default) — an adaptive learning rate algorithm
4. Stops early if the loss improvement falls below a threshold (`tol=1e-4` by default)

After this line, `model.coefs_` contains the trained weight matrices and `model.intercepts_` contains the bias vectors.

---

### 5.10 Evaluation

```python
y_pred = model.predict(X_test)
acc    = (y_pred == y_test).mean()

print(f"Test accuracy: {acc:.2%}  ({int(acc * len(y_test))}/{len(y_test)} correct)\n")
print(classification_report(y_test, y_pred,
      target_names=[label2name[i] for i in range(6)]))
```

```python
y_pred = model.predict(X_test)
```

Runs a forward pass on all 55 test puzzles. Returns a vector of predicted class indices.

```python
acc = (y_pred == y_test).mean()
```

`y_pred == y_test` creates a boolean array: `True` where prediction matches label.  
`.mean()` averages it: `True=1, False=0`, so the mean is the fraction correct.

```python
classification_report(y_test, y_pred, target_names=[...])
```

Prints a table with three metrics **per class**:

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Precision** | TP / (TP + FP) | Of all puzzles predicted as class X, how many actually were X? |
| **Recall** | TP / (TP + FN) | Of all puzzles that were class X, how many did the model find? |
| **F1** | 2 × P × R / (P + R) | Harmonic mean of precision and recall — single balanced score |
| **Support** | count | Number of test examples in this class |

A precision of 1.00 with recall of 0.88 means: every time the model predicted this class it was right, but it missed some examples of that class.

---

### 5.11 Inspecting Mistakes

```python
wrong = np.where(y_pred != y_test)[0]
if len(wrong):
    print(f"Misclassified ({len(wrong)}):")
    for i in wrong:
        stmts = {}
        for j, person in enumerate(PEOPLE):
            for k, stmt in enumerate(vocab):
                if X_test[i][j * len(vocab) + k] == 1.0:
                    stmts[person] = stmt
```

```python
wrong = np.where(y_pred != y_test)[0]
```

`np.where(condition)` returns the **indices** where condition is True — i.e. where prediction ≠ ground truth. The `[0]` unwraps the tuple that `np.where` returns.

The inner loop **decodes** the one-hot vector back into statement strings — the reverse of `encode()`. For each speaker `j` and each statement `k`, we check if that position in the feature vector is 1.0, and if so, that's the statement that speaker made.

```python
        true_r = role_assignments[y_test[i]]
        pred_r = role_assignments[y_pred[i]]
```

Converts integer labels back to human-readable tuples for printing.

This section is **diagnostic**: the goal is to understand *why* the model makes mistakes. In our case both errors involve confusing Knight and Spy — revealing the limits of surface-level pattern matching.

---

## 6. Reading the Output

Running `python mlp_classifier.py` produces:

```
Puzzles : 274
Features: 66  (3 speakers × 22 statements)
Classes : 6
```

This confirms the data was loaded and encoded correctly.

```
Test accuracy: 96.36%  (53/55 correct)
```

The model correctly identified the roles in 53 out of 55 unseen puzzles.

```
              precision    recall  f1-score   support

 A=K B=K C=S       1.00      0.88      0.93         8
 A=K B=S C=K       0.83      1.00      0.91        10
 A=K B=K C=S       1.00      1.00      1.00         7
 ...
```

Read each row as: *"For puzzles where the answer is [class], the model achieved precision X and recall Y."*

A precision of 0.83 in row 2 means: when the model predicted A=Knight, B=Spy, C=Knave, it was wrong 17% of the time — it sometimes confused this with another class.

```
Misclassified (2):

  A: I am not a Spy
  B: I am not a Spy
  C: B is not a Knight
  True : A=Spy    B=Knight C=Knave
  Pred : A=Knight B=Spy    C=Knave
```

The model swapped A and B's roles. Both A and B said "I am not a Spy" — identical statements — so the only distinguishing signal is C's statement. The model learned the wrong mapping for this specific combination.

---

## 7. Hyperparameter Guide

These are the knobs you can turn to experiment:

| Parameter | Current | Try smaller | Try larger | Effect |
|-----------|---------|-------------|------------|--------|
| `hidden_layer_sizes` | `(64, 32)` | `(32,)` | `(128, 64, 32)` | Smaller = faster, may underfit; larger = more expressive, may overfit |
| `activation` | `'relu'` | `'logistic'` | `'tanh'` | ReLU usually best; logistic can work on small data |
| `max_iter` | `500` | `100` | `2000` | Too few = underfitting; more rarely hurts if early stopping is on |
| `test_size` | `0.2` | `0.1` | `0.3` | Smaller test = more training data; larger = more reliable estimate |
| `random_state` | `42` | `0` | `123` | Change to test robustness — good models are stable across seeds |

A useful experiment: set `verbose=True` and watch the loss curve. If loss is still falling at epoch 500, increase `max_iter`. If it flattens early, training is efficient.

---

## 8. Exercises

Start here if you want to deepen your understanding:

**Beginner**
1. Change `hidden_layer_sizes` to `(32,)` — one smaller hidden layer. Does accuracy drop?
2. Change `test_size` to `0.3`. How many more test puzzles do you get? Does accuracy change?
3. Add `verbose=True` to `MLPClassifier`. Watch the loss print each epoch.

**Intermediate**
4. Add a third hidden layer: `hidden_layer_sizes=(128, 64, 32)`. Does more depth help?
5. Replace `activation='relu'` with `activation='tanh'`. Compare accuracy.
6. Run the script 5 times with `random_state` set to 0, 1, 2, 3, 4. How much does accuracy vary?

**Advanced**
7. The dataset has only 274 puzzles. Use the solver to generate 2,000 more and retrain. Does accuracy improve on the original 55 test puzzles?
8. Replace the one-hot encoding with **TF-IDF** features computed from the raw statement strings. Can the model generalise to statement *phrasings* it has never seen?
9. Add **cross-validation** using `cross_val_score` to get a more reliable accuracy estimate than a single train/test split.

---

## 9. Glossary

| Term | Definition |
|------|-----------|
| **MLP** | Multi-Layer Perceptron — a neural network with one or more hidden layers between input and output |
| **Neuron / Node** | A single unit in a layer — computes a weighted sum of its inputs plus a bias, then applies an activation function |
| **Weight** | A learned parameter — the strength of the connection between two neurons |
| **Bias** | A learned offset added to each neuron's weighted sum, allowing the activation threshold to shift |
| **Activation function** | A non-linear function applied after each layer — breaks the network out of being a simple linear transformation |
| **ReLU** | `max(0, x)` — the most common activation function for hidden layers |
| **Softmax** | Converts a vector of scores into probabilities that sum to 1 — used in the output layer for multi-class problems |
| **Forward pass** | Computing the output by passing data through the network layer by layer |
| **Loss / Cost** | A scalar measuring how wrong the prediction is — training aims to minimise this |
| **Cross-entropy** | The standard loss for classification — penalises low confidence in the correct class |
| **Backpropagation** | Algorithm for computing gradients of the loss with respect to every weight in the network |
| **Gradient descent** | Iteratively updating weights in the direction that reduces the loss |
| **Epoch** | One complete pass through all training data |
| **Learning rate** | How large each weight update step is — too large = unstable, too small = slow |
| **Adam** | An adaptive learning rate optimiser — adjusts the step size per parameter automatically |
| **One-hot encoding** | Representing a categorical value as a binary vector with a single 1 |
| **Feature vector** | The numerical representation of one input example |
| **Train/test split** | Dividing data into a set used for training and a separate set used for final evaluation |
| **Stratification** | Ensuring class proportions are preserved in both train and test splits |
| **Overfitting** | When a model memorises training data but fails to generalise to new examples |
| **Underfitting** | When a model is too simple to capture the patterns in the data |
| **Precision** | Of all predictions for a class, how many were correct |
| **Recall** | Of all true examples of a class, how many were found |
| **F1 score** | Harmonic mean of precision and recall — balanced single metric |
