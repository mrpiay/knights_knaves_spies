"""
Knights, Knaves & Spies — MLP Classifier
=========================================
Input:  3 statements (one-hot encoded) → 66 binary features
Output: one of 6 possible role assignments (A, B, C)
"""

import json
import sys
import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Load data ────────────────────────────────────────────────────
DATA = Path(__file__).parent.parent / 'data' / 'knights_knaves_puzzles.json'
with open(DATA) as f:
    puzzles = json.load(f)

PEOPLE = ['A', 'B', 'C']
ROLES  = ['Knight', 'Knave', 'Spy']

vocab    = sorted(set(s for p in puzzles for s in p['statements'].values()))
stmt2idx = {s: i for i, s in enumerate(vocab)}

role_assignments = [
    ('Knight', 'Knave', 'Spy'),
    ('Knight', 'Spy',   'Knave'),
    ('Knave',  'Knight','Spy'),
    ('Knave',  'Spy',   'Knight'),
    ('Spy',    'Knight','Knave'),
    ('Spy',    'Knave', 'Knight'),
]
label2name = {i: f"A={r[0][0]} B={r[1][0]} C={r[2][0]}"
              for i, r in enumerate(role_assignments)}
assign2label = {r: i for i, r in enumerate(role_assignments)}

# ── Feature encoding ─────────────────────────────────────────────
# Each puzzle → 66-dim binary vector: 3 speakers × 22 statements
def encode(puzzle):
    x = np.zeros(3 * len(vocab), dtype=np.float32)
    for j, person in enumerate(PEOPLE):
        x[j * len(vocab) + stmt2idx[puzzle['statements'][person]]] = 1.0
    y = assign2label[tuple(puzzle['solution'][p] for p in PEOPLE)]
    return x, y

X, y = zip(*[encode(p) for p in puzzles])
X, y = np.array(X), np.array(y)

print(f"Puzzles : {len(X)}")
print(f"Features: {X.shape[1]}  (3 speakers × {len(vocab)} statements)")
print(f"Classes : {len(role_assignments)}\n")

# ── Train / test split ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ── Train MLP ────────────────────────────────────────────────────
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    max_iter=500,
    random_state=42,
    verbose=False,
)

model.fit(X_train, y_train)

# ── Results ──────────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc    = (y_pred == y_test).mean()

print(f"Test accuracy: {acc:.2%}  ({int(acc * len(y_test))}/{len(y_test)} correct)\n")
print(classification_report(y_test, y_pred,
      target_names=[label2name[i] for i in range(6)]))

# ── Show wrong predictions ───────────────────────────────────────
wrong = np.where(y_pred != y_test)[0]
if len(wrong):
    print(f"Misclassified ({len(wrong)}):")
    for i in wrong:
        stmts = {}
        for j, person in enumerate(PEOPLE):
            for k, stmt in enumerate(vocab):
                if X_test[i][j * len(vocab) + k] == 1.0:
                    stmts[person] = stmt
        true_r = role_assignments[y_test[i]]
        pred_r = role_assignments[y_pred[i]]
        print(f"\n  A: {stmts['A']}")
        print(f"  B: {stmts['B']}")
        print(f"  C: {stmts['C']}")
        print(f"  True : A={true_r[0]:6s} B={true_r[1]:6s} C={true_r[2]}")
        print(f"  Pred : A={pred_r[0]:6s} B={pred_r[1]:6s} C={pred_r[2]}")

print("\nDone. Close this window when ready.")
input()
