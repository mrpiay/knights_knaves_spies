import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from knights_knaves_solver import solve_puzzle

DATA = Path(__file__).parent.parent / 'data' / 'knights_knaves_puzzles.json'
with open(DATA) as f:
    puzzles = json.load(f)

print(f"Total puzzles: {len(puzzles)}\n")
print("=" * 52)

correct = 0
wrong = 0

for i, p in enumerate(puzzles):
    stmts = p['statements']
    expected = p['solution']
    solutions = solve_puzzle(stmts)

    passed = len(solutions) == 1 and solutions[0] == expected
    status = "✅ PASS" if passed else "❌ FAIL"

    if passed:
        correct += 1
    else:
        wrong += 1

    print(f"\nPuzzle #{i+1:3d}  {status}")
    print(f"  A: {stmts['A']}")
    print(f"  B: {stmts['B']}")
    print(f"  C: {stmts['C']}")
    print(f"  Expected : A={expected['A']:6s}  B={expected['B']:6s}  C={expected['C']}")

    if not passed:
        if not solutions:
            print(f"  Got      : no valid assignment found")
        else:
            for s in solutions:
                print(f"  Got      : A={s['A']:6s}  B={s['B']:6s}  C={s['C']}")

print("\n" + "=" * 52)
print(f"Results: {correct} correct, {wrong} wrong out of {len(puzzles)} puzzles")
print("=" * 52)
print("\nDone. Close this window when you're ready.")
input()
