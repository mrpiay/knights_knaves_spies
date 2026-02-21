"""
Universal Knights, Knaves & Spies Puzzle Solver
================================================
Rules:
  - KNIGHT always tells the truth
  - KNAVE  always lies
  - SPY    can say anything (truth or lie)

There is exactly one Knight, one Knave, and one Spy among A, B, C.

Usage:
    python knights_knaves_solver.py

You can also import solve_puzzle() directly.
"""

from itertools import permutations
import re

# ── Role constants ──────────────────────────────────────────────────────────
KNIGHT = "Knight"
KNAVE  = "Knave"
SPY    = "Spy"
ROLES  = [KNIGHT, KNAVE, SPY]
PEOPLE = ["A", "B", "C"]


# ── Statement evaluator ─────────────────────────────────────────────────────

def evaluate_statement(statement: str, assignment: dict[str, str]) -> bool | None:
    """
    Evaluate whether a statement is TRUE given a role assignment.
    Returns True/False, or None if the statement is unrecognised.

    Supported statement patterns
    ─────────────────────────────
    "I am a Knight / Knave / Spy"
    "I am not a Knight / Knave / Spy"
    "<X> is a Knight / Knave / Spy"
    "<X> is not a Knight / Knave / Spy"
    "If you asked me, I would say that <X> is the spy"
    """
    stmt = statement.strip()

    # ── "If you asked me, I would say that X is the spy" ────────────────────
    # This is a double-nested conditional:
    #   The speaker S says: "If you asked me, I would say X is the spy."
    #   For a Knight  (always truthful): the inner claim "X is the spy" must be true.
    #   For a Knave   (always lies):     the inner claim "X is the spy" must be false,
    #                                    but the Knave would *say* the opposite, so the
    #                                    outer statement is a lie about what they'd say.
    #                                    Net result: X IS the spy.
    #   For a Spy: can say anything — we handle this by not constraining (always accept).
    #
    #   Classic analysis: Knight → X is spy; Knave → X is spy; Spy → unconstrained.
    m = re.fullmatch(
        r"If you asked me, I would say that ([ABC]) is the spy", stmt, re.IGNORECASE
    )
    if m:
        target = m.group(1).upper()
        return assignment[target] == SPY   # true for Knight AND Knave; Spy is unconstrained (handled below)

    # ── "I am a <Role>" ──────────────────────────────────────────────────────
    m = re.fullmatch(r"I am a (Knight|Knave|Spy)", stmt, re.IGNORECASE)
    if m:
        claimed_role = m.group(1).capitalize()
        return True   # placeholder; the actual truth-value is resolved by the speaker

    # ── "I am not a <Role>" ──────────────────────────────────────────────────
    m = re.fullmatch(r"I am not a (Knight|Knave|Spy)", stmt, re.IGNORECASE)
    if m:
        claimed_role = m.group(1).capitalize()
        return True   # placeholder

    # ── "<X> is a <Role>" ────────────────────────────────────────────────────
    m = re.fullmatch(r"([ABC]) is a (Knight|Knave|Spy)", stmt, re.IGNORECASE)
    if m:
        target, claimed_role = m.group(1).upper(), m.group(2).capitalize()
        return assignment[target] == claimed_role

    # ── "<X> is not a <Role>" ────────────────────────────────────────────────
    m = re.fullmatch(r"([ABC]) is not a (Knight|Knave|Spy)", stmt, re.IGNORECASE)
    if m:
        target, claimed_role = m.group(1).upper(), m.group(2).capitalize()
        return assignment[target] != claimed_role

    return None  # unrecognised statement


def statement_truth_value(
    speaker: str,
    statement: str,
    assignment: dict[str, str],
) -> bool:
    """
    Return the *actual* truth value of `statement` given `assignment`,
    taking into account first-person "I am / I am not" phrasing.
    """
    stmt = statement.strip()

    # Re-parse first-person claims with the speaker's actual role
    m = re.fullmatch(r"I am a (Knight|Knave|Spy)", stmt, re.IGNORECASE)
    if m:
        return assignment[speaker] == m.group(1).capitalize()

    m = re.fullmatch(r"I am not a (Knight|Knave|Spy)", stmt, re.IGNORECASE)
    if m:
        return assignment[speaker] != m.group(1).capitalize()

    # For "If you asked me…" the result depends only on assignment, not speaker type
    m = re.fullmatch(
        r"If you asked me, I would say that ([ABC]) is the spy", stmt, re.IGNORECASE
    )
    if m:
        target = m.group(1).upper()
        return assignment[target] == SPY   # always the factual content

    # Everything else is a direct third-person factual claim
    result = evaluate_statement(stmt, assignment)
    return result if result is not None else True


def is_consistent(
    speaker: str,
    statement: str,
    assignment: dict[str, str],
) -> bool:
    """
    Check whether a speaker's statement is consistent with their role.

    Knight: statement must be TRUE
    Knave:  statement must be FALSE
    Spy:    statement can be anything → always consistent

    Special case — "If you asked me, I would say that X is the spy":
      The actual truth-value of this statement depends on the speaker's role:
        Knight would say "X is spy" iff X IS the spy
          → outer statement TRUE iff X IS spy
        Knave would say "X is spy" iff X is NOT the spy (Knave lies)
          → outer statement TRUE iff X is NOT spy
          → Knave must output FALSE → outer must be FALSE → X IS the spy
        Spy: unconstrained
    """
    role = assignment[speaker]
    if role == SPY:
        return True

    # Handle conditional separately since its truth depends on the speaker's role
    m = re.fullmatch(
        r"If you asked me, I would say that ([ABC]) is the spy",
        statement.strip(), re.IGNORECASE
    )
    if m:
        target = m.group(1).upper()
        x_is_spy = (assignment[target] == SPY)
        if role == KNIGHT:
            # Knight says truth: "I would say X is spy" is true iff X IS spy
            return x_is_spy
        if role == KNAVE:
            # Knave would say "X is spy" iff X is NOT spy (Knave lies about inner)
            # So outer statement "I would say X is spy" is TRUE iff X is NOT spy
            # Knave must output FALSE → outer must be FALSE → NOT (x is NOT spy) → x IS spy
            return x_is_spy   # consistent iff X is the spy (same direction as Knight!)
        return True

    truth = statement_truth_value(speaker, statement, assignment)

    if role == KNIGHT:
        return truth is True
    if role == KNAVE:
        return truth is False

    return True


# ── Core solver ─────────────────────────────────────────────────────────────

def solve_puzzle(
    statements: dict[str, str],
    *,
    verbose: bool = False,
) -> list[dict[str, str]]:
    """
    Given a dict of {person: statement}, return all valid role assignments.

    Parameters
    ----------
    statements : dict  e.g. {"A": "I am a Knight", "B": "C is a Spy", "C": "B is a Knave"}
    verbose    : bool  if True, print which assignments pass/fail

    Returns
    -------
    List of dicts.  Each dict maps person → role.
    Empty list  → paradox / no solution.
    Multiple    → ambiguous puzzle.
    Singleton   → unique solution.
    """
    people = list(statements.keys())
    valid_assignments = []

    for role_perm in permutations(ROLES):
        assignment = dict(zip(people, role_perm))
        ok = all(
            is_consistent(person, statements[person], assignment)
            for person in people
        )
        if verbose:
            label = "✓" if ok else "✗"
            print(f"  {label}  {assignment}")
        if ok:
            valid_assignments.append(assignment)

    return valid_assignments


# ── Pretty printer ───────────────────────────────────────────────────────────

def format_solution(
    statements: dict[str, str],
    solutions: list[dict[str, str]],
) -> str:
    lines = ["┌─ Puzzle ──────────────────────────────────────┐"]
    for person, stmt in statements.items():
        lines.append(f"│  {person}: {stmt}")
    lines.append("└───────────────────────────────────────────────┘")

    if not solutions:
        lines.append("⚠️  No valid assignment exists (paradox).")
    elif len(solutions) == 1:
        sol = solutions[0]
        lines.append("✅  Unique solution:")
        for person in statements:
            lines.append(f"     {person} is the {sol[person]}")
    else:
        lines.append(f"⚠️  Ambiguous — {len(solutions)} valid assignments:")
        for sol in solutions:
            row = "  |  ".join(f"{p}={sol[p]}" for p in statements)
            lines.append(f"     {row}")

    return "\n".join(lines)


# ── Interactive CLI ──────────────────────────────────────────────────────────

EXAMPLE_PUZZLES = [
    # From the file — verified solutions
    {
        "A": "I am a Knight",
        "B": "I am not a Spy",
        "C": "I am a Knave",
    },  # expected: FTS  → A=Knave, B=Knight, C=Spy
    {
        "A": "I am a Knight",
        "B": "I am a Knave",
        "C": "B is a Knave",
    },  # expected: TSF
    {
        "A": "I am not a Spy",
        "B": "I am a Spy",
        "C": "I am a Knave",
    },  # expected: TFS
    {
        "A": "If you asked me, I would say that A is the spy",
        "B": "If you asked me, I would say that B is the spy",
        "C": "If you asked me, I would say that C is the spy",
    },  # all three claim to be spy via conditional — interesting edge case
    {
        "A": "B is a Spy",
        "B": "A is a Spy",
        "C": "A is a Knight",
    },  # expected: STF
    {
        "A": "C is not a Knight",
        "B": "C is a Spy",
        "C": "B is not a Knave",
    },  # expected: FST
]


def run_examples():
    print("=" * 52)
    print("  Knights, Knaves & Spies — Universal Solver")
    print("=" * 52)
    for i, puzzle in enumerate(EXAMPLE_PUZZLES, 1):
        solutions = solve_puzzle(puzzle)
        print(f"\n── Example {i} ──")
        print(format_solution(puzzle, solutions))


def interactive():
    print("\n── Enter your own puzzle ──")
    print("For each person (A, B, C) enter their statement.")
    print("Supported forms:")
    print("  I am a Knight/Knave/Spy")
    print("  I am not a Knight/Knave/Spy")
    print("  X is a Knight/Knave/Spy           (X ∈ A B C)")
    print("  X is not a Knight/Knave/Spy")
    print("  If you asked me, I would say that X is the spy")
    print()
    statements = {}
    for person in PEOPLE:
        stmt = input(f"  {person} says: ").strip()
        statements[person] = stmt
    solutions = solve_puzzle(statements, verbose=True)
    print()
    print(format_solution(statements, solutions))


if __name__ == "__main__":
    run_examples()
    print()
    again = input("Try your own puzzle? (y/n): ").strip().lower()
    if again == "y":
        interactive()
