#!/usr/bin/env python3
"""
Group students by preferences first and then by proficiency, producing target-sized groups and reports.
Includes an exact CP-SAT solver mode that maximizes the number of satisfied preferences under hard group-size constraints.
Outputs plots, TSV/TXT listings, an unmet-preferences report, and a network map PNG.
"""

from __future__ import annotations
import argparse
import collections
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.patches import Circle
import warnings

# Quiet seaborn/pandas deprecation chatter that doesn’t affect results
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

VALID_PROF = ["none", "basic", "intermediate", "proficient"]  # fixed order for plots


@dataclass
class Student:
    """Container for one student's name, proficiency, and preference list."""
    name: str
    proficiency: str
    pref_names: List[str] = field(default_factory=list)


def load_input(tsv_path: str) -> List[Student]:
    """Read and normalize the input TSV (name, proficiency, preferences)."""
    df = pd.read_csv(tsv_path, sep="\t", dtype=str).fillna("")
    cols = {c.lower().strip(): c for c in df.columns}
    for required in ["name", "proficiency", "preferences"]:
        if required not in cols:
            raise ValueError(f"Missing required column '{required}' in input file.")
    df = df.rename(columns={cols["name"]: "name",
                            cols["proficiency"]: "proficiency",
                            cols["preferences"]: "preferences"})
    students: List[Student] = []
    for _, row in df.iterrows():
        name = str(row["name"]).strip()
        prof = str(row["proficiency"]).strip().lower()
        if prof not in VALID_PROF:
            raise ValueError(f"Invalid proficiency '{row['proficiency']}' for student '{name}'. "
                             f"Allowed: {', '.join(VALID_PROF)}")
        prefs_raw = str(row["preferences"]).strip()
        pref_names = [p.strip() for p in prefs_raw.split(";") if p.strip()] if prefs_raw else []
        students.append(Student(name=name, proficiency=prof, pref_names=pref_names))
    return students


def compute_target_group_sizes(n: int, group_size: int) -> List[int]:
    """Compute target sizes: ⌊n/s⌋ groups with size s and the first (n mod s) groups with size s+1."""
    if n <= 0:
        return []
    if n <= group_size:
        return [n]
    g = n // group_size
    r = n % group_size
    sizes = [(group_size + 1 if i < r else group_size) for i in range(g)]
    assert sum(sizes) == n
    return sizes


def group_students(
    students: List[Student],
    group_size: int,
    style: str,
    rng: np.random.Generator
) -> Tuple[List[List[str]], Dict[str, int]]:
    """Assign students to groups using a preference-first pass then proficiency-based fill (similar/dissimilar)."""
    assert style in {"similar", "dissimilar"}
    n = len(students)
    if n == 0:
        return [], {}

    target_sizes = compute_target_group_sizes(n, group_size)
    num_groups = len(target_sizes)

    name_to_student: Dict[str, Student] = {s.name: s for s in students}
    lower_to_name = {s.name.lower(): s.name for s in students}

    # Normalize preference casing and drop unknowns
    for s in students:
        cleaned = []
        for pref in s.pref_names:
            key = pref.lower()
            if key in lower_to_name:
                cleaned.append(lower_to_name[key])
        s.pref_names = list(dict.fromkeys(cleaned))

    unassigned: Set[str] = set(s.name for s in students)
    groups: List[List[str]] = [[] for _ in range(num_groups)]

    by_prof: Dict[str, List[str]] = {p: [] for p in VALID_PROF}
    for s in students:
        by_prof[s.proficiency].append(s.name)
    for p in by_prof:
        rng.shuffle(by_prof[p])

    def pop_from_prof(p: str) -> Optional[str]:
        """Pop one unassigned student from a proficiency bucket."""
        while by_prof[p]:
            cand = by_prof[p].pop()
            if cand in unassigned:
                return cand
        return None

    def current_prof_counts(members: List[str]) -> Dict[str, int]:
        """Count proficiencies in a group."""
        cnt = {p: 0 for p in VALID_PROF}
        for nm in members:
            cnt[name_to_student[nm].proficiency] += 1
        return cnt

    def place_in_existing_group_with_pref(target_name: str) -> bool:
        """Try to place a student into a group containing any of their preferences."""
        prefs = set(name_to_student[target_name].pref_names)
        if not prefs:
            return False
        for gi, members in enumerate(groups):
            if len(members) >= target_sizes[gi]:
                continue
            if prefs.intersection(members):
                groups[gi].append(target_name)
                unassigned.discard(target_name)
                return True
        return False

    def open_new_group_and_seed(seed_name: str) -> bool:
        """Start the smallest not-full group with this student, then add their preferred partners if available."""
        candidates = [gi for gi in range(num_groups) if len(groups[gi]) < target_sizes[gi]]
        if not candidates:
            return False
        candidates.sort(key=lambda gi: len(groups[gi]))
        gi = candidates[0]
        groups[gi].append(seed_name)
        unassigned.discard(seed_name)
        for pref in name_to_student[seed_name].pref_names:
            if len(groups[gi]) >= target_sizes[gi]:
                break
            if pref in unassigned:
                groups[gi].append(pref)
                unassigned.discard(pref)
        return True

    pref_students = [s.name for s in students if s.pref_names]
    rng.shuffle(pref_students)
    pref_students.sort(key=lambda nm: len(name_to_student[nm].pref_names), reverse=True)
    for nm in list(pref_students):
        if nm not in unassigned:
            continue
        if place_in_existing_group_with_pref(nm):
            continue
        open_new_group_and_seed(nm)

    def fill_group_similar(gi: int):
        """Fill a group by matching its dominant proficiency (or the most abundant overall if empty)."""
        if len(groups[gi]) >= target_sizes[gi]:
            return
        counts = current_prof_counts(groups[gi])
        if sum(counts.values()) == 0:
            avail_counts = {p: sum(1 for nm in by_prof[p] if nm in unassigned) for p in VALID_PROF}
            for p in sorted(VALID_PROF, key=lambda p: avail_counts[p], reverse=True):
                cand = pop_from_prof(p)
                if cand:
                    groups[gi].append(cand); unassigned.discard(cand); return
            return
        dominant = sorted(VALID_PROF, key=lambda p: counts[p], reverse=True)[0]
        cand = pop_from_prof(dominant)
        if cand is None:
            for p in VALID_PROF:
                cand = pop_from_prof(p)
                if cand is not None:
                    break
        if cand is not None:
            groups[gi].append(cand); unassigned.discard(cand)

    def fill_group_dissimilar(gi: int):
        """Fill a group by choosing the least represented proficiency available to balance composition."""
        if len(groups[gi]) >= target_sizes[gi]:
            return
        counts = current_prof_counts(groups[gi])
        avail_counts = {p: sum(1 for nm in by_prof[p] if nm in unassigned) for p in VALID_PROF}
        for p in sorted(VALID_PROF, key=lambda p: (counts[p], -avail_counts[p])):
            cand = pop_from_prof(p)
            if cand is not None:
                groups[gi].append(cand); unassigned.discard(cand); return

    while unassigned:
        progressed = False
        for gi in range(num_groups):
            if len(groups[gi]) >= target_sizes[gi]:
                continue
            before = len(groups[gi])
            (fill_group_similar if style == "similar" else fill_group_dissimilar)(gi)
            progressed |= (len(groups[gi]) > before)
            if not unassigned:
                break
        if not progressed:
            for gi in range(num_groups):
                while len(groups[gi]) < target_sizes[gi] and unassigned:
                    any_prof = next((p for p in VALID_PROF if any(nm in unassigned for nm in by_prof[p])), None)
                    fallback = pop_from_prof(any_prof) if any_prof else next(iter(unassigned))
                    groups[gi].append(fallback); unassigned.discard(fallback)
            break

    membership: Dict[str, int] = {}
    for gi, g in enumerate(groups):
        for nm in g:
            membership[nm] = gi
    return groups, membership


def solve_optimal_groups_exact(
    students: List[Student],
    group_sizes: List[int],
    mutual_weight: int = 2,
    time_limit_sec: Optional[float] = 60.0,
    workers: Optional[int] = None,
) -> Tuple[List[List[str]], float, str]:
    """Solve for global maximum satisfied preferences with CP-SAT (returns groups, objective value, solver status)."""
    try:
        from ortools.sat.python import cp_model
    except Exception as e:
        raise RuntimeError(
            "Exact optimization requires OR-Tools. Install with: pip install ortools"
        ) from e

    names = [s.name for s in students]
    n = len(names)
    G = len(group_sizes)
    idx = {name: i for i, name in enumerate(names)}

    # Build directed preference list and weights; upweight mutual preferences if desired.
    prefs: List[Tuple[int, int]] = []
    pref_set = {(i, j) for i in range(n) for j in range(n)}
    raw_pref = {i: set() for i in range(n)}
    for s in students:
        i = idx[s.name]
        for p in s.pref_names:
            if p in idx:
                raw_pref[i].add(idx[p])
    weights: Dict[Tuple[int, int], int] = {}
    for i in range(n):
        for j in raw_pref[i]:
            w = 1
            if mutual_weight and i in raw_pref.get(j, set()):
                w = mutual_weight
            prefs.append((i, j))
            weights[(i, j)] = w

    m = cp_model.CpModel()
    x = {(i, g): m.NewBoolVar(f"x_{i}_{g}") for i in range(n) for g in range(G)}
    y = {(i, j, g): m.NewBoolVar(f"y_{i}_{j}_{g}") for (i, j) in prefs for g in range(G)}

    # Each student exactly one group
    for i in range(n):
        m.Add(sum(x[i, g] for g in range(G)) == 1)

    # Exact group sizes
    for g in range(G):
        m.Add(sum(x[i, g] for i in range(n)) == group_sizes[g])

    # Same-group linearization for each preferred pair
    for (i, j) in prefs:
        for g in range(G):
            m.Add(y[i, j, g] <= x[i, g])
            m.Add(y[i, j, g] <= x[j, g])
            m.Add(y[i, j, g] >= x[i, g] + x[j, g] - 1)

    # Objective: maximize total satisfied preferences (weighted)
    m.Maximize(sum(weights[(i, j)] * y[i, j, g] for (i, j) in prefs for g in range(G)))

    solver = cp_model.CpSolver()
    if time_limit_sec is not None:
        solver.parameters.max_time_in_seconds = float(time_limit_sec)
    if workers is not None:
        solver.parameters.num_search_workers = int(workers)

    status = solver.Solve(m)
    groups = [[] for _ in range(G)]
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i in range(n):
            for g in range(G):
                if solver.Value(x[i, g]):
                    groups[g].append(names[i])
        return groups, solver.ObjectiveValue(), solver.StatusName(status)
    else:
        return [[] for _ in range(G)], 0.0, solver.StatusName(status)


def save_proficiency_distribution(students: List[Student], out_stem: str):
    """Save overall proficiency counts as TSV and bar plot."""
    df = pd.DataFrame({"proficiency": [s.proficiency for s in students]})
    counts = df["proficiency"].value_counts().reindex(VALID_PROF, fill_value=0)
    counts_df = counts.rename_axis("proficiency").reset_index(name="count")
    counts_df.to_csv(f"{out_stem}__proficiency_distribution.tsv", sep="\t", index=False)

    plt.figure()
    sns.barplot(data=counts_df, x="proficiency", y="count", order=VALID_PROF)
    plt.title("Overall Proficiency Distribution")
    plt.xlabel("Proficiency"); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{out_stem}__proficiency_distribution.png", dpi=600)
    plt.close()


def group_combo_label(group_members: List[str], name_to_student: Dict[str, Student]) -> str:
    """Return a label like '2 basic + 1 none' describing a group's composition."""
    counts = collections.Counter(name_to_student[nm].proficiency for nm in group_members)
    parts = [f"{counts[p]} {p}" for p in VALID_PROF if counts[p] > 0]
    return " + ".join(parts) if parts else "empty"


def save_group_combo_plot(groups: List[List[str]], students: List[Student], out_stem: str):
    """Save group composition frequencies as TSV and bar plot."""
    name_to_student = {s.name: s for s in students}
    labels = [group_combo_label(g, name_to_student) for g in groups]
    freq = collections.Counter(labels)
    combos_df = pd.DataFrame({"combo": list(freq.keys()), "count": list(freq.values())}).sort_values("combo")
    combos_df.to_csv(f"{out_stem}__group_proficiency_combos.tsv", sep="\t", index=False)

    plt.figure()
    sns.barplot(data=combos_df, x="combo", y="count")
    plt.title("Group Proficiency Combinations")
    plt.xlabel("Combination"); plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{out_stem}__group_proficiency_combos.png", dpi=600)
    plt.close()


def save_unmet_preferences(groups: List[List[str]], students: List[Student], out_stem: str):
    """Save a TSV listing students whose preferences weren’t fully met."""
    name_to_group: Dict[str, int] = {nm: gi for gi, g in enumerate(groups) for nm in g}
    rows = []
    for s in students:
        if not s.pref_names:
            continue
        gi = name_to_group.get(s.name, None)
        if gi is None:
            continue
        group_set = set(groups[gi])
        unmet = [p for p in s.pref_names if p not in group_set]
        if unmet:
            rows.append({
                "student": s.name,
                "assigned_group_members": "; ".join(sorted([nm for nm in groups[gi] if nm != s.name])),
                "input_preferences": "; ".join(s.pref_names),
                "unmet_preferences": "; ".join(unmet)
            })
    pd.DataFrame(rows, columns=["student","assigned_group_members","input_preferences","unmet_preferences"]) \
      .to_csv(f"{out_stem}__unmet_preferences.tsv", sep="\t", index=False)


def save_network_map(groups: List[List[str]], students: List[Student], out_stem: str, seed: Optional[int] = None):
    """Draw a clustered, non-overlapping preference network (nodes colored by group) and save as PNG."""
    G = nx.DiGraph()
    name_to_group: Dict[str, int] = {}
    for gi, g in enumerate(groups):
        for nm in g:
            name_to_group[nm] = gi
            G.add_node(nm, group=gi)

    roster = set(name_to_group)
    for s in students:
        for pref in s.pref_names:
            if s.name in roster and pref in roster:
                G.add_edge(s.name, pref)

    if G.number_of_nodes() == 0:
        plt.figure(figsize=(10, 8))
        plt.title("Student Preference Network (no nodes)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{out_stem}__network_map.png", dpi=600)
        plt.close()
        return

    rng = np.random.default_rng(int(seed) if seed is not None else None)
    seed_int = int(seed) if seed is not None else None

    num_groups = len(groups)
    angle_step = 2 * np.pi / max(num_groups, 1)
    cluster_radius = 8.0
    group_centers = {
        gi: np.array([cluster_radius * np.cos(gi * angle_step),
                      cluster_radius * np.sin(gi * angle_step)])
        for gi in range(num_groups)
    }

    positions: Dict[str, np.ndarray] = {}
    local_radius = 1.8
    base_k = 0.8
    for gi, members in enumerate(groups):
        if not members:
            continue
        H = G.subgraph(members).copy()
        k = base_k / max(np.sqrt(len(members)), 1.0)
        local_pos = nx.spring_layout(H, seed=seed_int, k=k, iterations=300)
        pts = np.array(list(local_pos.values()))
        if len(pts) > 0:
            pts = pts - pts.mean(axis=0)
            denom = np.max(np.linalg.norm(pts, axis=1)) or 1.0
            pts = (pts / denom) * local_radius
        for (node, _), p in zip(local_pos.items(), pts):
            positions[node] = group_centers[gi] + p

    min_sep = 0.35
    nodes = list(positions.keys())
    for _ in range(60):
        moved = False
        for i in range(len(nodes)):
            ni = nodes[i]; pi = positions[ni]
            for j in range(i + 1, len(nodes)):
                nj = nodes[j]; pj = positions[nj]
                delta = pi - pj
                dist = np.linalg.norm(delta)
                if dist < 1e-9:
                    jitter = rng.normal(scale=1e-3, size=2)
                    positions[ni] = pi + jitter
                    positions[nj] = pj - jitter
                    moved = True
                elif dist < min_sep:
                    push = (min_sep - dist) / 2.0
                    unit = delta / dist
                    positions[ni] = pi + unit * push
                    positions[nj] = pj - unit * push
                    moved = True
        if not moved:
            break

    plt.figure(figsize=(11, 9))
    pos = {n: (positions[n][0], positions[n][1]) for n in G.nodes()}
    node_colors = [name_to_group[nm] for nm in G.nodes()]

    for gi, center in group_centers.items():
        if not groups[gi]:
            continue
        circle = Circle(center, local_radius * 1.15, facecolor="none", edgecolor="lightgray", linewidth=1.0, zorder=0)
        plt.gca().add_patch(circle)

    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=10, width=0.9, alpha=0.5, edge_color="gray")
    nx.draw_networkx_nodes(G, pos, node_size=420, node_color=node_colors, cmap=plt.cm.tab20,
                           linewidths=1.0, edgecolors="black")
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.axis("off")
    plt.title("Student Preference Network (group-clustered, non-overlapping)")
    plt.tight_layout()
    plt.savefig(f"{out_stem}__network_map.png", dpi=600)
    plt.close()


def save_groups_tsv(groups: List[List[str]], students: List[Student], out_stem: str):
    """Save a TSV mapping group index to student name and proficiency."""
    name_to_prof = {s.name: s.proficiency for s in students}
    rows = [{"group_index": gi, "name": nm, "proficiency": name_to_prof[nm]}
            for gi, g in enumerate(groups) for nm in g]
    pd.DataFrame(rows).to_csv(f"{out_stem}__assigned_groups.tsv", sep="\t", index=False)


def save_groups_txt(groups: List[List[str]], out_stem: str):
    """Save a human-readable TXT listing of each group and its members (names only)."""
    lines = []
    for gi, g in enumerate(groups):
        lines.append(f"Group {gi} ({len(g)}): " + ", ".join(g))
    with open(f"{out_stem}__groups.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    """Parse CLI args, run grouping (heuristic or exact), and write all outputs."""
    parser = argparse.ArgumentParser(description="Group students; heuristic or exact solver; emit plots and reports.")
    parser.add_argument("input_tsv", help="Path to input TSV with name, proficiency, preferences")
    parser.add_argument("--group-size", type=int, required=True, help="Ideal group size (>=2 recommended)")
    parser.add_argument("--style", choices=["similar", "dissimilar"], required=True,
                        help="How to fill groups by proficiency after respecting preferences (heuristic mode).")
    parser.add_argument("--out-stem", type=str, default=None,
                        help="Optional output filename stem. Defaults to input stem.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (0…4294967295).")

    # Exact solver options
    parser.add_argument("--optimize", choices=["heuristic", "exact"], default="heuristic",
                        help="Use heuristic (default) or exact CP-SAT to maximize satisfied preferences.")
    parser.add_argument("--mutual-weight", type=int, default=2,
                        help="Weight for mutual preferences in exact mode (one-way is weight 1).")
    parser.add_argument("--time-limit", type=float, default=60.0,
                        help="Time limit in seconds for exact mode (set 0 or negative for no limit).")
    parser.add_argument("--workers", type=int, default=None,
                        help="CP-SAT parallel workers in exact mode (default uses OR-Tools default).")

    args = parser.parse_args()

    if args.group_size <= 0:
        print("ERROR: --group-size must be > 0", file=sys.stderr)
        sys.exit(2)

    in_stem = os.path.splitext(os.path.basename(args.input_tsv))[0]
    out_stem = args.out_stem or in_stem

    students = load_input(args.input_tsv)

    if args.optimize == "exact":
        sizes = compute_target_group_sizes(len(students), args.group_size)
        tl = None if args.time_limit and args.time_limit <= 0 else float(args.time_limit)
        try:
            groups, obj, status = solve_optimal_groups_exact(
                students,
                sizes,
                mutual_weight=args.mutual_weight,
                time_limit_sec=tl,
                workers=args.workers,
            )
            if not any(groups):
                print(f"Exact solver returned status {status} and no assignment; falling back to heuristic.", file=sys.stderr)
                rng = np.random.default_rng(int(args.seed))
                groups, _ = group_students(students, args.group_size, args.style, rng=rng)
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            print("Falling back to heuristic mode.", file=sys.stderr)
            rng = np.random.default_rng(int(args.seed))
            groups, _ = group_students(students, args.group_size, args.style, rng=rng)
    else:
        rng = np.random.default_rng(int(args.seed))
        groups, _ = group_students(students, args.group_size, args.style, rng=rng)

    save_proficiency_distribution(students, out_stem)
    save_group_combo_plot(groups, students, out_stem)
    save_unmet_preferences(groups, students, out_stem)
    save_network_map(groups, students, out_stem, seed=int(args.seed))
    save_groups_tsv(groups, students, out_stem)
    save_groups_txt(groups, out_stem)

    print(f"Created {len(groups)} groups. Output stem: {out_stem}")
    for i, g in enumerate(groups):
        print(f"Group {i} ({len(g)}): {', '.join(g)}")


if __name__ == "__main__":
    main()
