#!/usr/bin/env python3
"""
Group students by preferences and proficiency, and output plots + reports.

Usage:
  python group_students.py input.tsv --group-size 3 --style dissimilar [--out-stem my_run] [--seed 42]

Input TSV columns:
  - name (str)
  - proficiency (one of: proficient, intermediate, basic, none)
  - preferences (semicolon-delimited list of names; empty allowed)
"""

from __future__ import annotations
import argparse
import collections
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

VALID_PROF = ["none", "basic", "intermediate", "proficient"]  # order for plots

# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Student:
    name: str
    proficiency: str
    pref_names: List[str] = field(default_factory=list)

# -----------------------------
# IO & parsing
# -----------------------------

def load_input(tsv_path: str) -> List[Student]:
    df = pd.read_csv(tsv_path, sep="\t", dtype=str).fillna("")
    # normalize columns
    cols = {c.lower().strip(): c for c in df.columns}
    for required in ["name", "proficiency", "preferences"]:
        if required not in cols:
            raise ValueError(f"Missing required column '{required}' in input file.")
    df = df.rename(columns={cols["name"]: "name",
                            cols["proficiency"]: "proficiency",
                            cols["preferences"]: "preferences"})
    # normalize entries
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

# -----------------------------
# Group sizing logic
# -----------------------------

def compute_target_group_sizes(n: int, group_size: int) -> List[int]:
    """Create as many groups of size 'group_size' as possible and
    distribute stragglers to make r groups with size 'group_size+1'.

    If n < group_size, return [n].
    """
    if n <= 0:
        return []
    if n <= group_size:
        return [n]
    g = n // group_size  # number of groups
    r = n % group_size   # number of groups that will have +1
    sizes = []
    # First r groups have size group_size+1, others have group_size
    for i in range(g):
        sizes.append(group_size + (1 if i < r else 0))
    # Defensive assert:
    assert sum(sizes) == n, f"Target sizes {sizes} do not sum to n={n}"
    return sizes

# -----------------------------
# Grouping algorithm
# -----------------------------

def group_students(
    students: List[Student],
    group_size: int,
    style: str,
    rng: np.random.Generator
) -> Tuple[List[List[str]], Dict[str, int]]:
    """
    Preference-first, then proficiency-based filling.

    style: 'similar' or 'dissimilar'
    Returns:
      groups: list of lists of student names
      membership: mapping name -> group_index
    """
    assert style in {"similar", "dissimilar"}
    n = len(students)
    if n == 0:
        return [], {}

    target_sizes = compute_target_group_sizes(n, group_size)
    if not target_sizes:
        return [], {}
    num_groups = len(target_sizes)

    name_to_student: Dict[str, Student] = {s.name: s for s in students}
    # Build a lowercase map for resolving case-insensitive preference matching
    lower_to_name = {s.name.lower(): s.name for s in students}

    # Clean preference names to match canonical casing; drop unknown prefs
    for s in students:
        cleaned = []
        for pref in s.pref_names:
            key = pref.lower()
            if key in lower_to_name:
                cleaned.append(lower_to_name[key])
        s.pref_names = list(dict.fromkeys(cleaned))  # dedupe, preserve order

    # Bookkeeping
    unassigned: Set[str] = set(s.name for s in students)
    groups: List[List[str]] = [[] for _ in range(num_groups)]
    capacities = target_sizes.copy()  # remaining capacity per group

    # Helper queues by proficiency
    by_prof: Dict[str, List[str]] = {p: [] for p in VALID_PROF}
    for s in students:
        by_prof[s.proficiency].append(s.name)
    # We'll pop from these; shuffle to avoid bias
    for p in by_prof:
        rng.shuffle(by_prof[p])

    def pop_from_prof(p: str) -> Optional[str]:
        while by_prof[p]:
            cand = by_prof[p].pop()
            if cand in unassigned:
                return cand
        return None

    def current_prof_counts(members: List[str]) -> Dict[str, int]:
        cnt = {p: 0 for p in VALID_PROF}
        for nm in members:
            cnt[name_to_student[nm].proficiency] += 1
        return cnt

    def place_in_existing_group_with_pref(target_name: str) -> bool:
        """Try to place into a group that already contains any of their preferences."""
        prefs = set(name_to_student[target_name].pref_names)
        if not prefs:
            return False
        for gi, members in enumerate(groups):
            if capacities[gi] <= 0:
                continue
            if prefs.intersection(members):
                groups[gi].append(target_name)
                unassigned.discard(target_name)
                return True
        return False

    def open_new_group_and_seed(seed_name: str) -> bool:
        """Open the next group with free capacity and seed with this student, then try to add their preferred partners."""
        # pick first group with available capacity and empty or smallest current size
        candidates = [gi for gi in range(num_groups) if capacities[gi] > 0]
        if not candidates:
            return False
        # Prefer the smallest (to fill from empty up)
        candidates.sort(key=lambda gi: len(groups[gi]))
        gi = candidates[0]
        groups[gi].append(seed_name)
        unassigned.discard(seed_name)
        # try to add the seed's preferred partners
        for pref in name_to_student[seed_name].pref_names:
            if capacities[gi] - len(groups[gi]) <= 0:
                break
            if pref in unassigned:
                groups[gi].append(pref)
                unassigned.discard(pref)
        return True

    # -------- Pass 1: preference-driven placement --------
    # Order by number of preferences (desc), then random for tie-breaking
    pref_students = [s.name for s in students if len(s.pref_names) > 0]
    rng.shuffle(pref_students)
    pref_students.sort(key=lambda nm: len(name_to_student[nm].pref_names), reverse=True)

    for nm in list(pref_students):
        if nm not in unassigned:
            continue
        # Try to sit next to preferred partners if they already exist in a group
        if place_in_existing_group_with_pref(nm):
            continue
        # Otherwise open a new group slot and seed it with this student and their preferred partners
        open_new_group_and_seed(nm)

    # -------- Pass 2: fill remaining seats based on style --------
    # Prepare list of groups sorted by remaining capacity so we always fill those first
    def fill_group_similar(gi: int):
        # Aim to match dominant proficiency in the group
        if capacities[gi] - len(groups[gi]) <= 0:
            return
        counts = current_prof_counts(groups[gi])
        # If group empty, just take from the most abundant pool overall
        if sum(counts.values()) == 0:
            # choose prof with most remaining candidates
            avail_counts = {p: sum(1 for nm in by_prof[p] if nm in unassigned) for p in VALID_PROF}
            choices = sorted(VALID_PROF, key=lambda p: avail_counts[p], reverse=True)
            chosen = None
            for p in choices:
                cand = pop_from_prof(p)
                if cand:
                    groups[gi].append(cand)
                    unassigned.discard(cand)
                    return
            return
        # find dominant proficiency in current group
        dominant = sorted(VALID_PROF, key=lambda p: counts[p], reverse=True)[0]
        # first try dominant
        cand = pop_from_prof(dominant)
        if cand is None:
            # try any other with availability, preferring nearest neighbors (same-ish level)
            for p in VALID_PROF:
                cand = pop_from_prof(p)
                if cand is not None:
                    break
        if cand is not None:
            groups[gi].append(cand)
            unassigned.discard(cand)

    def fill_group_dissimilar(gi: int):
        # Aim to balance mix: pick the proficiency with lowest current count in this group (if available)
        if capacities[gi] - len(groups[gi]) <= 0:
            return
        counts = current_prof_counts(groups[gi])
        # order profs by ascending count, tie-break by availability (desc)
        avail_counts = {p: sum(1 for nm in by_prof[p] if nm in unassigned) for p in VALID_PROF}
        order = sorted(VALID_PROF, key=lambda p: (counts[p], -avail_counts[p]))
        chosen: Optional[str] = None
        for p in order:
            cand = pop_from_prof(p)
            if cand is not None:
                chosen = cand
                break
        if chosen is not None:
            groups[gi].append(chosen)
            unassigned.discard(chosen)

    # Keep filling until capacities are met
    # We honor target sizes by ensuring we don't exceed each group's target size
    target_len = target_sizes[:]  # absolute target count per group
    while unassigned:
        progressed = False
        # iterate groups with remaining space
        for gi in range(num_groups):
            if len(groups[gi]) >= target_len[gi]:
                continue
            if style == "similar":
                before = len(groups[gi])
                fill_group_similar(gi)
                after = len(groups[gi])
            else:
                before = len(groups[gi])
                fill_group_dissimilar(gi)
                after = len(groups[gi])
            if after > before:
                progressed = True
            if not unassigned:
                break
        # If we couldn't make progress (e.g., due to earlier seeding skew), force-place remaining into any group with space
        if not progressed:
            for gi in range(num_groups):
                while len(groups[gi]) < target_len[gi] and unassigned:
                    # pop from any prof bucket
                    any_prof = next((p for p in VALID_PROF if any(nm in unassigned for nm in by_prof[p])), None)
                    fallback = None
                    if any_prof is not None:
                        fallback = pop_from_prof(any_prof)
                    else:
                        # scan unassigned directly
                        fallback = next(iter(unassigned))
                    groups[gi].append(fallback)
                    unassigned.discard(fallback)
            break

    # Build membership map
    membership: Dict[str, int] = {}
    for gi, g in enumerate(groups):
        for nm in g:
            membership[nm] = gi
    return groups, membership

# -----------------------------
# Reporting helpers
# -----------------------------

def save_proficiency_distribution(
    students: List[Student],
    out_stem: str
):
    df = pd.DataFrame({"proficiency": [s.proficiency for s in students]})
    counts = df["proficiency"].value_counts().reindex(VALID_PROF, fill_value=0)
    counts_df = counts.rename_axis("proficiency").reset_index(name="count")
    counts_df.to_csv(f"{out_stem}__proficiency_distribution.tsv", sep="\t", index=False)

    plt.figure()
    sns.barplot(data=counts_df, x="proficiency", y="count", order=VALID_PROF)
    plt.title("Overall Proficiency Distribution")
    plt.xlabel("Proficiency")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{out_stem}__proficiency_distribution.png", dpi=600)
    plt.close()

def group_combo_label(group_members: List[str], name_to_student: Dict[str, Student]) -> str:
    counts = collections.Counter(name_to_student[nm].proficiency for nm in group_members)
    # order by VALID_PROF and format like "2 basic + 1 none"
    parts = []
    for p in VALID_PROF:
        if counts[p] > 0:
            parts.append(f"{counts[p]} {p}")
    return " + ".join(parts) if parts else "empty"

def save_group_combo_plot(
    groups: List[List[str]],
    students: List[Student],
    out_stem: str
):
    name_to_student = {s.name: s for s in students}
    labels = [group_combo_label(g, name_to_student) for g in groups]
    freq = collections.Counter(labels)
    combos_df = pd.DataFrame({"combo": list(freq.keys()), "count": list(freq.values())}).sort_values("combo")
    combos_df.to_csv(f"{out_stem}__group_proficiency_combos.tsv", sep="\t", index=False)

    plt.figure()
    sns.barplot(data=combos_df, x="combo", y="count")
    plt.title("Group Proficiency Combinations")
    plt.xlabel("Combination")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{out_stem}__group_proficiency_combos.png", dpi=600)
    plt.close()

def save_unmet_preferences(
    groups: List[List[str]],
    students: List[Student],
    out_stem: str
):
    name_to_group: Dict[str, int] = {}
    for gi, g in enumerate(groups):
        for nm in g:
            name_to_group[nm] = gi

    rows = []
    for s in students:
        if not s.pref_names:
            continue
        gi = name_to_group[s.name]
        group_set = set(groups[gi])
        unmet = [p for p in s.pref_names if p not in group_set]
        if unmet:
            rows.append({
                "student": s.name,
                "assigned_group_members": "; ".join(sorted([nm for nm in groups[gi] if nm != s.name])),
                "input_preferences": "; ".join(s.pref_names),
                "unmet_preferences": "; ".join(unmet)
            })
    out_df = pd.DataFrame(rows, columns=["student","assigned_group_members","input_preferences","unmet_preferences"])
    out_df.to_csv(f"{out_stem}__unmet_preferences.tsv", sep="\t", index=False)

def save_network_map(
    groups: List[List[str]],
    students: List[Student],
    out_stem: str,
    seed: Optional[int] = None
):
    G = nx.DiGraph()
    # Add nodes with group attribute
    name_to_group: Dict[str, int] = {}
    for gi, g in enumerate(groups):
        for nm in g:
            name_to_group[nm] = gi
            G.add_node(nm, group=gi)

    # Add directed edges for preferences
    name_set = set(name_to_group.keys())
    for s in students:
        for pref in s.pref_names:
            if s.name in name_set and pref in name_set:
                G.add_edge(s.name, pref)

    # Layout; seed for reproducibility
    rng = np.random.default_rng(seed)
    pos = nx.spring_layout(G, seed=rng.integers(0, 1_000_000))

    # Color nodes by group
    groups_order = list(range(len(groups)))
    # Assign a color per group using matplotlib default cycle
    # We'll map group index -> integer to pass to draw
    node_colors = [name_to_group[nm] for nm in G.nodes()]
    # Draw
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=400, node_color=node_colors, cmap=plt.cm.tab20)
    nx.draw_networkx_labels(G, pos, font_size=8)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=10, width=0.8, alpha=0.6)
    plt.axis("off")
    plt.title("Student Preference Network (colored by assigned group)")
    plt.tight_layout()
    plt.savefig(f"{out_stem}__network_map.png", dpi=600)
    plt.close()

def save_groups_tsv(groups: List[List[str]], students: List[Student], out_stem: str):
    """Not requested, but often useful for review."""
    name_to_prof = {s.name: s.proficiency for s in students}
    rows = []
    for gi, g in enumerate(groups):
        for nm in g:
            rows.append({"group_index": gi, "name": nm, "proficiency": name_to_prof[nm]})
    pd.DataFrame(rows).to_csv(f"{out_stem}__assigned_groups.tsv", sep="\t", index=False)

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Group students by preferences and proficiency.")
    parser.add_argument("input_tsv", help="Path to input TSV with name, proficiency, preferences")
    parser.add_argument("--group-size", type=int, required=True, help="Ideal group size (>=2 recommended)")
    parser.add_argument("--style", choices=["similar", "dissimilar"], required=True,
                        help="How to fill groups by proficiency after respecting preferences.")
    parser.add_argument("--out-stem", type=str, default=None,
                        help="Optional output filename stem. Defaults to input stem.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    if args.group_size <= 0:
        print("ERROR: --group-size must be > 0", file=sys.stderr)
        sys.exit(2)

    in_stem = os.path.splitext(os.path.basename(args.input_tsv))[0]
    out_stem = args.out_stem or in_stem

    rng = np.random.default_rng(args.seed)

    # Load
    students = load_input(args.input_tsv)

    # Group
    groups, membership = group_students(students, args.group_size, args.style, rng=rng)

    # Outputs
    save_proficiency_distribution(students, out_stem)
    save_group_combo_plot(groups, students, out_stem)
    save_unmet_preferences(groups, students, out_stem)
    save_network_map(groups, students, out_stem, seed=args.seed)
    save_groups_tsv(groups, students, out_stem)  # helper

    # Nice summary to stdout
    print(f"Created {len(groups)} groups. Output stem: {out_stem}")
    for i, g in enumerate(groups):
        print(f"Group {i} ({len(g)}): {', '.join(g)}")

if __name__ == "__main__":
    main()
