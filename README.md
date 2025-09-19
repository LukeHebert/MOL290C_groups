# Student Grouping (Preferences + Proficiency)

This tool forms student groups from a TSV input by first honoring **partner preferences** where possible, then filling remaining seats by **proficiency** using either:

- `similar` – try to keep proficiency levels alike within each group
- `dissimilar` – try to balance proficiency levels within each group

It also produces:
- A 600 dpi bar plot of overall proficiencies (`…__proficiency_distribution.png`) and its counts (`…__proficiency_distribution.tsv`)
- A 600 dpi bar plot of group proficiency combos (`…__group_proficiency_combos.png`) and their frequencies (`…__group_proficiency_combos.tsv`)
- A TSV of unmet preferences (`…__unmet_preferences.tsv`)
- A 600 dpi **network map** (directed edges = preferences, node color = group)

> All output filenames use the **input file’s stem** unless you pass `--out-stem`.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Recommended arguments for MOL290 TA's
```
python3 group_students.py your_class_data.tsv --group-size 3 --style similar --out-stem test_similar-optimal --optimize exact --mutual-weight 2 --time-limit 120
```
Though the time limit is probably unnecessary.
`your_class_data.tsv` should be organized in the same format as [test.tsv](/demo/test.tsv)