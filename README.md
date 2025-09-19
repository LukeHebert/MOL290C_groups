# Student Grouping (Preferences + Proficiency)

This tool forms student groups from a TSV input by first honoring **partner preferences** where possible, then filling remaining seats by **proficiency** using either:

- `similar` – try to keep proficiency levels alike within each group
- `dissimilar` – try to balance proficiency levels within each group

It also produces:
- A 600 dpi bar plot of overall proficiencies (`…__proficiency_distribution.png`) and its counts (`…__proficiency_distribution.tsv`)
- A 600 dpi bar plot of group proficiency combos (`…__group_proficiency_combos.png`) and their frequencies (`…__group_proficiency_combos.tsv`)
- A TSV of unmet preferences (`…__unmet_preferences.tsv`)
- A 600 dpi **network map** (directed edges = preferences, node color = group)

> You can use `python group_students.py --help` for a list of parameters & their descriptions

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Recommended arguments for MOL290 TA's

If you are also a TA for MOL290 and would like to use this script to make grouping a little easier, this is the set of parameters that served me well. 

```
python3 group_students.py your_class_data.tsv --group-size 3 --style similar --out-stem test_similar-optimal --optimize exact --mutual-weight 2 --time-limit 120
```

- Remember that `your_class_data.tsv` should be organized in the same format as [test.tsv](/demo/test.tsv)

- Though I doubt this will be necessary: If you find that your system takes too long for `--optimize exact` then feel free to try the heuristic approach with `--optimize heuristic` which may not perfectly optimize groups according to student preferences, but will do a decent job and take less time.