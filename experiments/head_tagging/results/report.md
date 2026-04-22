# Head-tagging validation — induction-copy task

Baseline accuracy: second-half = **99.2%**, overall = **53.2%**.

Each row ablates a single head (gate -> 0) and measures the accuracy drop. Large negative `delta_acc_second_half` means that head was important for induction.

| layer | head | tag | avg_ent | avg_sharp | gate | Δacc 2nd-half | Δacc overall |
|---:|---:|:---|---:|---:|---:|---:|---:|
| 0 | 0 | `sharp_selector` | 1.28 | 0.61 | 0.911 | -0.124 | -0.062 |
| 0 | 1 | `sharp_selector` | 1.54 | 0.54 | 0.907 | -0.024 | -0.012 |
| 0 | 2 | `sharp_selector` | 1.62 | 0.51 | 0.905 | -0.023 | -0.012 |
| 0 | 3 | `balanced` | 1.70 | 0.49 | 0.904 | -0.014 | -0.007 |
| 1 | 3 | `balanced` | 2.40 | 0.29 | 0.888 | -0.003 | -0.002 |
| 1 | 1 | `position_tracker` | 1.58 | 0.54 | 0.892 | -0.001 | -0.001 |
| 1 | 2 | `position_tracker` | 1.79 | 0.47 | 0.892 | -0.001 | -0.000 |
| 1 | 0 | `position_tracker` | 1.39 | 0.59 | 0.891 | -0.000 | +0.000 |
