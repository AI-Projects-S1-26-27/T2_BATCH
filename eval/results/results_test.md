# Results on the test split (42 headline pairs)

F1 per dimension (precision / recall in brackets):

| Dimension | OURS | B1 | B2 | B3 | ABLATION |
|---|---|---|---|---|---|
| Information selection | 0.92 (1.00 / 0.85) | 0.96 (0.93 / 1.00) | 0.96 (0.93 / 1.00) | 0.97 (0.95 / 1.00) | 0.92 (1.00 / 0.85) |
| Actor/action framing | 0.35 (0.50 / 0.27) | 0.00 (0.00 / 0.00) | 0.00 (0.00 / 0.00) | 0.56 (0.39 / 1.00) | 0.48 (0.32 / 0.91) |
| Evaluative wording | 0.24 (0.80 / 0.14) | 0.30 (1.00 / 0.18) | 0.30 (1.00 / 0.18) | 0.95 (0.93 / 0.96) | 0.39 (0.88 / 0.25) |
| Consequence emphasis | 0.00 (0.00 / 0.00) | 0.00 (0.00 / 0.00) | 0.00 (0.00 / 0.00) | 0.33 (0.20 / 1.00) | 0.10 (0.06 / 0.33) |
| **Micro F1** | 0.61 | 0.69 | 0.69 | 0.82 | 0.60 |
| **Macro F1** | 0.38 | 0.32 | 0.32 | 0.70 | 0.47 |
| **Evidence overlap with gold words (TPs)** | 98% | 100% | 96% | 89% | 96% |
| **Differences output** | 110 | 238 | 192 | 136 | 143 |

Gold positives per dimension: Information selection 39, Actor/action framing 11, Evaluative wording 28, Consequence emphasis 3.
B1/B2 can only output D1/D3 (see baselines/common.py), so their D2/D4 scores are 0 by design.
B3 citations found verbatim in the headline: 96%.
