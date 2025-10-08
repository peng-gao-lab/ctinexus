## Introduction

This data repository contains CTI reports collected from well-known CTI sharing platforms. It includes the following components:

- [annotation](annotation): A carefully annotated dataset that serves as the ground truth for triplet extraction, entity alignment, and link prediction tasks.
- [test](test): A test set specifically designed to evaluate the performance of CTINexus.
- [demo](demo): A demonstration set curated to select the optimal examples for in-context learning (ICL).

## Resampling the Test Set

To resample the test set or adjust its size, you can use the script [data_split.py](data_split.py) located in this repository.
