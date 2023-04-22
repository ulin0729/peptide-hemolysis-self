# Peptide sequence hemolysis classifier

## Introduction
The purpose of this system is to predict if a peptide sequence will cause hemolysis under certain concentration.

It comprises 9 different classifiers with different thresholds.

Each classifier is an ensembled model of several different machine learning model. (see below)

The system is trained on DBAASP dataset.

## Usage
`python3 api.py [-h] -f FASTA_NAME -o OUTPUT_CSV -t THRESHOLD`

## Limitations
The threshold can be one of the following options {10, 20, 30, 40, 50, 60, 70, 80, 90}
Warning: Classifiers with threshold over 50 are trained on insufficient dataset. The result might be biased.

The sequences in the input file should be in the following fasta format:
`>ID|Conc={concentration}
XXX......XXX`

The concentration is limited to (0,300] ug/ml. If the input concentration is not in this range, the maximum 300 ug/ml will be used instead.

