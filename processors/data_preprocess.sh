#!/bin/bash

# bash execution of normalization file and store System.out results into data_preprocess.out
python3 -u data_normalizer.py &> data_preprocess.out
