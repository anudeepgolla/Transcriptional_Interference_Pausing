#!/usr/bin/env bash

#python3 ../models/param_classification_engine.py  ../data/param/param_ds_looselabel_sample_adj_v1.csv pause_status param_loose
#python3 ../models/param_classification_engine.py  ../data/param/param_ds_prcs_selectivelabel_sample_adj_v1.csv pause_status param_selective
#python3 ../models/param_classification_engine.py  ../data/param/param_ds_strictlabel_sample_adj_v1.csv pause_status param_strict


python3 ../models/param_classification_engine.py  ../data/seq/sequence_ds_onehot_spike_v1.csv energy_spike seq_pred_spike_onehot
python3 ../models/param_classification_engine.py  ../data/seq/sequence_ds_ordinal_spike_v1.csv energy_spike seq_pred_spike_ordinal
python3 ../models/param_classification_engine.py  ../data/seq/sequence_ds_kmers_6_spike_v1.csv energy_spike seq_pred_spike_kmers_6


python3 ../models/param_classification_engine.py  ../data/seq/sequence_ds_onehot_given_spike_loose_v1.csv pause_status seq_given_spike_onehot_loose
python3 ../models/param_classification_engine.py  ../data/seq/sequence_ds_onehot_given_spike_selective_v1.csv pause_status seq_given_spike_onehot_selective
python3 ../models/param_classification_engine.py  ../data/seq/sequence_ds_onehot_given_spike_strict_v1.csv pause_status seq_given_spike_onehot_strict

python3 ../models/param_classification_engine.py  ../data/seq/sequence_ds_ordinal_given_spike_loose_v1.csv pause_status seq_given_spike_ordinal_loose
python3 ../models/param_classification_engine.py  ../data/seq/sequence_ds_ordinal_given_spike_selective_v1.csv pause_status seq_given_spike_ordinal_selective
python3 ../models/param_classification_engine.py  ../data/seq/sequence_ds_ordinal_given_spike_strict_v1.csv pause_status seq_given_spike_ordinal_strict

python3 ../models/param_classification_engine.py  ../data/seq/sequence_ds_kmers_6_given_spike_loose_v1.csv pause_status seq_given_spike_kmers_6_loose
python3 ../models/param_classification_engine.py  ../data/seq/sequence_ds_kmers_6_given_spike_selective_v1.csv pause_status seq_given_spike_kmers_6_selective
python3 ../models/param_classification_engine.py  ../data/seq/sequence_ds_kmers_6_given_spike_strict_v1.csv pause_status seq_given_spike_kmers_6_strict


python3 ../models/param_classification_engine.py  ../data/seq/sequence_ds_onehot_no_spike_loose_v1.csv pause_status seq_no_spike_onehot_loose
python3 ../models/param_classification_engine.py  ../data/seq/sequence_ds_onehot_no_spike_selective_v1.csv pause_status seq_no_spike_onehot_selective
python3 ../models/param_classification_engine.py  ../data/seq/sequence_ds_onehot_no_spike_strict_v1.csv pause_status seq_no_spike_onehot_strict

python3 ../models/param_classification_engine.py  ../data/seq/sequence_ds_ordinal_no_spike_loose_v1.csv pause_status seq_no_spike_ordinal_loose
python3 ../models/param_classification_engine.py  ../data/seq/sequence_ds_ordinal_no_spike_selective_v1.csv pause_status seq_no_spike_ordinal_selective
python3 ../models/param_classification_engine.py  ../data/seq/sequence_ds_ordinal_no_spike_strict_v1.csv pause_status seq_no_spike_ordinal_strict

python3 ../models/param_classification_engine.py  ../data/seq/sequence_ds_kmers_6_no_spike_loose_v1.csv pause_status seq_no_spike_kmers_6_loose
python3 ../models/param_classification_engine.py  ../data/seq/sequence_ds_kmers_6_no_spike_selective_v1.csv pause_status seq_no_spike_kmers_6_selective
python3 ../models/param_classification_engine.py  ../data/seq/sequence_ds_kmers_6_no_spike_strict_v1.csv pause_status seq_no_spike_kmers_6_strict









