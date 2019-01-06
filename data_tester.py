import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('data/processed_genomic_data_set_v2.csv')

cols = list(df.columns)

print(df.head(5))
print(df.dtypes)
print(cols)
print(len(cols))
print(df.nunique())


