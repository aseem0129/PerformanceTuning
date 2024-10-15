# optimization.py
import pandas as pd
import numpy as np
import time
import dask.dataframe as dd

np.random.seed(42)
data = {
    'A': np.random.choice(['group1', 'group2', 'group3'], 10000000),  # 10 million rows
    'B': np.random.choice(['subgroup1', 'subgroup2'], 10000000),
    'C': np.random.randn(10000000),
    'D': np.random.randn(10000000),
}

df = pd.DataFrame(data)

# Convert 'A' and 'B' to categorical data type for optimization
df['A'] = df['A'].astype('category')
df['B'] = df['B'].astype('category')

# Capturing execution times for all functions

# Original groupby
start = time.time()
df.groupby(['A', 'B']).sum()
end = time.time()
groupby_time = end - start

# Original pivot_table
start = time.time()
df.pivot_table(index='A', columns='B', values='C', aggfunc='sum')
end = time.time()
pivot_time = end - start

# Optimized groupby with as_index=False
start = time.time()
df.groupby(['A', 'B'], as_index=False).sum()
end = time.time()
optimized_groupby_time = end - start

# Optimized pivot_table with fill_value=0
start = time.time()
df.pivot_table(index='A', columns='B', values='C', aggfunc='sum', fill_value=0)
end = time.time()
optimized_pivot_time = end - start

ddf = dd.from_pandas(df, npartitions=8)  # Dividing the dataset into 8 partitions for parallel processing


start = time.time()
ddf.groupby(['A', 'B']).sum().compute()  # compute() triggers the parallel computation
end = time.time()
dask_groupby_time = end - start

# Pivot Table after Dask
start = time.time()
df_pivot = df.pivot_table(index='A', columns='B', values='C', aggfunc='sum', fill_value=0)
end = time.time()
dask_pivot_time = end - start

# groupby + unstack as an alternative to pivot_table
start = time.time()
result = df.groupby(['A', 'B'])['C'].sum().unstack(fill_value=0)
end = time.time()
unstack_pivot_time = end - start


timing_data = {
    'Function': [
        'Groupby (Original)',
        'Pivot Table (Original)',
        'Groupby (Optimized)',
        'Pivot Table (Optimized)',
        'Groupby (Dask)',
        'Pivot Table (Dask)',
        'Pivot (groupby + unstack)'
    ],
    'Execution Time (seconds)': [
        groupby_time,         # Original groupby time
        pivot_time,           # Original pivot_table time
        optimized_groupby_time,  # Optimized groupby time
        optimized_pivot_time,    # Optimized pivot_table time
        dask_groupby_time,    # Dask groupby time
        dask_pivot_time,      # Pivot after Dask time
        unstack_pivot_time    # groupby + unstack time
    ]
}

# Convert timing data to a pandas DataFrame
df_timing = pd.DataFrame(timing_data)

# Save the DataFrame as a CSV file
df_timing.to_csv('timing_results.csv', index=False)

print("Timing data saved to timing_results.csv")
