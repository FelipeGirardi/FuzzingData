import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('fuzzer01', sep='\,\ ', engine='python')
df.columns = df.columns.str.replace('# ', '')

time_group = df.groupby('relative_time')
mean_edges = time_group.mean().rename(columns={"edges_found": "mean_edges_found"})
edges_25per = time_group.quantile(q=.25).rename(columns={"edges_found": "edges_found_25per"})
edges_75per = time_group.quantile(q=.75).rename(columns={"edges_found": "edges_found_75per"})

dframes = [mean_edges, edges_25per, edges_75per]
merged_df = pd.concat(dframes, join='outer', axis=1)

x = merged_df.index
y = merged_df.mean_edges_found
y_low = merged_df.edges_found_25per
y_high = merged_df.edges_found_75per

plt.plot(x, y, color='darkorchid', label='mean') 
plt.plot(x, y_low, color='darkorchid', linestyle='-.', label='25th %') 
plt.plot(x, y_high, color='darkorchid', linestyle='-.', label='75th %') 
plt.legend()

plt.fill_between(x, y_low, y_high, alpha =.1, color = 'darkorchid')

# plt.grid(alpha=.2, which='both')
plt.xlabel('Time (s)')
plt.ylabel('Edges found')
plt.show()

# df_stats = df.groupby(['relative_time']).describe()
# x = df_stats.index
# medians = df_stats[('edges_found', '50%')]
# quartiles1 = df_stats[('edges_found', '25%')]
# quartiles3 = df_stats[('edges_found', '75%')]

# fig, ax = plt.subplots()
# ax.plot(x, medians) 
# ax.fill_between(x, quartiles1, quartiles3, alpha=0.3); 

# plt.show()