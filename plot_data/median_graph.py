import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#808080']

for i in range(1,9):
    df = pd.read_csv('fuzzer0' + str(i), sep='\,\ ', engine='python')
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

    plt.figure(1)
    plt.plot(x, y, color=colors[i-1]) 
    plt.plot(x, y_low, color=colors[i-1], linestyle='-.') 
    plt.plot(x, y_high, color=colors[i-1], linestyle='-.')

    plt.fill_between(x, y_low, y_high, alpha =.1, color = colors[i-1])

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