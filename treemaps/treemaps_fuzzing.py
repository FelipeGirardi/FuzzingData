import sys
import math
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import squarify

COVERAGE_DATA_FILE = sys.argv[1]

def plotGraph(values, label, ax, fileName):
    ax.set_title(label, fontsize='8')
    is_value_covered_0 = values[0] == 0
    is_value_not_covered_0 = values[1] == 0
    values = [values[1]] if is_value_covered_0 else [values[0]] if is_value_not_covered_0 else values
    colors = ['r'] if is_value_covered_0 else ['g'] if is_value_not_covered_0 else ['g', 'r']
    # plt.subplots_adjust(right=0.9)
    squarify.plot(sizes=values, alpha=0.8, edgecolor="white", value=values, ax=ax, color=colors, text_kwargs={'fontsize':8})
    # plt.savefig(fileName)

# Get edge data
df = pd.read_csv(COVERAGE_DATA_FILE, engine='python')
df.columns = df.columns.str.replace('# ', '')
df = df[df['edges_total'] != 0]
df['edges_not_covered'] = df['edges_total'] - df['edges_covered']
df['percent_covered'] = df['edges_covered']/df['edges_total']
df['percent_not_covered'] = df['edges_not_covered']/df['edges_total']
df = df.sort_values('percent_not_covered', ascending=False)

edges_covered = df['edges_covered'].values
edges_covered_copy = edges_covered[edges_covered != 0]
edges_total = df['edges_total'].values
edges_not_covered = df['edges_not_covered'].values
labels = df['function'].values
file_name = df['file'].iloc[0]
n_rows_df = len(df)

# Treemap: edges covered in green, edges not covered in red
n_rows_plot = round(math.sqrt(n_rows_df))
n_columns_plot = math.ceil(math.sqrt(n_rows_df))
fig, axs = plt.subplots(n_rows_plot, n_columns_plot, constrained_layout=True)
# fig.tight_layout()

for ax1 in axs:
    for ax2 in ax1:
        ax2.axis('off')
fig.suptitle('Edges covered and not covered for ' + file_name)

for index, value_covered in enumerate(edges_covered):
    value_not_covered = edges_not_covered[index]
    values = [value_covered, value_not_covered]
    plotGraph(values, labels[index], axs[index // n_columns_plot][index % n_columns_plot], 'edge_coverage_treemap')

# # Treemap: percentage of edges covered
# percent_edges_covered = (edges_covered_copy/edges_total)*100
# plotGraph(percent_edges_covered, labels, 'perc_edges_covered_treemap', 'Percentage of edges covered for ' + file_name)

# # Treemap: percentage of edges not covered
# percent_edges_not_covered = 100-percent_edges_covered
# percent_edges_not_covered = percent_edges_not_covered[percent_edges_not_covered != 0]
# plotGraph(percent_edges_not_covered, labels, 'perc_edges_not_covered_treemap', 'Percentage of edges not covered for ' + file_name)

plt.show()