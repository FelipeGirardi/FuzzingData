import sys
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import squarify

COVERAGE_DATA_FILE = sys.argv[1]

def plotGraph(values, label, ax, fileName):
    ax.axis('off')
    ax.set_title(label, fontsize='8')
    squarify.plot(sizes=values, alpha=0.8, edgecolor="white", value=values, ax=ax, color=['g', 'r'], text_kwargs={'fontsize':8})
    plt.savefig(fileName)
    
# Get edge data
df = pd.read_csv(COVERAGE_DATA_FILE, engine='python')
df.columns = df.columns.str.replace('# ', '')
df = df[df['edges_total'] != 0]
edges_covered = df['edges_covered'].values
edges_covered_copy = edges_covered[edges_covered != 0]
edges_total = df['edges_total'].values
edges_not_covered = edges_total - edges_covered
labels = df['function'].values
file_name = df['file'].iloc[0]

# Treemap: edges covered in green, edges not covered in red
fig, axs = plt.subplots(3, 3)
fig.suptitle('Edges covered and not covered for ' + file_name)

for index, value in enumerate(edges_covered):
    v_not_covered = edges_not_covered[index]
    value_to_add = [value] if (v_not_covered == 0) else [value, v_not_covered]
    plotGraph(value_to_add, labels[index], axs[index // 3][index % 3], 'edge_coverage_treemap')

# # Treemap: percentage of edges covered
# percent_edges_covered = (edges_covered_copy/edges_total)*100
# plotGraph(percent_edges_covered, labels, 'perc_edges_covered_treemap', 'Percentage of edges covered for ' + file_name)

# # Treemap: percentage of edges not covered
# percent_edges_not_covered = 100-percent_edges_covered
# percent_edges_not_covered = percent_edges_not_covered[percent_edges_not_covered != 0]
# plotGraph(percent_edges_not_covered, labels, 'perc_edges_not_covered_treemap', 'Percentage of edges not covered for ' + file_name)

plt.show()