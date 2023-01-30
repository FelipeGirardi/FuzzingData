import sys
import math
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import squarify
import plotly.express as px
import plotly.graph_objects as go

COVERAGE_DATA_FILE = sys.argv[1]

def plotSubgraph(values, label, ax):
    ax.set_title(label, fontsize='8')
    is_value_covered_0 = values[0] == 0
    is_value_not_covered_0 = values[1] == 0
    values = [values[1]] if is_value_covered_0 else [values[0]] if is_value_not_covered_0 else values
    colors = ['r'] if is_value_covered_0 else ['g'] if is_value_not_covered_0 else ['g', 'r']
    squarify.plot(sizes=values, alpha=0.8, edgecolor="white", value=values, ax=ax, color=colors, text_kwargs={'fontsize':8})
    
def plotPercentageGraph(values, labels, fileName):
    min_val = min(values)
    max_val = max(values)
    cmap = matplotlib.cm.Blues
    norm = matplotlib.colors.Normalize(vmin=min_val, vmax=max_val)
    colors = [cmap(norm(value)) for value in values]
    squarify.plot(sizes=values, alpha=0.8, edgecolor="white", label=labels, color=colors, text_kwargs={'fontsize':8})
    plt.savefig(fileName)

# Get edge data
df = pd.read_csv(COVERAGE_DATA_FILE, engine='python')
df.columns = df.columns.str.replace('# ', '')
df = df[df['edges_total'] != 0]
df['edges_not_covered'] = df['edges_total'] - df['edges_covered']
df = df.sort_values('edges_not_covered', ascending=False)

edges_covered = df['edges_covered'].values
edges_total = df['edges_total'].values
edges_not_covered = df['edges_not_covered'].values
labels = df['function'].values
file_name = df['file'].iloc[0]
n_rows_df = len(df)
no_covered_edges_indexes = []

# Treemap 1: edges covered in green, edges not covered in red
n_rows_plot = round(math.sqrt(n_rows_df))
n_columns_plot = math.ceil(math.sqrt(n_rows_df))
n_plots = n_rows_plot * n_columns_plot
height_ratios = []
edges_not_covered_split = np.array_split(edges_not_covered, n_rows_plot)
width_ratios = np.mean(edges_not_covered_split, axis=0)
for arr in edges_not_covered_split:
    height_ratios.append(np.mean(arr))

fig, axs = plt.subplots(n_rows_plot, n_columns_plot, gridspec_kw={'width_ratios': width_ratios, 'height_ratios': height_ratios})
plt.subplots_adjust(hspace=0.5, wspace=0.1)

for ax1 in axs:
    for ax2 in ax1:
        ax2.axis('off')
fig.suptitle('Edges not covered for ' + file_name)

for index, value_covered in enumerate(edges_covered):
    if value_covered == 0:
        no_covered_edges_indexes.append(index)
    value_not_covered = edges_not_covered[index]
    values = [value_covered, value_not_covered]
    plotSubgraph(values, labels[index], axs[index // n_columns_plot][index % n_columns_plot])

plt.savefig('edge_coverage_treemap')

# Plotly treemap
df_covered = df.copy()
df_covered = df_covered.drop('edges_not_covered', axis=1)
df_covered = df_covered.rename({'edges_covered': 'edges'}, axis=1)
df_not_covered = df_covered.copy()
df_covered['covered'] = ['Covered'] * len(df_covered)
df_covered['color'] = ['green'] * len(df_covered)

df_not_covered['edges'] = edges_not_covered
df_not_covered['covered'] = ['Not covered'] * len(df_not_covered)
df_covered['color'] = ['red'] * len(df_not_covered)
df_all = pd.concat([df_covered, df_not_covered])

fig = px.treemap(df_all, path=['file', 'function', 'edges'], values='edges', color='covered', color_discrete_map={'(?)': df_all['color']})
fig.update_traces(root_color="lightgrey")
fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
fig.show()

# Treemap 2: percentage of edges covered
for _, num in enumerate(no_covered_edges_indexes):
    edges_covered_copy = np.delete(edges_covered, num)
    edges_total_copy = np.delete(edges_total, num)
    labels_copy = np.delete(labels, num)

percent_edges_covered_not_rounded = (edges_covered_copy/edges_total_copy)*100
percent_edges_covered_rounded = []
for i, elem in enumerate(percent_edges_covered_not_rounded):
    n_rounded = np.around(elem, 1)
    percent_edges_covered_rounded.append(n_rounded)
    n_rounded_str = '\n' + str(n_rounded) + '%'
    labels_copy[i] += n_rounded_str
plt.figure()
plt.title('Percentage of edges covered for ' + file_name)
plt.axis('off')
plotPercentageGraph(percent_edges_covered_rounded, labels_copy, 'perc_edges_covered_treemap')

# # Treemap: percentage of edges not covered
# percent_edges_not_covered = 100-percent_edges_covered
# percent_edges_not_covered = percent_edges_not_covered[percent_edges_not_covered != 0]
# plotGraph(percent_edges_not_covered, labels, 'perc_edges_not_covered_treemap', 'Percentage of edges not covered for ' + file_name)

# plt.show()