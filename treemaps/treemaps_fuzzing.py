import sys
import math
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import squarify
import plotly.express as px

COVERAGE_DATA_FILE_1 = sys.argv[1]
COVERAGE_DATA_FILE_2 = sys.argv[2]

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

# Get edge data for main file
df = pd.read_csv(COVERAGE_DATA_FILE_1, engine='python')
df.columns = df.columns.str.replace('# ', '')
df = df[df['edges_total'] != 0]
df['edges_not_covered'] = df['edges_total'] - df['edges_covered']

edges_not_covered = df['edges_not_covered'].values

# Treemap 1: edges covered in green, edges not covered in red (plotly)
df_covered = df.copy()
df_covered = df_covered.drop('edges_not_covered', axis=1)
df_covered = df_covered.rename({'edges_covered': 'edges'}, axis=1)
df_not_covered = df_covered.copy()

df_covered['ids'] = df_covered['edges'].apply(lambda x: str(x) + ' ')
df_covered['covered'] = ['Covered'] * len(df_covered)
df_not_covered['edges'] = edges_not_covered
df_not_covered['ids'] = df_not_covered['edges'].apply(lambda x: str(x) + '  ')
df_not_covered['covered'] = ['Not_covered'] * len(df_not_covered)
df_all = pd.concat([df_covered, df_not_covered])

fig = px.treemap(df_all, path=['file', 'function', 'ids'], values='edges', color='covered', color_discrete_map={'(?)': 'lightgrey', 'Covered': 'green', 'Not_covered': 'red'})
fig.update_traces(root_color="lightgrey")
fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
fig.show()

# Treemap 2: percentage of edges covered (squarify)
no_covered_edges_indexes = []
files = df['file'].unique()
df_new = df.groupby('file')

for f in files:
    df_file = df_new.get_group(f)
    edges_covered = df_file['edges_covered'].values
    edges_total = df_file['edges_total'].values
    edges_not_covered = df_file['edges_not_covered'].values
    labels = df_file['function'].values
    
    for index, value_covered in enumerate(edges_covered):
        if value_covered == 0:
            no_covered_edges_indexes.append(index)

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
    plt.title('Percentage of edges covered for ' + f)
    plt.axis('off')
    percent_edges_covered_rounded = list(filter(lambda x: x>0, percent_edges_covered_rounded))
    plotPercentageGraph(percent_edges_covered_rounded, labels_copy, 'perc_edges_covered_treemap')

# Treemap 3: differential edges covered between 2 files
df2 = pd.read_csv(COVERAGE_DATA_FILE_2, engine='python')
df2.columns = df2.columns.str.replace('# ', '')
df2 = df2[df2['edges_total'] != 0]

covered_edges_diffs = []
covered_edges_diffs_pos_neg = []
covered_edges_remainders = []

for i in df.index:
    diffValue = df['edges_covered'][i] - df2['edges_covered'][i]
    if diffValue > 0:
        covered_edges_diffs_pos_neg.append('DiffPos')
    elif diffValue < 0:
        covered_edges_diffs_pos_neg.append('DiffNeg')
    else:
        covered_edges_diffs_pos_neg.append('DiffEq')
    diff = abs(diffValue)
    remainder = df['edges_total'][i] - diff
    covered_edges_diffs.append(diff)
    covered_edges_remainders.append(remainder)
    
df_diffs = df2.copy()
df_diffs['diffs'] = pd.Series(covered_edges_diffs)
covered_edges_diffs_str = map(lambda x: str(x) + ' ', covered_edges_diffs)
df_diffs['ids'] = pd.Series(covered_edges_diffs_str)
df_diffs['isDiffOrRemainder'] = pd.Series(covered_edges_diffs_pos_neg)
df_remainders = df2.copy()
df_remainders['diffs'] = pd.Series(covered_edges_remainders)
covered_edges_remainders_str = map(lambda x: str(x) + '  ', covered_edges_remainders)
df_remainders['ids'] = pd.Series(covered_edges_remainders_str)
df_remainders['isDiffOrRemainder'] = ['Remainder'] * len(df_remainders)
df_plot = pd.concat([df_diffs, df_remainders], ignore_index=True)
df_plot = df_plot.drop('edges_covered', axis=1)
df_plot = df_plot.drop('edges_total', axis=1)

fig2 = px.treemap(df_plot, path=['file', 'function', 'ids'], values='diffs', color='isDiffOrRemainder', color_discrete_map={'(?)': 'lightgrey', 'DiffPos': 'green', 'DiffNeg': 'red', 'DiffEq': 'lightgrey', 'Remainder': 'gray'})
fig2.update_traces(root_color="lightgrey")
fig2.update_layout(margin = dict(t=50, l=25, r=25, b=25))
fig2.show()

plt.show()