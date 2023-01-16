import sys
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import squarify

COVERAGE_DATA_FILE = sys.argv[1]

def plotGraph(values, labels, fileName, graphTitle):
    min_val = min(values)
    max_val = max(values)
    cmap = matplotlib.cm.Blues
    norm = matplotlib.colors.Normalize(vmin=min_val, vmax=max_val)
    colors = [cmap(norm(value)) for value in values]
    plt.figure()
    plt.axis('off')
    sns.set_style(style="whitegrid")
    squarify.plot(sizes=values, alpha=0.8, edgecolor="white", label=labels, color=colors, text_kwargs={'fontsize':8}).set(title=graphTitle)
    plt.savefig(fileName)
    
# Get edge data
df = pd.read_csv(COVERAGE_DATA_FILE, engine='python')
df.columns = df.columns.str.replace('# ', '')
df = df[df['edges_total'] != 0]
edges_covered = df['edges_covered']
edges_covered_copy = edges_covered[edges_covered != 0]
edges_total = df['edges_total']
labels = df['function']
file_name = df['file'].iloc[0]

# Treemap 1: total number of edges (relative)
plotGraph(edges_total, labels, 'total_edges_treemap', 'Total edges for ' + file_name)

# Treemap 2: number of edges covered (relative)
plotGraph(edges_covered_copy, labels, 'edges_covered_treemap', 'Edges covered for ' + file_name)

# Treemap 3: number of edges not covered
edges_not_covered = edges_total - edges_covered
edges_not_covered = edges_not_covered[edges_not_covered != 0]
plotGraph(edges_not_covered, labels, 'edges_not_covered_treemap', 'Edges not covered for ' + file_name)

# Treemap 4: percentage of edges covered
percent_edges_covered = (edges_covered_copy/edges_total)*100
plotGraph(percent_edges_covered, labels, 'perc_edges_covered_treemap', 'Percentage of edges covered for ' + file_name)

# Treemap 5: percentage of edges not covered
percent_edges_not_covered = 100-percent_edges_covered
percent_edges_not_covered = percent_edges_not_covered[percent_edges_not_covered != 0]
plotGraph(percent_edges_not_covered, labels, 'perc_edges_not_covered_treemap', 'Percentage of edges not covered for ' + file_name)

plt.show()