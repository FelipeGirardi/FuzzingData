import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import squarify

# Get edge data
df = pd.read_csv('coverage_data.csv', engine='python')
df.columns = df.columns.str.replace('# ', '')
df = df[df['edges_total'] != 0]
edges_covered = df['edges_covered']
edges_covered_copy = edges_covered[edges_covered != 0]
edges_total = df['edges_total']
labels = df['function']
file_name = df['file'].iloc[0]

sns.set_style(style="whitegrid")
cmap = matplotlib.cm.Blues

# Treemap 1: total number of edges (relative)
min_val = min(edges_total)
max_val = max(edges_total)
norm = matplotlib.colors.Normalize(vmin=min_val, vmax=max_val)
colors = [cmap(norm(value)) for value in edges_total]
plt.figure()
plt.axis('off')
plt.savefig('total_edges_treemap')
squarify.plot(sizes=edges_total, alpha=0.7, edgecolor="white", label=labels, color=colors, text_kwargs={'fontsize':8}).set(title='Total edges for ' + file_name)

# Treemap 2: number of edges covered (relative)
min_val = min(edges_covered_copy)
max_val = max(edges_covered_copy)
norm = matplotlib.colors.Normalize(vmin=min_val, vmax=max_val)
colors = [cmap(norm(value)) for value in edges_covered_copy]
plt.figure()
plt.axis('off')
plt.savefig('edges_covered_treemap')
squarify.plot(sizes=edges_covered_copy, alpha=0.7, label=labels, color=colors, text_kwargs={'fontsize':8}).set(title='Edges covered for ' + file_name)

# Treemap 3: number of edges not covered
edges_not_covered = edges_total - edges_covered
edges_not_covered = edges_not_covered[edges_not_covered != 0]
min_val = min(edges_not_covered)
max_val = max(edges_not_covered)
norm = matplotlib.colors.Normalize(vmin=min_val, vmax=max_val)
colors = [cmap(norm(value)) for value in edges_not_covered]
plt.figure()
plt.axis('off')
plt.savefig('edges_not_covered_treemap')
squarify.plot(sizes=edges_not_covered, alpha=0.7, label=labels, color=colors, text_kwargs={'fontsize':8}).set(title='Edges covered for ' + file_name)

# Treemap 4: percentage of edges covered
percent_edges_covered = (edges_covered_copy/edges_total)*100
min_val = min(percent_edges_covered)
max_val = max(percent_edges_covered)
norm = matplotlib.colors.Normalize(vmin=min_val, vmax=max_val)
colors = [cmap(norm(value)) for value in percent_edges_covered]
plt.figure()
plt.axis('off')
plt.savefig('perc_edges_covered_treemap')
squarify.plot(sizes=percent_edges_covered, alpha=0.7, label=labels, color=colors, text_kwargs={'fontsize':8}).set(title='Percentage of edges covered for ' + file_name)

# Treemap 5: percentage of edges not covered
percent_edges_not_covered = 100-percent_edges_covered
percent_edges_not_covered = percent_edges_not_covered[percent_edges_not_covered != 0]
min_val = min(percent_edges_not_covered)
max_val = max(percent_edges_not_covered)
norm = matplotlib.colors.Normalize(vmin=min_val, vmax=max_val)
colors = [cmap(norm(value)) for value in percent_edges_not_covered]
plt.figure()
plt.axis('off')
plt.savefig('perc_edges_not_covered_treemap')
squarify.plot(sizes=percent_edges_not_covered, alpha=0.7, label=labels, color=colors, text_kwargs={'fontsize':8}).set(title='Percentage of edges not covered for ' + file_name)

plt.show()