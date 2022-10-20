import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
# from scipy.interpolate import make_interp_spline

INTERVAL_SECONDS = int(sys.argv[1])

# Function to find the number closest to n which is divisible by m
def closestNumber(n, m):
    q = int(n / m)
    n1 = m * q
    n2 = (m * (q + 1))
     
    if (abs(n - n1) <= abs(n - n2)) :
        return n1
    return n2

def average(x, y):
    return int(round((x+y)/2))

def mean(lst):
    return int(round(sum(lst) / len(lst)))

values_list = []
edge_values_dict = {}

# Loop each fuzzer CSV file to get edges_found and total_execs values in given time interval
for i in range(1,9):
    df = pd.read_csv('fuzzer0' + str(i), sep='\,\ ', engine='python')
    df.columns = df.columns.str.replace('# ', '')
    
    new_df = df[['relative_time', 'edges_found']].copy()
    n_rows_range = range(len(new_df.index))
    prev_time_value = -1
    
    for i in n_rows_range:
        time_value = int(new_df['relative_time'].iloc[i])
        rounded_time_value = closestNumber(time_value, INTERVAL_SECONDS)
        rounded_time_value_str = str(rounded_time_value)
        edge_value = new_df['edges_found'].iloc[i]
        edge_dict_keys = list(edge_values_dict.keys())

        if prev_time_value >= 0:
            if rounded_time_value - prev_time_value > INTERVAL_SECONDS:
                mid_time_value = average(rounded_time_value, prev_time_value)
                mid_time_value_str = str(mid_time_value)
                if mid_time_value_str not in edge_dict_keys:
                    edge_values_dict[mid_time_value_str] = []
                edge_values_dict[mid_time_value_str] += [average(prev_edge_value, edge_value)]
            elif rounded_time_value == prev_time_value:
                continue
        
        if rounded_time_value_str not in edge_dict_keys:
            edge_values_dict[rounded_time_value_str] = []
        
        edge_values_dict[rounded_time_value_str] += [edge_value]
        prev_time_value = rounded_time_value
        prev_edge_value = edge_value

graph_time_values = edge_dict_keys
graph_edge_mean_values = []
graph_edge_min_values = []
graph_edge_max_values = []

for key in edge_values_dict:
    edge_val_list = edge_values_dict[key]
    graph_edge_mean_values.append(mean(edge_val_list))
    graph_edge_min_values.append(min(edge_val_list))
    graph_edge_max_values.append(max(edge_val_list))

# -------------------------
  
# Plotting code (for later)

# final_result_df = pd.concat(result_df, ignore_index=True)
# final_result_df = final_result_df.groupby('relative_time').mean()

# relative_time_list = final_result_df.index.values
# edges_found_list = final_result_df['edges_found']
# x_y_spline = make_interp_spline(relative_time_list, edges_found_list)
# x_ = np.linspace(relative_time_list.min(), relative_time_list.max(), 500)
# y_ = x_y_spline(x_)

# plt.plot(x_, y_)

# xmax = max(final_result_df['relative_time'])
# ymax = max(final_result_df['edges_found'])
# plt.ylim(0, ymax + 100)
# plt.xlim(0, xmax + 100)
# sns.lineplot(data=final_result_df, x="relative_time", y="edges_found", errorbar=('ci', 100))

# plt.show()

# df = final_result_df.groupby('relative_time').mean()
# relative_time_list = final_result_df.index.values
# edges_found_list = final_result_df['edges_found']
#     # edges_found_list_mean = df['edges_found'].mean()

# plt.figure(1)
# xmin, xmax = min(relative_time_list), max(relative_time_list)
# ymin, ymax = min(edges_found_list), max(edges_found_list)
# plt.ylim(ymin, ymax)
# plt.xlim(xmin, xmax)
# plt.xlabel('Time (s)')
# plt.ylabel('Edges found')
# plt.plot(relative_time_list, edges_found_list, linewidth = 2.0, color='darkorchid')
    
# total_execs plot
# total_execs_list = df['total_execs']
# plt.figure(2)
# ymin, ymax = min(total_execs_list), max(total_execs_list)
# plt.ylim(ymin, ymax)
# plt.xlim(xmin, xmax)
# plt.plot(relative_time_list, total_execs_list, linewidth = 2.0, color=colors[i-1])
# plt.xlabel('Time (s)')
# plt.ylabel('Total executions')