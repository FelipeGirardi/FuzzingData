import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import warnings
# warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
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
    return round((x+y)/2)

# Loop each fuzzer CSV file to get edges_found and total_execs values in given time interval
for i in range(1,9):
    df = pd.read_csv('fuzzer0' + str(i), sep='\,\ ', engine='python')
    df.columns = df.columns.str.replace('# ', '')
    
    new_df = df[['relative_time', 'edges_found']].copy()
    n_rows_range = range(len(new_df.index))
    graph_values_list = []
    add_value_counter = 0
    
    for i in n_rows_range:
        time_value = int(new_df['relative_time'].iloc[i])
        rounded_time_value = closestNumber(time_value, INTERVAL_SECONDS)
        edge_value = new_df['edges_found'].iloc[i]
        
        if graph_values_list:
            if(rounded_time_value - graph_values_list[i-1+add_value_counter][0] > INTERVAL_SECONDS):
                graph_values_list.append([rounded_time_value - INTERVAL_SECONDS, average(graph_values_list[i-1+add_value_counter][1], edge_value)])
                add_value_counter += 1
        
        graph_values_list.append([rounded_time_value, edge_value])
        
    

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