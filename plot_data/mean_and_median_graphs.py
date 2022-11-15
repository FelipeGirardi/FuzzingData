import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics

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

def evaluate(a, x, t, edge_values, time_values):
    edges_covered_mult_by_percent = a * x
    target_time = (1-a) * t
    time_found = float("inf")
    for i, elem in enumerate(edge_values):
        if elem >= edges_covered_mult_by_percent:
            edge_found = elem
            time_found = time_values[i]
            break

    if time_found <= target_time:
        return [edge_found, time_found]
    else:
        return [-1, -1]


values_list = []
edge_values_dict = {}
exec_values_dict = {}

# Loop each fuzzer CSV file to get edges_found and total_execs values in given time interval
for i in range(1,9):
    df = pd.read_csv('fuzzer0' + str(i), sep='\,\ ', engine='python')
    df.columns = df.columns.str.replace('# ', '')
    
    new_df = df[['relative_time', 'total_execs', 'edges_found']].copy()
    n_rows_range = range(len(new_df.index))
    prev_time_value = -1
    prev_edge_value = 0
    prev_exec_value = 0
    
    for i in n_rows_range:
        time_value = int(new_df['relative_time'].iloc[i])
        rounded_time_value = closestNumber(time_value, INTERVAL_SECONDS)
        rounded_time_value_str = str(rounded_time_value)
        edge_value = new_df['edges_found'].iloc[i]
        edge_dict_keys = list(edge_values_dict.keys())
        exec_value = new_df['total_execs'].iloc[i]
        exec_dict_keys = list(exec_values_dict.keys())

        if prev_time_value >= 0:
            if rounded_time_value - prev_time_value > INTERVAL_SECONDS:
                mid_time_value = average(rounded_time_value, prev_time_value)
                mid_time_value_str = str(mid_time_value)
                if mid_time_value_str not in edge_dict_keys:
                    edge_values_dict[mid_time_value_str] = []
                if mid_time_value_str not in exec_dict_keys:
                    exec_values_dict[mid_time_value_str] = []
                edge_values_dict[mid_time_value_str] += [average(prev_edge_value, edge_value)]
                exec_values_dict[mid_time_value_str] += [average(prev_exec_value, exec_value)]
            elif rounded_time_value == prev_time_value:
                continue
        
        if rounded_time_value_str not in edge_dict_keys:
            edge_values_dict[rounded_time_value_str] = []
        if rounded_time_value_str not in exec_dict_keys:
            exec_values_dict[rounded_time_value_str] = []
        
        edge_values_dict[rounded_time_value_str] += [edge_value]
        exec_values_dict[rounded_time_value_str] += [exec_value]
        prev_time_value = rounded_time_value
        prev_edge_value = edge_value
        prev_exec_value = exec_value

graph_time_values = []

graph_edge_mean_values = []
graph_edge_min_values = []
graph_edge_max_values = []
graph_edge_median_values = []
graph_edge_q3_values = []
graph_edge_q1_values = []

graph_exec_mean_values = []
graph_exec_min_values = []
graph_exec_max_values = []
graph_exec_median_values = []
graph_exec_q3_values = []
graph_exec_q1_values = []

# Transforming edge values from dict to list

for key in edge_values_dict:
    graph_time_values.append(int(key))
    edge_val_list = edge_values_dict[key]
    graph_edge_mean_values.append(mean(edge_val_list))
    graph_edge_median_values.append(statistics.median(edge_val_list))
    min_edge_value = min(edge_val_list)
    graph_edge_min_values.append(min_edge_value)
    max_edge_value = max(edge_val_list)
    graph_edge_max_values.append(max_edge_value)

    edge_val_list_np = np.array(edge_val_list)
    q3_edge, q1_edge = np.percentile(edge_val_list_np, [75, 25])
    graph_edge_q3_values.append(q3_edge)
    graph_edge_q1_values.append(q1_edge)

# Transforming exec values from dict to list

for key in exec_values_dict:
    exec_val_list = exec_values_dict[key]
    graph_exec_mean_values.append(mean(exec_val_list))
    graph_exec_median_values.append(statistics.median(exec_val_list))
    min_exec_value = min(exec_val_list)
    graph_exec_min_values.append(min_exec_value)
    max_exec_value = max(exec_val_list)
    graph_exec_max_values.append(max_exec_value)

    exec_val_list_np = np.array(exec_val_list)
    q3_exec, q1_exec = np.percentile(exec_val_list_np, [75, 25])
    graph_exec_q3_values.append(q3_exec)
    graph_exec_q1_values.append(q1_exec)

# Plotting mean edges graph

plt.figure(1)
plt.xlabel('Time (s)')
plt.ylabel('Edges found')
plt.plot(graph_time_values, graph_edge_mean_values, linewidth=2, color='b')
# plt.plot(graph_time_values, graph_edge_min_values, linewidth=1, color='b', alpha=0.4)
# plt.plot(graph_time_values, graph_edge_max_values, linewidth=1, color='b', alpha=0.4)
plt.fill_between(graph_time_values, graph_edge_max_values, graph_edge_min_values, color='royalblue', alpha=0.6)
plt.xlim(left=0)
plt.ylim(bottom=0)

# Plotting mean execs graph

plt.figure(2)
plt.xlabel('Time (s)')
plt.ylabel('Total executions')
plt.plot(graph_time_values, graph_exec_mean_values, linewidth=2, color='b')
# plt.plot(graph_time_values, graph_exec_min_values, linewidth=1, color='b', alpha=0.4)
# plt.plot(graph_time_values, graph_exec_max_values, linewidth=1, color='b', alpha=0.4)
plt.fill_between(graph_time_values, graph_exec_max_values, graph_exec_min_values, color='royalblue', alpha=0.8)
plt.xlim(left=0)
plt.ylim(bottom=0)

# Plotting median/IQR edges graph

plt.figure(3)
plt.xlabel('Time (s)')
plt.ylabel('Edges found')
plt.plot(graph_time_values, graph_edge_median_values, linewidth=2, color='b')
# plt.plot(graph_time_values, graph_edge_max_values, linewidth=1, color='b', alpha=0.4)
# plt.plot(graph_time_values, graph_edge_min_values, linewidth=1, color='b', alpha=0.4)
plt.fill_between(graph_time_values, graph_edge_max_values, graph_edge_min_values, color='royalblue', alpha=0.6)
plt.fill_between(graph_time_values, graph_edge_q3_values, graph_edge_q1_values, color='royalblue', alpha=0.8)
plt.xlim(left=0)
plt.ylim(bottom=0)

# Plotting median/IQR execs graph

plt.figure(4)
plt.xlabel('Time (s)')
plt.ylabel('Total executions')
plt.plot(graph_time_values, graph_exec_median_values, linewidth=2, color='b')
# plt.plot(graph_time_values, graph_exec_max_values, linewidth=1, color='b', alpha=0.4)
# plt.plot(graph_time_values, graph_exec_min_values, linewidth=1, color='b', alpha=0.4)
plt.fill_between(graph_time_values, graph_exec_max_values, graph_exec_min_values, color='royalblue', alpha=0.6)
plt.fill_between(graph_time_values, graph_exec_q3_values, graph_exec_q1_values, color='royalblue', alpha=0.8)
plt.xlim(left=0)
plt.ylim(bottom=0)

plt.show()

# Calculating the relation rule between execution time and number of edges found

total_time = graph_time_values[-1]
total_edges_covered = graph_edge_mean_values[-1]
percent_result = 0

for i in range(99, 0, -1):
    percent_coverage = i/100
    result = evaluate(percent_coverage, total_edges_covered, total_time, graph_edge_mean_values, graph_time_values)
    if result[0] != -1:
        percent_coverage_result = percent_coverage
        percent_time_result = 1 - percent_coverage_result
        edge_result = result[0]
        time_result = result[1]
        print("%.2f" % percent_coverage_result)
        print("%.2f" % percent_time_result)
        print(edge_result)
        print(time_result)
        break
