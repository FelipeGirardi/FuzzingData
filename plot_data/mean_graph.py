import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#808080']

for i in range(1,9):
    data = pd.read_csv('fuzzer0' + str(i), sep='\,\ ', engine='python')
    data.columns = data.columns.str.replace('# ', '')
    data = data.groupby('edges_found').mean()
    relative_time_list = data['relative_time'].tolist()
    edges_found_list = data.index.values

    plt.figure(1)
    xmin, xmax = min(relative_time_list), max(relative_time_list)
    ymin, ymax = min(edges_found_list), max(edges_found_list)
    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)
    plt.xlabel('Time (s)')
    plt.ylabel('Edges found')
    plt.plot(relative_time_list, edges_found_list, linewidth = 2.0, color=colors[i-1], label='fuzzer0' + str(i))
    leg = plt.legend(loc='lower right')
    
plt.show()


# total_execs plot

# total_execs_list = data['total_execs'].tolist()
# plt.figure(2)
# ymin, ymax = min(total_execs_list), max(total_execs_list)
# plt.ylim(ymin, ymax)
# plt.xlim(xmin, xmax)
# plt.plot(relative_time_list, total_execs_list, linewidth = 2.0, color='red')
# plt.xlabel('Time (s)')
# plt.ylabel('Total executions')
# plt.show()