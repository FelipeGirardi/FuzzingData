# FuzzingData
This repository contains data (in the form of charts, graphs, etc. created via Python/Matplotlib) that was obtained while running fuzzing campaigns in different fuzzers, with the aim of comparing the performance of these fuzzers.

To execute: 

1) in the plot_data folder, write on the terminal "python3 mean_and_median_graphs.py 5 1000" 

2) the following graphs will be generated in a folder called "graphs":

- Mean line graph with time as x-axis and edges found as y-axis
- Median line graph + interquartile range with time as x-axis and edges found as y-axis

Many graphs will be generated, which represent the progression of covered edges throughout the campaign. The red dot in the graph represents the optimal point in which x% of edges were covered in (100-x)% of the time, which means that the campaign could be stopped at that point.

Change the first argument number to increase or decrease the time interval in the x-axis. Change the second argument to limit the time of the campaign.

The treemaps folder contains treemap visualizations for fuzzer coverage data.