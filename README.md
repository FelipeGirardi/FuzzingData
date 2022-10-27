# FuzzingData
This repository contains data (in the form of charts, graphs, etc. created via Python/Matplotlib) that was obtained while running fuzzing campaigns in different fuzzers, with the aim of comparing the performance of these fuzzers.

To execute: 

1) in the plot_data folder, write on the terminal "python3 mean_and_median_graphs.py 5" 

2) this will generate 4 graphs:

- Mean line graph with time as x-axis and edges found as y-axis
- Mean line graph with time as x-axis and number of executions as y-axis
- Median line graph + interquartile range with time as x-axis and edges found as y-axis
- Median line graph + interquartile range with time as x-axis and number of executions as y-axis

Change the argument number to increase or decrease the time interval in the x-axis and see the changes in the graphs.