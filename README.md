# FuzzingData
This repository contains data (in the form of charts, graphs, etc. created via Python/Matplotlib) that was obtained while running fuzzing campaigns in different fuzzers, with the aim of comparing the performance of these fuzzers.

To execute: 

1) in the plot_data folder, write on the terminal "python3 mean_and_median_graphs.py 5 1000" 

2) the following graphs will be generated in a folder called "graphs":

- Mean line graph with time as x-axis and edges found as y-axis
- Median line graph + interquartile range with time as x-axis and edges found as y-axis

Many graphs will be generated, which represent the progression of covered edges throughout the campaign. The red dot in the graph represents the optimal point in which x% of edges were covered in (100-x)% of the time, which means that the campaign could be stopped at that point.

Change the first argument number to increase or decrease the time interval in the x-axis. Change the second argument to limit the time of the campaign.


The treemaps folder contains treemap visualizations for fuzzer coverage data. To execute:

1) In the folder, write on the terminal "python3 treemaps_fuzzing.py [name of CSV file 1] [name of CSV file 2]"

Optional arguments: --filename [list of file names] --no-function

There are 4 treemap visualizations available:

- Treemap 1: for first CSV file given, for each function in each file in the list of file names, edges covered in green, edges not covered in red
- Treemap 2: for first CSV file given, percentage of edges covered per function for each file in list of file names
- Treemap 3: difference of edges covered and remainders in each function between 2 CSV files (differece in red/blue, remainder in gray)
- Treemap 4: for first CSV file given, sum of edges covered in green, sum of edges not covered in red (only shown when --no-function is true)
