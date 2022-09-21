import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from src.map_elites.analysis_utils import get_norm_qd_score, interpolate_log_file, plot_multiple_value_vs_evals

# Define num_evals range used in interpolation
start = 0
end = 900
interval = 20

# Instantiate lists that hold interpolated data per type (llqd, daqd)
llqd_data_list = []
daqd_data_list = []

# Define metric of interest, 'qd_score' or 'coverage'
metric = 'qd_score'

# Iterate through log_files per run to retrieve the interpolated data

# in 0_from_2 runs 3, 6, 8 are faulty, so [0,1,2,4,5,7,9]

for run in range(10):
    # For env 1 llqd
    llqd_data_list += [interpolate_log_file('results/experiment_2/llqd_transfer_learn/2_from_1/run_' + str(run) +'/1/log_file.dat',
                                       value=metric, start=start, end=end, interval=interval)]

for run in range(10):
    # For env 2 daqd
    daqd_model = '2'
    daqd_data_list += [interpolate_log_file('results/experiment_2/daqd_baseline/run_' + str(run) + '/' + daqd_model + '/log_file.dat',
                                       value=metric, start=start, end=end, interval=interval)]

# plot_value_vs_evals(llqd_data_list, start, end, interval)
# plot_value_vs_evals(daqd_data_list, start, end, interval)

plot_multiple_value_vs_evals([llqd_data_list, daqd_data_list], start, end, interval, ['LLQD median and IQ range', 'DAQD median and IQ range'], metric)


# python3 run_scripts/experiment_2_plots.py
