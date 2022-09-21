import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import stats
from scipy.stats import ranksums


def get_norm_qd_score(fitness_sum, archive_size):
    """
    Calculate the QD score given the fitness_sum and archive_size
    fitness_sum: np.array of shape intervals x 1, where intervals is number of entries in log_file we evaluate
    archive_size: np.array of shape intervals x 1, where intervals is number of entries in log_file we evaluate

    returns: np.array of shape intervals x 1, where intervals is number of entries in log_file we evaluate
    """
    norm_qd_score = (fitness_sum + archive_size * np.pi) / np.pi  # fitness_sum + archive_size * pi gives archive_size times a value between 0 and pi
    return norm_qd_score


def interpolate_log_file(log_file, value='coverage', start=0, end=900, interval=20):
    """
    Used to get interpolated data from a single log_file
    Can do this for coverage vs num_evals or QD Score vs num_evals depending on value parameter
    Define start, end and interval of num_evals through the start, end, interval parameters
    """
    if value == 'coverage':
        data = np.loadtxt(log_file, usecols=[1, 3])  # evals, archive
        data = np.insert(data, [0], [0, 0], axis=0)  # insert 0 datapoint at beginning
        x = data[:, 0]  # x is now num_evals
        y = data[:, 1]  # y is now archive_size

    elif value == 'qd_score':
        data = np.loadtxt(log_file, usecols=[1, 3, 5])  # evals, archive_size, fitness_sum
        data = np.insert(data, [0], [0, 0, 0], axis=0)  # insert 0 datapoint at beginning
        data[:, 2] = get_norm_qd_score(data[:, 2], data[:, 1])  # replace fitness_sum with qd_score; function takes in fitness_sum and archive_size in order
        x = data[:, 0]  # x is now num_evals
        y = data[:, 2]  # y is now QD Score

    else:
        raise NotImplementedError

    function_approximation = interp1d(x, y, kind='quadratic', fill_value='extrapolate')  # Learn the function approximation given the real data

    x_new = np.arange(start, end, interval)  # Set up the new domain for interpolation

    return function_approximation(x_new)


def plot_multiple_value_vs_evals(list_of_list_of_int_data, start, end, interval, list_of_titles, ylabel):
    """
    Used to plot one or many averaged (with IQ range) value histories (QD Score or Coverage vs num_evals)
    list_of_list_of_int_data: list of lists, the latter of which are the interpolated data for the same type across runss, ie
    [daqd_log_file_run_0, daqd_log_file_run_1, etc]. The second list could be [llqd_log_file_run_0, llqd_log_file_run_1, etc]

    list_of_titles contains the titles of each list in order, in above example case ['daqd', 'llqd']

    ylabel is label of y axis, either 'qd_core" or "coverage"

    We then plot the result
    """
    mid_values = []
    final_values = []
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots()
    for i in range(len(list_of_list_of_int_data)):
        list_of_int_data = list_of_list_of_int_data[i]
        data_stack = np.vstack(list_of_int_data)
        # print(data_stack[:,-1])
        mid_values.append(data_stack[:, 20])  # 20 is at 400 evals; 40 at 800 evals
        final_values.append(data_stack[:, 40])  # 20 is at 400 evals; 40 at 800 evals
        median = np.mean(data_stack, 0)
        iq_low = np.percentile(data_stack, 25, axis=0)
        iq_high = np.percentile(data_stack, 75, axis=0)
        x_new = np.arange(start, end, interval)
        ax.plot(x_new, median, '-', label=list_of_titles[i])
        ax.fill_between(x_new, iq_low, iq_high, alpha=0.2)
    plt.xlabel("Real evaluations", fontsize=14)
    if ylabel == 'coverage':
        plt.ylabel("Coverage", fontsize=14)
    elif ylabel == 'qd_score':
        plt.ylabel("QD score", fontsize=14)
    plt.legend(loc="upper left", fontsize=14)
    plt.tight_layout()
    plt.show()
    # print(stats.ttest_ind(mid_values[0], mid_values[1], equal_var=False, alternative='greater'))
    # print(stats.ttest_ind(final_values[0], final_values[1], equal_var=False, alternative='greater'))
    print("@400 ranksum:",ranksums(mid_values[0], mid_values[1], alternative='greater'))
    print("@800 ranksum", ranksums(final_values[0], final_values[1], alternative='greater'))


def get_final_qd_score_and_coverage(log_file):
    """
    Function to retrieve the final QD score and coverage for llqd/experiment_3/200_easy (recomputed),
    which we will later compare to the recomputed archives for DAQD
    """
    pass


# fitness= np.array([[-84.37631631084145],
#                    [-110.0396654815323],
#                    [-120.0830308249851],
#                    [-133.97164652764783]
#                    ])
#
# evals = np.array([[223],
#                    [460],
#                    [678],
#                    [897]
#                    ])
#
# print(get_norm_qd_score(fitness, evals))
