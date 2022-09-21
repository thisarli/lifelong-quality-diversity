import os
import numpy as np
from scipy import stats
from scipy.stats import ranksums, wilcoxon


statistics_llqd = []
statistics_daqd = []

for run in range(20):
    path = 'results/experiment_3/200_easy/run_' + str(run) + '/recomputed/'
    results = np.vstack([np.loadtxt(path + f) for f in os.listdir(path) if f.endswith('.txt')])
    num_reliable = results[:, 0].sum()
    num_total = (results[:, 0] / results[:, 2]).sum()

    # reliable solutions out of total solutions
    reliability = num_reliable / num_total

    # of the reliable solutions, what is the coverage and qd score (their quality)
    total_reliable_coverage = results[:, 0].sum()
    total_reliable_qd_score = results[:, 1].sum()

    print(f"----------run {run}--------")
    print("Reliability is:", reliability)
    print("Reliable coverage is", total_reliable_coverage)
    print("Reliable QD score iw:", total_reliable_qd_score)
    statistics_llqd.append([reliability, total_reliable_coverage, total_reliable_qd_score])

    path = 'results/experiment_4/daqd/run_' + str(run) + '/recomputed/filtered/'

    results = np.vstack([np.loadtxt(path + f) for f in os.listdir(path) if f.endswith('.txt')])
    num_reliable = results[:, 0].sum()
    num_total = (results[:, 0] / results[:, 2]).sum()

    # reliable solutions out of total solutions
    reliability = num_reliable / num_total

    # of the reliable solutions, what is the coverage and qd score (their quality)
    total_reliable_coverage = results[:, 0].sum()
    total_reliable_qd_score = results[:, 1].sum()

    print(f"----------run {run}--------")
    print("Reliability is:", reliability)
    print("Reliable coverage is", total_reliable_coverage)
    print("Reliable QD score iw:", total_reliable_qd_score)
    statistics_daqd.append([reliability, total_reliable_coverage, total_reliable_qd_score])

stats_daqd = np.array(statistics_daqd)
stats_llqd = np.array(statistics_llqd)

print("-----------Final Stats-----------")
print("Mean LLQD coverage outperformance (%)", np.mean(stats_llqd[:, 1] / stats_daqd[:, 1] - 1))
print("Median LLQD coverage outperformance (%)", np.percentile(stats_llqd[:, 1] / stats_daqd[:, 1] - 1, 50))
print("IQ_low LLQD coverage outperformance (%)", np.percentile(stats_llqd[:, 1] / stats_daqd[:, 1] - 1, 25))
print("IQ_high LLQD coverage outperformance (%)", np.percentile(stats_llqd[:, 1] / stats_daqd[:, 1] - 1, 75))
print("Mean Coverage LLQD > DAQD P-value ind:", stats.ttest_ind(stats_llqd[:, 1], stats_daqd[:, 1], equal_var=False, alternative='greater'))
print("Mean Coverage LLQD > DAQD P-value rel:", stats.ttest_rel(stats_llqd[:, 1], stats_daqd[:, 1], alternative='greater'))
print("Ranksum Coverage LLQD > DAQD P-value:", wilcoxon(stats_llqd[:, 1], stats_daqd[:, 1], alternative='greater'))

print("Mean LLQD coverage", np.mean(stats_llqd[:, 1]))
print("Mean DAQD coverage", np.mean(stats_daqd[:, 1]))
print("Median LLQD coverage", np.median(stats_llqd[:, 1]))
print("Median DAQD coverage", np.median(stats_daqd[:, 1]))


print("Mean LLQD QD score outperformance (%)", np.mean(stats_llqd[:, 2] / stats_daqd[:, 2] - 1))
print("Median LLQD QD score outperformance (%)", np.percentile(stats_llqd[:, 2] / stats_daqd[:, 2] - 1, 50))
print("IQ_low LLQD QD score outperformance (%)", np.percentile(stats_llqd[:, 2] / stats_daqd[:, 2] - 1, 25))
print("IQ_high LLQD QD score outperformance (%)", np.percentile(stats_llqd[:, 2] / stats_daqd[:, 2] - 1, 75))
print("Mean QD Score LLQD > DAQD P-value ind:", stats.ttest_ind(stats_llqd[:, 2], stats_daqd[:, 2], equal_var=False, alternative='greater'))
print("Mean QD Score LLQD > DAQD P-value rel:", stats.ttest_rel(stats_llqd[:, 2], stats_daqd[:, 2], alternative='greater'))
print("Ranksum QD Score LLQD > DAQD P-value:", wilcoxon(stats_llqd[:, 2], stats_daqd[:, 2], alternative='greater'))

print("Mean LLQD QD score", np.mean(stats_llqd[:, 2]))
print("Mean DAQD QD score", np.mean(stats_daqd[:, 2]))
print("Median LLQD QD score", np.median(stats_llqd[:, 2]))
print("Median DAQD QD score", np.median(stats_daqd[:, 2]))

print("Median LLQD reliability outperformance (pp)", np.percentile(stats_llqd[:, 0] - stats_daqd[:, 0], 50))
print("IQ_low LLQD reliability outperformance (pp)", np.percentile(stats_llqd[:, 0] - stats_daqd[:, 0], 25))
print("IQ_high LLQD reliability outperformance (pp)", np.percentile(stats_llqd[:, 0] - stats_daqd[:, 0], 75))
print("Mean LLQD reliability outperformance (pp)", np.mean(stats_llqd[:, 0] - stats_daqd[:, 0]))
print("Median reliability llqd (%)", np.median(stats_llqd[:, 0]))
print("Mean reliability llqd (%)", np.mean(stats_llqd[:, 0]))
print("Median reliability daqd (%)", np.median(stats_daqd[:, 0]))
print("Mean reliability daqd (%)", np.mean(stats_daqd[:, 0]))
print("Mean Reliability LLQD > DAQD P-value ind:", stats.ttest_ind(stats_llqd[:, 0], stats_daqd[:, 0], equal_var=False, alternative='greater'))
print("Mean Reliability LLQD > DAQD P-value rel:", stats.ttest_rel(stats_llqd[:, 0], stats_daqd[:, 0], alternative='greater'))
print("Ranksum Reliability LLQD > DAQD P-value:", wilcoxon(stats_llqd[:, 0], stats_daqd[:, 0], alternative='greater'))

# Use related test here, as you evaluate scores that come from the same sequence of environments across LLQD and DAQD (ie paired)

