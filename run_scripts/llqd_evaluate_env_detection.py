import numpy as np

from src.map_elites.llqd_utils import get_strict_score, random_strict_score
from scipy import stats
from scipy.stats import ranksums, wilcoxon

for eps_num in ['03', '05', '07']:
    scores_per_run = []
    for run in range(0, 20):
        # env_est_list = np.load('results/experiment_1/eps_'+eps_num+'/run_'+str(run)+'/envs_vs_estimates.npy')
        env_est_list = np.load('results/experiment_3/200_easy' + '/run_' + str(run) + '/envs_vs_estimates.npy')

        map_dict, score_list = get_strict_score(env_est_list[0], env_est_list[1])
        score = np.sum(score_list[1:])
        scores_per_run.append(score)

        # print(env_est_list)
        # print(get_strict_score(env_est_list[0], env_est_list[1]))
        # print(score)
    print(f"-----EPS: {eps_num}-----")
    print(scores_per_run)
    print('Mean:', np.mean(scores_per_run), '{:.1%}'.format(np.mean(scores_per_run) / 5))
    print('Median:', np.median(scores_per_run), '{:.1%}'.format(np.median(scores_per_run) / 5))
    print('25th percentile', np.percentile(scores_per_run, 25), '{:.1%}'.format(np.percentile(scores_per_run, 25) / 5))
    print('75th percentile', np.percentile(scores_per_run, 75), '{:.1%}'.format(np.percentile(scores_per_run, 75) / 5))

# python run_scripts/llqd_evaluate_env_detection.py

# Random counterfactual
random_strict_score()

# Wilcoxon ranksum
easy = [5, 3, 3, 5, 5, 4, 4, 2, 2, 5, 2, 5, 5, 5, 3, 4, 5, 5, 5, 4]
diff = [4, 4, 2, 2, 3, 5, 5, 5, 4, 2, 3, 5, 5, 3, 5, 3, 5, 5, 2, 5]

print("Ranksum % correct diff < easy P-value:", ranksums(diff, easy, alternative='less'))