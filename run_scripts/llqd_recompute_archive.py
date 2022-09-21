import multiprocessing
import argparse
import shutil
import os

from src.map_elites.analysis_utils import get_norm_qd_score
from src.map_elites.llqd_utils import get_dynamics_model, switch_env

from src.envs.hexapod_dart.hexapod_env import HexapodEnv

from src.map_elites import common as cm
from src.map_elites.llqd import evaluate_

import numpy as np
import pandas as pd

from src.map_elites import unstructured_container

from itertools import compress


def addition_condition(s_list, archive, params):
    """
    Checks if species in the s_list qualify for being added to the archive.
    Appends to the archive in here, and returns a list of species to be added

    s_list: list of Species members
    archive: is a list or dict containing the species which contains the genotype, descriptor, etc.

    returns
    add_list: list of species members
    """
    add_list = []  # list of species solutions that were added
    discard_list = []
    for s in s_list:  # for each solution check if gets added to archive
        success = unstructured_container.add_to_archive(s, archive, params)

        if success:  # if success, then add it to the list of species to be added
            add_list.append(s)
        else:  # if not, then add it to the list of species to be discarded
            discard_list.append(s)  # not important for alogrithm but to collect stats

    return archive, add_list, discard_list


def recompute_archive(archive_path, f_real, recompute_name, log_dir, params, pool, filter=False):
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    data = pd.read_csv(archive_path, header=None)  # 'path/some_archive.dat'
    genotypes = data.iloc[:, 4:-1]
    original_descriptors = data.iloc[:, 0:3]  # same order as genotypes
    genotypes = genotypes.to_numpy()
    original_descriptors = original_descriptors.to_numpy()
    # print("original descriptors", original_descriptors)
    print(len(original_descriptors))
    # print("data length in numpy", len(genotypes))
    to_evaluate = []
    for i in range(len(genotypes)):
      to_evaluate += [(genotypes[i], f_real)]

    s_list = cm.parallel_eval(evaluate_, to_evaluate, pool, params)
    updated_descriptors = [np.hstack((s.fitness, s.desc)) for s in s_list]
    updated_descriptors = np.asarray(updated_descriptors)
    # print("updated descriptors", updated_descriptors)
    print(len(updated_descriptors))
    original_archive_length = len(s_list)
    print(f"Length of the s_list pre filter is: {len(s_list)}")

    # Filter out solutions where the updated BD is sufficiently different

    filter_mask = (np.isclose(original_descriptors, updated_descriptors, atol=0.025).sum(axis=1) >= 3).tolist()  # TRUE if similar
    filtered_s_list = list(compress(s_list, filter_mask))
    print(f"Length of the s_list post filter is: {len(filtered_s_list)}")
    filtered_archive_length = len(filtered_s_list)
    true_positive_rate = filtered_archive_length / original_archive_length  # ie the proportion of solutions that we can trust without re-computing, which we can't in real life

    # print(f"The length of the s_list is {len(s_list)}")
    if filter is True:
        # archive, add_list, discard_list = addition_condition(filtered_s_list, [], params) # no need to go through add condition as cannot have 2 sol in same BD if filter ensures no chance in BD
        archive = filtered_s_list
    else:
        archive, add_list, discard_list = addition_condition(s_list, [], params) # need to go through add condition

    cm.save_archive(archive, recompute_name, params, log_dir)
    fit_list = np.array([s.fitness for s in archive])
    fitness_sum = np.sum(fit_list)
    archive_size = len(archive)
    qd_score = get_norm_qd_score(fitness_sum, archive_size)

    results = np.array([archive_size, qd_score, true_positive_rate])

    with open(log_dir + f"/coverage_and_qd_{recompute_name}.txt", 'wb') as f:
        np.savetxt(f, results)
    print('Mean fitness in archive', np.mean(fit_list))  # #7
    print('Median fitness in archive', np.median(fit_list))  # #8
    print('Size of archive / coverage', len(archive))
    print('QD Score', qd_score)
    print('Percent (decimal) of solutions we can trust from original archive:', true_positive_rate)
    print('Fitness sum', fitness_sum)
    return


# This is a script to adjust an archive file to a new environment

def main(args):
    px = \
        {
            # type of qd 'unstructured, grid, cvt'
            "type": args.qd_type,

            # more of this -> higher-quality CVT
            "cvt_samples": 25000,
            # we evaluate in batches to parallelize
            "batch_size": args.b_size,
            # proportion of total number of niches to be filled before starting
            "random_init": 0.005,
            # batch for random initialization
            "random_init_batch": 100,
            # when to write results (one generation = one batch)
            "dump_period": args.dump_period,

            # do we use several cores?
            "parallel": True,
            # min/max of genotype parameters - check mutation operators too
            "min": 0.0,
            "max": 1.0,

            # ------------MUTATION PARAMS---------#
            # selector ["uniform", "random_search"]
            "selector": args.selector,
            # mutation operator ["iso_dd", "polynomial", "sbx"]
            "mutation": args.mutation,

            # probability of mutating each number in the genotype
            "mutation_prob": 0.2,

            # param for 'polynomial' mutation for variation operator
            "eta_m": 10.0,

            # only useful if you use the 'iso_dd' variation operator
            "iso_sigma": 0.01,
            "line_sigma": 0.2,

            # --------UNSTRUCTURED ARCHIVE PARAMS----#
            # l value - should be smaller if you want more individuals in the archive
            # - solutions will be closer to each other if this value is smaller.
            "nov_l": 0.015,
            "eps": 0.1,  # usually 10%
            "k": 15,  # from novelty search

            # --------MODEL BASED PARAMS-------#
            "t_nov": 0.03,
            "t_qua": 0.0,
            "k_model": 15,
            # Comments on model parameters:
            # t_nov is correlated to the nov_l value in the unstructured archive
            # If it is smaller than the nov_l value, we are giving the model more chances which might be more wasteful
            # If it is larger than the nov_l value, we are imposing that the model must predict something more novel than we would normally have before even trying it out
            # fitness is always positive - so t_qua

            "model_variant": "dynamics",  # "direct", # "dynamics" or "direct"
            "train_model_on": True,
            "train_freq": 40,  # train at a or condition between train freq and evals_per_train
            "evals_per_train": 50,  # 500
            "log_model_stats": False,
            "log_time_stats": False,

            # 0 for random emitter, 1 for optimizing emitter
            # 2 for random walk emitter, 3 for model disagreement emitter
            "emitter_selection": 0,

        }

    params = px
    archive_path = args.filename
    log_dir = args.log_dir
    env_id = args.env_id

    # setup the parallel processing pool

    num_cores = multiprocessing.cpu_count()  # use all cores

    pool = multiprocessing.get_context("spawn").Pool(num_cores)

    # Deterministic = "det", Probabilistic = "prob"
    dynamics_model_type = "prob"

    print("Dynamics model type: ", dynamics_model_type)
    dynamics_model, dynamics_model_trainer = get_dynamics_model(dynamics_model_type)

    switch_env(env_id)

    env = HexapodEnv(dynamics_model=dynamics_model,
                     render=False,
                     record_state_action=True,
                     ctrl_freq=100)

    f_real = env.evaluate_solution

    recompute_archive(archive_path, f_real, env_id, log_dir, params, pool, args.filter)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str)  # file to visualize rollouts from
    parser.add_argument("--env_id", type=int, default=0)
    # -----------------Type of QD---------------------#
    # options are 'cvt', 'grid' and 'unstructured'
    parser.add_argument("--qd_type", type=str, default="unstructured")

    # ---------------CPU usage-------------------#
    parser.add_argument("--num_cores", type=int, default=8)

    # -----------Store results + analysis-----------#
    parser.add_argument("--log_dir", type=str)

    # -----------QD params for cvt or GRID---------------#
    # ONLY NEEDED FOR CVT OR GRID MAP ELITES - not needed for unstructured archive
    parser.add_argument("--dim_map", default=2, type=int)  # Dim of behaviour descriptor
    parser.add_argument("--grid_shape", default=[100, 100], type=list)  # num discretization
    parser.add_argument("--n_niches", default=3000, type=int)

    # ----------population params--------#
    parser.add_argument("--b_size", default=200, type=int)  # For parallelization -
    parser.add_argument("--dump_period", default=5000, type=int)
    parser.add_argument("--max_evals", default=400, type=int)  # max number of evaluation 1e6
    parser.add_argument("--selector", default="uniform", type=str)
    parser.add_argument("--mutation", default="iso_dd", type=str)

    parser.add_argument("--save_name", default="standard", type=str)
    parser.add_argument("--filter", default=True, type=bool)

    args = parser.parse_args()

    main(args)

# python3 run_scripts/llqd_recompute_archive.py --log_dir results/experiment_0/recomputed/archive_5014 --filename results/experiment_0/archive_5014.dat --env_id 1
# --log_dir specifies the folder into which to save the result; --filename specifies the archive file we want to recompute; --env_id specifies the env to recompute in

# python3 run_scripts/llqd_recompute_archive.py --log_dir results/experiment_4/daqd/run_0/recomputed/ --filename results/experiment_4/daqd/run_0/0/archive_1300.dat --env_id 1 --filter True

# python3 run_scripts/llqd_recompute_archive.py --log_dir results/experiment_3/200_easy/run_0/recomputed/ --filename results/experiment_3/200_easy/run_0/0/archive_200.dat --env_id 0 --filter True