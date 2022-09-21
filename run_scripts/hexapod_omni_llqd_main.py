import os, sys
import time
import argparse
import numpy as np
import torch
from copy import deepcopy
import shutil
import src.torch.pytorch_util as ptu

from src.map_elites.llqd import ModelBasedLLQD
from src.map_elites.llqd_utils import evaluate_transition_likelihood, get_dynamics_model, evaluate_likelihood_for_slist, \
    make_random_int_list, make_switching_int_list
from src.envs.hexapod_dart.hexapod_env import HexapodEnv
from src.models.surrogate_models.det_surrogate import DeterministicQDSurrogate
from src.data_management.replay_buffers.simple_replay_buffer import SimpleReplayBuffer
from src.map_elites import common as cm
from src.map_elites.llqd_utils import switch_env


def get_surrogate_model():
    from src.trainers.qd.surrogate import SurrogateTrainer
    dim_x = 36  # genotype dimnesion
    model = DeterministicQDSurrogate(gen_dim=dim_x, bd_dim=2, hidden_size=64)
    model_trainer = SurrogateTrainer(model, batch_size=32)

    return model, model_trainer


def main(args):

    EPSILON = args.epsilon

    print(f"------------------------------------------------Starting LLQD---------------------------------------------")
    MAX_TOTAL_EVALS = 1000000
    model_count = 0

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
            "random_init_batch": args.random_init_batch,
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
            "min_model_add": args.min_model_add,

        }

    dim_x = 36  # genotype size
    obs_dim = 48
    action_dim = 18

    # Deterministic = "det", Probabilistic = "prob"
    dynamics_model_type = "prob"

    print("Dynamics model type: ", dynamics_model_type)
    dynamics_model, dynamics_model_trainer = get_dynamics_model(dynamics_model_type)
    surrogate_model, surrogate_model_trainer = get_surrogate_model()

    env = HexapodEnv(dynamics_model=dynamics_model,
                     render=False,
                     record_state_action=True,
                     ctrl_freq=100)

    f_real = env.evaluate_solution

    if dynamics_model_type == "det":
        f_model = env.evaluate_solution_model
    elif dynamics_model_type == "prob":
        f_model = env.evaluate_solution_model_ensemble

    # ------- START of LLQD LOOP -------

    # initialize replay buffer
    replay_buffer = SimpleReplayBuffer(
        max_replay_buffer_size=150000,
        observation_dim=obs_dim,
        action_dim=action_dim,
        env_info_sizes=dict(),
    )

    def create_new_model(s_list_transfer, dynamics_model_transfer, model_count, archive_transfer):
        """
        Function to create a new LLQD object.
        """

        model_count += 1

        new_dynamics_model, new_dynamics_model_trainer = get_dynamics_model(dynamics_model_type)
        new_surrogate_model, new_surrogate_model_trainer = get_surrogate_model()

        new_env = HexapodEnv(dynamics_model=new_dynamics_model,
                             render=False,
                             record_state_action=True,
                             ctrl_freq=100)

        new_f_real = new_env.evaluate_solution

        if dynamics_model_type == "det":
            new_f_model = new_env.evaluate_solution_model
        elif dynamics_model_type == "prob":
            new_f_model = new_env.evaluate_solution_model_ensemble

        new_replay_buffer = SimpleReplayBuffer(
            max_replay_buffer_size=150000,
            observation_dim=obs_dim,
            action_dim=action_dim,
            env_info_sizes=dict(),
        )

        log_dir = (args.log_dir + '/' + str(model_count))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        new_llqd = ModelBasedLLQD(args.dim_map, dim_x,
                                  new_f_real, new_f_model,
                                  new_surrogate_model, new_surrogate_model_trainer,
                                  new_dynamics_model, new_dynamics_model_trainer,
                                  new_replay_buffer,
                                  n_niches=args.n_niches,
                                  params=px, log_dir=args.log_dir + '/' + str(model_count))

        new_llqd.s_list = s_list_transfer
        new_llqd.dynamics_model.load_state_dict(dynamics_model_transfer.state_dict())
        new_llqd.gen_start_time = time.time()
        print(f"New model made; the log_dir is {new_llqd.log_dir}")

        return new_llqd, model_count

    # We will have multiple LLQD objects, ideally one for each new environment.
    # When we reach a state where the likelihoods are sufficiently far away from the mean for the given model,
    # we create a new object / model.
    # For batches of observations, we take the mean likelihood and use this to determine which of the available
    # models is best.
    # If all are sufficiently far away (worse), create a new model object.

    # 0. Make a random list of envs that the simulator will visit
    if args.high_difficulty == 1:
        # constant random changes of env
        env_list = make_switching_int_list(6, 3)
    else:
        # randome changes of env
        env_list = make_random_int_list(6, 3, initialise=False)

    print(f"The envs we will visit in order are {env_list}")
    env_counter = 0
    env_id = env_list[env_counter]

    switch_env(env_id)  # set env to baseline
    print(f"We are in env {env_id}")

    # 0. Create dir if does not exist yet
    log_dir = (args.log_dir + '/' + str(model_count))
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    # 1. Create an initial model and a list to append future models to
    llqd = ModelBasedLLQD(args.dim_map, dim_x,
                          f_real, f_model,
                          surrogate_model, surrogate_model_trainer,
                          dynamics_model, dynamics_model_trainer,
                          replay_buffer,
                          n_niches=args.n_niches,
                          params=px, log_dir=args.log_dir + '/' + str(model_count))
    print(f"the log_dir is {llqd.log_dir}")
    # Train the first model so we have starting likelihoods
    llqd.compute_s_list(num_cores_set=args.num_cores, max_evals=args.max_evals)
    llqd.compute_more_stuff(num_cores_set=args.num_cores, max_evals=args.max_evals)

    model_list = [llqd]
    tmp_best_model = llqd

    current_evals = [model.n_evals for model in model_list]
    most_evals = max(current_evals)
    print(f"The model with the highest evals curently has {most_evals} evals.")

    loop_counter = 0

    visited_models = [0]

    while env_counter < len(env_list) - 1:
        # Extra stopping condition based on total evals across all models
        if sum(current_evals) > MAX_TOTAL_EVALS:
            break

        env_counter += 1
        env_id = env_list[env_counter]
        print(f"Env counter value: {env_counter}")
        print("Visited:", visited_models)
        print("Env_list", env_list)

        print("-----------------------------WE ARE CHANGING THE ENV---------------------------------")
        switch_env(env_id)
        print(f"The env has switched to env {env_id}")

        loop_counter += 1
        print(f"We have entered the loop for the {loop_counter} time")

        # 2. Compute species lists for the current best model
        print("Computing s_list for best fitting model...")
        tmp_best_model.compute_s_list(num_cores_set=args.num_cores, max_evals=args.max_evals)
        # make a copy of this s_list to enable passing it to other models
        s_list_transfer = deepcopy(tmp_best_model.s_list)
        print("Successfully computed s_list for best fitting model. Its size is:", len(tmp_best_model.s_list))

        # 3. Compare the likelihoods for all models and pick the best (closest to zero)
        print("Computing likelihoods of s_list observations for all models")
        likelihoods = [evaluate_likelihood_for_slist(tmp_best_model.s_list, model.dynamics_model) for model in
                       model_list]
        print("The likelihoods for the s_list per model are:", likelihoods)
        likelihood_means = [model.mean_likelihood for model in model_list]  # retrieve mean likelihood for each object
        print("The likelihood means per model are:", likelihood_means)
        likelihood_stds = [model.std_likelihood for model in model_list]  # retrieve std of likelihoods for each object
        print("The likelihood stdds per model are:", likelihood_stds)

        # 3.1. Create a new model if none of the existing models are good enough
        num_models = len(model_list)
        print(f"We currently have {num_models} models in the system")

        # Save the index of the model that fits the datapoints best in case a new model needs to be initialised
        transfer_learning_candidate_idx = np.argmax(likelihoods)
        print(f"The model that best fits the s_list datapoints and will be used in case we need a new model has index {transfer_learning_candidate_idx}")

        print(f"We are currently in model {int(np.where([tmp_best_model is model_list[i] for i in range(len(model_list))])[0])}")
        print("Now checking if we need to switch models...")
        for k in range(num_models):
            # Pick the highest likelihood model and check it is within the bounds
            idx_max = np.argmax(likelihoods)
            if likelihoods[idx_max] > (likelihood_means[idx_max] - EPSILON * likelihood_stds[idx_max]):
                # Proceed with this best model
                tmp_best_model = model_list[idx_max]
                # Pass the s_list into this model
                tmp_best_model.s_list = s_list_transfer
                break
            else:
                # Check the next best model
                del likelihoods[idx_max]
                del likelihood_means[idx_max]
                del likelihood_stds[idx_max]

        # If none of the models were adequate, create a new model
        # Add any logic that needs to be done on a first model run here
        if len(likelihoods) == 0:
            print("Substantially different env was detected. Create new model")

            # Create copies of elements we want to transfer to the new model from the existing model that best fits
            # the s_list datapoints

            dynamics_model_transfer = model_list[transfer_learning_candidate_idx].dynamics_model
            archive_transfer = deepcopy(model_list[transfer_learning_candidate_idx].archive)

            tmp_best_model, model_count = create_new_model(s_list_transfer, dynamics_model_transfer, model_count, archive_transfer)
            model_list.append(tmp_best_model)

            print("Check that buffer of new model is 0:", tmp_best_model.replay_buffer._size)

        print(f"After checking the transitions, we are now in model {int(np.where([tmp_best_model is model_list[i] for i in range(len(model_list))])[0])}")
        visited_models.append(int(np.where([tmp_best_model is model_list[i] for i in range(len(model_list))])[0]))

        # 4. Add transitions to buffer of best model, train, and compute holdout set likelihoods
        print("Entering update_buffer_and_train() function...")
        tmp_best_model.compute_more_stuff(num_cores_set=args.num_cores, max_evals=args.max_evals)
        print("Successfully updated buffer for the best fitting model, performed training and holdout likelihood stats.")

        # 5. Update evals and continue with loop if necessary
        current_evals = [model.n_evals for model in model_list]
        most_evals = max(current_evals)
        print(f"The model with the highest number of evals now has {most_evals} evals.")

    print("---------Final stats---------")
    print(f"Number of models: {len(model_list)}")
    print("The size of each model's archive in order is:")
    for model in model_list:
        print(len(model.archive))

    print("The size of each model's replay buffer in order is:")
    for model in model_list:
        print(model.replay_buffer._size)

    print("The number of real evals of each model in order is:")
    for model in model_list:
        print(model.n_evals)

    print("The most recent mean and std likelihood of each model in order is:")
    for model in model_list:
        print("Mean:", model.mean_likelihood, "Std:", model.std_likelihood)

    print("The true environments simulated in order were:", visited_models)
    print("The detected LLQD objects were:", visited_models)

    # 6 Save the env_list and visited_list
    env_visited_array = np.array([env_list, visited_models])
    with open(args.log_dir + '/envs_vs_estimates.npy', 'wb') as f:
        np.save(f, env_visited_array)
    with open(args.log_dir + '/envs_vs_estimates.txt', 'wb') as g:
        np.savetxt(g, env_visited_array)


    # 7 Save the archives
    for model in model_list:
        cm.save_archive(model.archive, model.n_evals, model.params, model.log_dir)
        ptu.save_model(model.dynamics_model, model.log_dir + '/model.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--grid_shape", default=[100, 100], type=list)  # num discretizer
    parser.add_argument("--n_niches", default=3000, type=int)

    # ----------population params--------#
    parser.add_argument("--b_size", default=200, type=int)  # For parallelization -
    parser.add_argument("--dump_period", default=5000, type=int)
    parser.add_argument("--max_evals", default=400, type=int)  # max number of evaluation 1e6
    parser.add_argument("--selector", default="uniform", type=str)
    parser.add_argument("--mutation", default="iso_dd", type=str)

    parser.add_argument("--save_name", default="standard", type=str)
    parser.add_argument("--epsilon", default=0.5, type=float)

    parser.add_argument("--min_model_add", default=100, type=int)
    parser.add_argument("--random_init_batch", default=100, type=int)

    parser.add_argument("--high_difficulty", default=0, type=int)

    args = parser.parse_args()

    main(args)

# python3 run_scripts/hexapod_omni_llqd_main.py --log_dir results/test_main_script --dump_period 500000 --epsilon 0.5 --min_model_add 200 --random_init_batch 200 --high_difficulty 0
