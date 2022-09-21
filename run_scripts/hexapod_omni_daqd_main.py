import os, sys
import argparse
import numpy as np
import torch

import src.torch.pytorch_util as ptu
from src.map_elites.llqd_utils import evaluate_transition_likelihood, get_dynamics_model, switch_env

from src.map_elites.mbqd import ModelBasedQD

from src.envs.hexapod_dart.hexapod_env import HexapodEnv


from src.models.surrogate_models.det_surrogate import DeterministicQDSurrogate

# added in get dynamics model section
#from src.trainers.mbrl.mbrl_det import MBRLTrainer
#from src.trainers.mbrl.mbrl import MBRLTrainer
#from src.trainers.qd.surrogate import SurrogateTrainer

from src.data_management.replay_buffers.simple_replay_buffer import SimpleReplayBuffer


def get_surrogate_model():
    from src.trainers.qd.surrogate import SurrogateTrainer
    dim_x=36 # genotype dimnesion    
    model = DeterministicQDSurrogate(gen_dim=dim_x, bd_dim=2, hidden_size=64)
    model_trainer = SurrogateTrainer(model, batch_size=32)

    return model, model_trainer



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
        
        #------------MUTATION PARAMS---------#
        # selector ["uniform", "random_search"]
        "selector" : args.selector,
        # mutation operator ["iso_dd", "polynomial", "sbx"]
        "mutation" : args.mutation,
    
        # probability of mutating each number in the genotype
        "mutation_prob": 0.2,

        # param for 'polynomial' mutation for variation operator
        "eta_m": 10.0,
        
        # only useful if you use the 'iso_dd' variation operator
        "iso_sigma": 0.01,
        "line_sigma": 0.2,

        #--------UNSTURCTURED ARCHIVE PARAMS----#
        # l value - should be smaller if you want more individuals in the archive
        # - solutions will be closer to each other if this value is smaller.
        "nov_l": 0.015,
        "eps": 0.1, # usually 10%
        "k": 15,  # from novelty search


        #--------MODEL BASED PARAMS-------#
        "t_nov": 0.03,
        "t_qua": 0.0, 
        "k_model": 15,
        # Comments on model parameters:
        # t_nov is correlated to the nov_l value in the unstructured archive
        # If it is smaller than the nov_l value, we are giving the model more chances which might be more wasteful 
        # If it is larger than the nov_l value, we are imposing that the model must predict something more novel than we would normally have before even trying it out
        # fitness is always positive - so t_qua

        "model_variant": "dynamics", #"direct", # "dynamics" or "direct"  
        "train_model_on": True, #                                                                              
        "train_freq": 40, # train at a or condition between train freq and evals_per_train
        "evals_per_train": 50, # 500
        "log_model_stats": False,
        "log_time_stats": False, 

        # 0 for random emiiter, 1 for optimizing emitter
        # 2 for random walk emitter, 3 for model disagreement emitter
        "emitter_selection": 0,
        
    }

    dim_x = 36 #genotype size
    obs_dim = 48
    action_dim = 18
    
    # Deterministic = "det", Probablistic = "prob" 
    dynamics_model_type = "prob"

    print("Dynamics model type: ", dynamics_model_type) 
    dynamics_model, dynamics_model_trainer = get_dynamics_model(dynamics_model_type)
    surrogate_model, surrogate_model_trainer = get_surrogate_model()

    env_id = args.env_id
    switch_env(env_id)

    env = HexapodEnv(dynamics_model=dynamics_model,
                     render=False,
                     record_state_action=True,
                     ctrl_freq=100)
    
    f_real = env.evaluate_solution # maybe move f_real and f_model inside

    if dynamics_model_type == "det":
        f_model = env.evaluate_solution_model 
    elif dynamics_model_type == "prob":
        f_model = env.evaluate_solution_model_ensemble 
        
    # initialize replay buffer
    replay_buffer = SimpleReplayBuffer(
        max_replay_buffer_size=1000000,
        observation_dim=obs_dim,
        action_dim=action_dim,
        env_info_sizes=dict(),
    )

    log_dir = (args.log_dir)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    mbqd = ModelBasedQD(args.dim_map, dim_x,
                        f_real, f_model,
                        surrogate_model, surrogate_model_trainer,
                        dynamics_model, dynamics_model_trainer,
                        replay_buffer, 
                        n_niches=args.n_niches,
                        params=px, log_dir=log_dir)

    mbqd.compute(num_cores_set=args.num_cores, max_evals=args.max_evals)

    save_name = args.save_name

    print("Saving torch model")
    ptu.save_model(mbqd.dynamics_model, 'models/'+save_name+'.pth')
    print("Successfully saved torch model")

    # mbqd.dynamics_model.eval()
    # # Load transitions
    # transitions_light_train = np.load('transitions_10_01_01_0.npy')
    # transitions_light_test = np.load('transitions_light_test.npy')
    #
    # transitions_mid = np.load('transitions_mid.npy')
    # transitions_heavy = np.load('transitions_heavy.npy')
    #
    # with torch.no_grad():
    #     for _ in range(10):
    #
    #         print("-----------------------------------------")
    #         likelihood_light_train = evaluate_transition_likelihood(transitions_light_train, 1000, mbqd.dynamics_model)
    #         print("Light Train:", likelihood_light_train)
    #
    #         likelihood_light_test = evaluate_transition_likelihood(transitions_light_test, 1000, mbqd.dynamics_model)
    #         print("Light Test:", likelihood_light_test)
    #
    #         likelihood_mid = evaluate_transition_likelihood(transitions_mid, 1000, mbqd.dynamics_model)
    #         print("Mid Test:", likelihood_mid)
    #
    #         likelihood_heavy = evaluate_transition_likelihood(transitions_heavy, 1000, mbqd.dynamics_model)
    #         print("Heavy Test:", likelihood_heavy)
    #




        # # Retrieve random sample of transitions
        # states_light, actions_light, next_states_light = get_random_sample_from_transitions(transitions_data_light, 1000, 48, 18)
        # states_heavy, actions_heavy, next_states_heavy = get_random_sample_from_transitions(transitions_data_heavy, 1000, 48, 18)
        #
        #
        # states_actions_array_light = np.concatenate((states_light, actions_light), axis=1)
        # states_actions_tensor_light = torch.tensor(states_actions_array_light).float()
        #
        # states_actions_array_heavy = np.concatenate((states_heavy, actions_heavy), axis=1)
        # states_actions_tensor_heavy = torch.tensor(states_actions_array_heavy).float()
        #
        # # Calling the dynamics model with 2 sample state action pairs; this yields the model predicted distribution
        # # samples, mean, logstd = mbqd.dynamics_model.forward(torch.nn.Parameter(torch.tensor([[-4.29982639e-10, 1.96232476e-09, 1.11453299e-08, -5.42446867e-09,
        # #                               1.80567388e-09, -1.40505213e-02, 1.74212540e-15, -2.26103712e-11,
        # #                               2.61300635e-11, -4.49956241e-16, -2.21574682e-11, 2.63495374e-11,
        # #                               5.15193700e-16, -2.28658587e-11, 2.62275462e-11, -1.96428467e-15,
        # #                               -2.25240193e-11, 2.60015434e-11, -4.63886259e-16, -2.21578555e-11,
        # #                               2.63474853e-11, 8.00803811e-16, -2.28855947e-11, 2.62785049e-11,
        # #                               2.56878681e-10, 2.50426359e-09, 1.25299829e-10, 2.71612164e-10,
        # #                               -1.79752598e-11, 5.03735204e-06, -1.99526963e-14, 4.55343455e-11,
        # #                               -4.21084271e-11, 6.60828992e-15, -8.51115952e-11, 8.15887523e-11,
        # #                               -4.13509474e-15, 4.89911341e-11, -4.50304242e-11, 1.93695864e-14,
        # #                               3.38275851e-11, -3.12088288e-11, 2.79418788e-15, -6.94027513e-11,
        # #                               6.65769205e-11, -6.30055873e-15, 3.67447994e-11, -3.36958311e-11,
        # #                               3.75270946e-01, 3.37914545e-01, 3.37914545e-01, 1.25381390e-01,
        # #                               -7.18869682e-01, -7.18869682e-01, -1.90682903e-01, -4.26521225e-02,
        # #                               -4.26521225e-02, 3.28282425e-01, 5.87112986e-01, 5.87112986e-01,
        # #                               -3.70667218e-01, -4.48065730e-01, -4.48065730e-01, 6.11226405e-02,
        # #                               7.25500199e-01, 7.25500199e-01], [-4.29982639e-10, 1.96232476e-09, 1.11453299e-08, -5.42446867e-09,
        # #                               1.80567388e-09, -1.40505213e-02, 1.74212540e-15, -2.26103712e-11,
        # #                               2.61300635e-11, -4.49956241e-16, -2.21574682e-11, 2.63495374e-11,
        # #                               5.15193700e-16, -2.28658587e-11, 2.62275462e-11, -1.96428467e-15,
        # #                               -2.25240193e-11, 2.60015434e-11, -4.63886259e-16, -2.21578555e-11,
        # #                               2.63474853e-11, 8.00803811e-16, -2.28855947e-11, 2.62785049e-11,
        # #                               2.56878681e-10, 2.50426359e-09, 1.25299829e-10, 2.71612164e-10,
        # #                               -1.79752598e-11, 5.03735204e-06, -1.99526963e-14, 4.55343455e-11,
        # #                               -4.21084271e-11, 6.60828992e-15, -8.51115952e-11, 8.15887523e-11,
        # #                               -4.13509474e-15, 4.89911341e-11, -4.50304242e-11, 1.93695864e-14,
        # #                               3.38275851e-11, -3.12088288e-11, 2.79418788e-15, -6.94027513e-11,
        # #                               6.65769205e-11, -6.30055873e-15, 3.67447994e-11, -3.36958311e-11,
        # #                               3.75270946e-01, 3.37914545e-01, 3.37914545e-01, 1.25381390e-01,
        # #                               -7.18869682e-01, -7.18869682e-01, -1.90682903e-01, -4.26521225e-02,
        # #                               -4.26521225e-02, 3.28282425e-01, 5.87112986e-01, 5.87112986e-01,
        # #                               -3.70667218e-01, -4.48065730e-01, -4.48065730e-01, 6.11226405e-02,
        # #                               7.25500199e-01, 7.25500199e-01]])), deterministic=False, return_dist=True)

        # samples_light, mean_light, logstd_light = mbqd.dynamics_model.forward(torch.nn.Parameter(states_actions_tensor_light), deterministic=False, return_dist=True)
        # samples_heavy, mean_heavy, logstd_heavy = mbqd.dynamics_model.forward(torch.nn.Parameter(states_actions_tensor_heavy), deterministic=False, return_dist=True)
        #
        #
        # # print("Mean", np.shape(mean), 'LogStd', np.shape(logstd))
        # # Compare observed next_state_residual with dynamics model predicted to see if in new environment
        #
        # # Now retrieve the log likelihood of observing the actual next state observations
        # next_state_deltas_light = next_states_light - states_light
        # next_state_deltas_heavy = next_states_heavy - states_heavy
        #
        # next_state_delta_observations_light = torch.tensor([next_state_deltas_light]).float()
        # next_state_delta_observations_heavy = torch.tensor([next_state_deltas_heavy]).float()
        #
        # # next_state_observations = torch.tensor([[[3.14704460e-03, -2.45940741e-03, -2.15887032e-02, -7.12400901e-04,
        # #    -7.21014294e-04, -1.00108792e-02,  4.74218214e-02,  4.99999999e-02,
        # #     4.99999999e-02,  3.99101361e-02, -5.00000000e-02, -5.00000000e-02,
        # #    -5.00000000e-02, -1.35765922e-02, -1.35765922e-02,  4.69886174e-02,
        # #     4.99999999e-02,  4.99999999e-02, -5.00000000e-02, -5.00000000e-02,
        # #    -5.00000000e-02,  1.94559407e-02,  5.00000000e-02,  5.00000000e-02,
        # #     3.02836940e-01, -2.32912920e-01, -2.62434388e+00, -6.29547819e-02,
        # #    -7.90046036e-02,  4.04727138e-01,  5.00000000e+00,  5.00000000e+00,
        # #     5.00000001e+00,  3.99101361e+00, -5.00000000e+00, -5.00000000e+00,
        # #    -5.00000000e+00, -1.35765922e+00, -1.35765923e+00,  5.00000000e+00,
        # #     5.00000000e+00,  5.00000000e+00, -5.00000000e+00, -5.00000000e+00,
        # #    -5.00000000e+00,  1.94559407e+00,  5.00000000e+00,  5.00000000e+00], [3.14704460e-03, -2.45940741e-03, -2.15887032e-02, -7.12400901e-04,
        # #    -7.21014294e-04, -1.00108792e-02,  4.74218214e-02,  4.99999999e-02,
        # #     4.99999999e-02,  3.99101361e-02, -5.00000000e-02, -5.00000000e-02,
        # #    -5.00000000e-02, -1.35765922e-02, -1.35765922e-02,  4.69886174e-02,
        # #     4.99999999e-02,  4.99999999e-02, -5.00000000e-02, -5.00000000e-02,
        # #    -5.00000000e-02,  1.94559407e-02,  5.00000000e-02,  5.00000000e-02,
        # #     3.02836940e-01, -2.32912920e-01, -2.62434388e+00, -6.29547819e-02,
        # #    -7.90046036e-02,  4.04727138e-01,  5.00000000e+00,  5.00000000e+00,
        # #     5.00000001e+00,  3.99101361e+00, -5.00000000e+00, -5.00000000e+00,
        # #    -5.00000000e+00, -1.35765922e+00, -1.35765923e+00,  5.00000000e+00,
        # #     5.00000000e+00,  5.00000000e+00, -5.00000000e+00, -5.00000000e+00,
        # #    -5.00000000e+00,  1.94559407e+00,  5.00000000e+00,  5.00000000e+00]]])
        # #
        # likelihoods_per_model_light = gaussian_pdf(next_state_delta_observations_light.detach().numpy(), mean_light.detach().numpy(), np.exp(logstd_light.detach().numpy()))
        # likelihoods_per_model_heavy = gaussian_pdf(next_state_delta_observations_heavy.detach().numpy(), mean_heavy.detach().numpy(), np.exp(logstd_heavy.detach().numpy()))
        #
        #
        # log_likelihoods_light = get_log_likelihoods(likelihoods_per_model_light, reduction='mean')
        # print("Likelihoods per model light:", log_likelihoods_light)
        #
        # log_likelihoods_heavy = get_log_likelihoods(likelihoods_per_model_heavy, reduction='mean')
        # print("Likelihoods per model heavy:", log_likelihoods_heavy)

        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #-----------------Type of QD---------------------#
    # options are 'cvt', 'grid' and 'unstructured'
    parser.add_argument("--qd_type", type=str, default="unstructured")
    
    #---------------CPU usage-------------------#
    parser.add_argument("--num_cores", type=int, default=8)
    
    #-----------Store results + analysis-----------#
    parser.add_argument("--log_dir", type=str)
    
    #-----------QD params for cvt or GRID---------------#
    # ONLY NEEDED FOR CVT OR GRID MAP ELITES - not needed for unstructured archive
    parser.add_argument("--dim_map", default=2, type=int) # Dim of behaviour descriptor
    parser.add_argument("--grid_shape", default=[100,100], type=list) # num discretizat
    parser.add_argument("--n_niches", default=3000, type=int)

    #----------population params--------#
    parser.add_argument("--b_size", default=200, type=int) # For parralellization - 
    parser.add_argument("--dump_period", default=5000, type=int) 
    parser.add_argument("--max_evals", default=400, type=int) # max number of evaluation
    parser.add_argument("--selector", default="uniform", type=str)
    parser.add_argument("--mutation", default="iso_dd", type=str)

    parser.add_argument("--save_name", default="standard", type=str)
    parser.add_argument("--env_id", default=0, type=int)

    
    args = parser.parse_args()
    
    main(args)

#  python3 run_scripts/hexapod_omni_daqd_main.py --log_dir results/experiment_0 --dump_period 6000 --max_evals 5000 --save_name experiment_0_vanilla_daqd_env0

#  python3 run_scripts/hexapod_omni_daqd_main.py --log_dir results/discard --dump_period 6000 --max_evals 800 --save_name model_env_0 --env_id 0