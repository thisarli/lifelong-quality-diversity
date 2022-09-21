import numpy as np
import torch
import matplotlib.pyplot as plt
import src.torch.pytorch_util as ptu
from multiprocessing import get_context
import time, os

from src.envs.hexapod_dart.hexapod_env import HexapodEnv
from src.map_elites.llqd_utils import convert_states_actions_to_transitions, get_all_model_likelihoods, switch_env, \
    evaluate_likelihood_for_slist, get_dynamics_model

from src.map_elites import common as cm
from src.map_elites.llqd import evaluate_

def random_archive_init(to_evaluate, f_real):
    """
    Generates random genotypes and appends them to a passed in list of genotypes to evaluate, with the
    model evaluation function
    """
    for i in range(0, 50):
        x = np.random.uniform(low=0.0, high=1.0, size=36)
        to_evaluate += [(x, f_real)]

    return to_evaluate

def parallel_eval(evaluate_function, to_evaluate):

    s_list = map(evaluate_function, to_evaluate)
    return list(s_list)

# from src.envs.hexapod_dart.hexapod_env import HexapodEnv

env = HexapodEnv(dynamics_model=None,
                     render=False,
                     record_state_action=True,
                     ctrl_freq=100)

f_real = env.evaluate_solution

switch_env(0)
# robot = env.init_robot()

to_evaluate = []
# for i in range(100):
#     ctrl = [np.random.rand() for i in range(36)]
#     to_evaluate.append((ctrl, f_real))

to_evaluate = random_archive_init(to_evaluate, f_real)  # init real archive by generating random
# genotypes and putting them into the to_evaluate list along with the real eval function
# to_evaluate = self.random_archive_init_model(to_evaluate) # init synthetic archive
s_list = parallel_eval(evaluate_, to_evaluate)

print(len(s_list))

dynamics_model, _ = get_dynamics_model("prob")
for model_name in ['model_env_0.pth', 'model_env_1.pth', 'model_env_2.pth']:
    model_path = 'models/' + model_name
    dynamics_model = ptu.load_model(dynamics_model, model_path)
    likelihood = evaluate_likelihood_for_slist(s_list, dynamics_model)
    print(model_name, likelihood)