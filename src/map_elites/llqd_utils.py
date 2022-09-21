# To hold the utility functions used in llqd
import shutil
from os import listdir
import numpy as np
import torch
import src.torch.pytorch_util as ptu

from src.models.dynamics_models.deterministic_model import DeterministicDynModel
from src.models.dynamics_models.probabilistic_ensemble import ProbabilisticEnsemble


def gaussian_pdf(datapoint, mean, variance):
    """
    likelihood for each dimension in next state, acc. to pdf of Gaussian specified by mean and variance

    mean: np.array of shape (ensemble_size, num_samples, next_state_dims)
    variance: np.array of shape (ensemble_size, num_samples, next_state_dims)

    returns: np.array of shape (ensemble_size, num_samples, next_state_dims)
    """
    return (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-((datapoint - mean) ** 2) / (2 * variance))


def get_log_likelihoods(likelihoods_per_model, reduction='mean'):
    """
    Retrieves some summary measure (e.g. mean across samples) of the sum of log likelihoods (across all
    next_state dimensions), for each ensemble member

    likelihoods_per_model: array containing likelihoods
        np.array of shape (ensemble_size, num_samples, next_state_dims)

    reduction: sum, median, or mean

    returns: aggregate likelihoods per ensemble member
        np.array of shape (ensemble_size,)
    """
    if reduction == 'sum':
        return np.log(likelihoods_per_model).sum(axis=2).sum(axis=1)
    elif reduction == 'median':
        return np.median(np.log(likelihoods_per_model).sum(axis=2), axis=1)
    elif reduction == 'mean':
        return np.mean(np.log(likelihoods_per_model).sum(axis=2), axis=1)
    else:
        raise (NotImplementedError)


def get_log_likelihoods_mean_std(likelihoods_per_model, reduction='mean'):
    """
    Retrieves some summary measure (e.g. mean across samples) and std of the sum of log likelihoods (across all
    next_state dimensions), for each ensemble member. Mean and std across samples are adjusted to ignore -inf values.

    likelihoods_per_model: array containing likelihoods
        np.array of shape (ensemble_size, num_samples, next_state_dims)

    reduction: median, mean, or no reduction to retrieve likelihoods per transition

    returns: aggregate likelihoods per ensemble member, std of likelihoods per ensemble member
        Two np.arrays of shape (ensemble_size,) each
    """
    sum_of_log_likelihoods = np.log(likelihoods_per_model).sum(axis=2)  # shape (ensemble_size, num_samples)

    if reduction == 'median':
        return np.median(sum_of_log_likelihoods, axis=1), np.ma.masked_invalid(sum_of_log_likelihoods).std(axis=1).data
    elif reduction == 'mean':
        return np.ma.masked_invalid(sum_of_log_likelihoods).mean(axis=1).data, np.ma.masked_invalid(
            sum_of_log_likelihoods).std(axis=1).data
    elif reduction == 'no':
        return np.ma.masked_invalid(sum_of_log_likelihoods).data, None
    else:
        raise (NotImplementedError)


def get_random_sample_from_transitions(transitions, num_samples, state_dims, action_dims):
    """
    Retrieves a random sample of transitions (state, action, next_state) from a transitions Dataset, which is retrieved
    from buffer

    transitions: dataset containing state, action, reward(0), done(0), next_state
        np.array of shape(num_transitions, state_dims + action_dims + 2 + next_state_dims)

    num_samples: number of transition samples we want to randomly draw (-1: use all samples)
        int

    state_dims: dimensionality of state(48)
        int

    action_dims: dimensionality of action(18)

    returns: states, actions, next_states
        np.arrays of shape(num_samples, state_dims or action_dims)
    """
    if num_samples == -1:
        random_sample = transitions
    else:
        rand_indices = [np.random.randint(len(transitions)) for i in range(num_samples)]
        random_sample = transitions[rand_indices]
    states = random_sample[:, 0:state_dims]
    actions = random_sample[:, state_dims: state_dims + action_dims]
    next_states = random_sample[:, state_dims + action_dims + 2:]
    return states, actions, next_states


def evaluate_transition_likelihood(transitions_data, num_samples, model, inf_adjusted=False):
    """
    Aggregation function to retrieve aggregated log likelihoods per ensemble member for a set of transitions

    transitions: dataset containing state, action, reward(0), done(0), next_state
        np.array of shape(num_transitions, state_dims + action_dims + 2 + next_state_dims)

    num_samples: number of transition samples we want to randomly draw
        int

    model: trained probabilistic ensemble model

    inf_adjusted: if True, adjusts log_likelihoods to ignore -inf values, so that mean and std are not distorted

    returns: aggregate likelihoods per ensemble member
        np.array of shape (ensemble_size,)
    """
    states, actions, next_states = get_random_sample_from_transitions(transitions_data, num_samples, 48, 18)
    states_actions_array = np.concatenate((states, actions), axis=1)
    states_actions_tensor = torch.tensor(states_actions_array).float()
    model.eval()
    with torch.no_grad():
        samples, mean, logstd = model.forward(torch.nn.Parameter(states_actions_tensor), deterministic=False,
                                              return_dist=True)
    next_state_deltas = next_states - states
    next_state_delta_observations = torch.tensor([next_state_deltas]).float()
    likelihoods_per_model = gaussian_pdf(next_state_delta_observations.detach().numpy(), mean.detach().numpy(), np.exp(
        2 * logstd.detach().numpy()))  # should be 2*logstd for var, but empirically works better without
    if inf_adjusted is False:
        log_likelihoods = get_log_likelihoods(likelihoods_per_model, reduction='median')
    else:
        log_likelihoods, _ = get_log_likelihoods_mean_std(likelihoods_per_model, reduction='mean')  #'no' to show likelihoods per transition
    return log_likelihoods


def evaluate_holdout_likelihoods(x_test, y_test, model):
    """
    Aggregation function to retrieve aggregated log likelihoods per ensemble member for a holdout set from ModelTrainer

    x_test: dataset containing state, action
        np.array of shape(num_transitions, state_dims + action_dims)

    y_test: dataset containing next state
        np.array of shape(num_transitions, state_dims)

    model: the model on which we evaluate the holdout set
        trained probabilistic ensemble model

    returns: mean likelihood and standard deviation across num_transitions. Adjusted to ignore -inf
        float, float
    """
    states_actions_tensor = torch.tensor(x_test).float()
    model.eval()
    with torch.no_grad():
        samples, mean, logstd = model.forward(torch.nn.Parameter(states_actions_tensor), deterministic=False,
                                              return_dist=True)
    next_state_delta_observations = torch.tensor([y_test]).float()
    likelihoods_per_model = gaussian_pdf(next_state_delta_observations.detach().numpy(), mean.detach().numpy(), np.exp(
        2 * logstd.detach().numpy()))  # should be 2*logstd for var, but empirically works better without
    log_likelihoods_mean, log_likelihoods_std = get_log_likelihoods_mean_std(likelihoods_per_model, reduction='mean')
    return log_likelihoods_mean.mean(), log_likelihoods_std.mean()


# Functions and imports to generate the likelihood of all models using the data of one sim


def convert_states_actions_to_transitions(states, actions):
    """
    Takes in states, actions, and converts them into transitions dataset
    This means adjusting states and actions to exclude the last observation,
    inserting zero rewards and done indicators, as well as generating next states
    states and actions would typically come from Species.obj_traj and Species.act_traj

    states: arrays of shape (num_obs, state_dims), usually (300, 48)
    actions: arrays of shape (num_obs, action_dims), usually (300, 18)

    transitions: dataset containing state, action, reward(0), done(0), next_state
        np.array of shape(num_transitions, state_dims + action_dims + 2 + next_state_dims)
    """
    states_usable = states[:-1]
    actions_usable = actions[:-1]
    next_states_usable = states[1:]
    rewards_dones = np.zeros((len(states_usable), 2))
    return np.hstack((states_usable, actions_usable, rewards_dones, next_states_usable))


def get_all_model_likelihoods(models_path, transitions):
    """
    Calculates the likelihoods for all models using all datapoints in transitions.

    models_path: path to models directory
        str

    transitions: dataset containing state, action, reward(0), done(0), next_state
        np.array of shape(num_transitions, state_dims + action_dims + 2 + next_state_dims)

    returns: dict containing for each available model the aggregated (across samples) likelihoods per ensembble
        member
    """
    likelihoods_dict = {}
    model_name_list = [f for f in listdir(models_path)]

    # Check if any model is available, if not exit
    if len(model_name_list) == 0:
        print("No models currently exist")
        return

    for model_name in model_name_list:
        dynamics_model, _ = get_dynamics_model("prob")
        model_path = models_path + '/' + model_name
        dynamics_model = ptu.load_model(dynamics_model, model_path)
        dynamics_model.eval()
        with torch.no_grad():
            print(f"-----------------{model_name}------------------------")
            likelihood = evaluate_transition_likelihood(transitions, -1, dynamics_model, True)
            print(likelihood.mean())
            likelihoods_dict[model_name] = likelihood

    return likelihoods_dict


def get_dynamics_model(dynamics_model_type):
    """
    Helper function to retrieve the dynamics model
    """
    obs_dim = 48
    action_dim = 18

    ## INIT MODEL ##
    if dynamics_model_type == "prob":
        from src.trainers.mbrl.mbrl import MBRLTrainer
        variant = dict(
            mbrl_kwargs=dict(
                ensemble_size=4,
                layer_size=500,
                learning_rate=1e-3,
                batch_size=512,
            )
        )
        M = variant['mbrl_kwargs']['layer_size']
        dynamics_model = ProbabilisticEnsemble(
            ensemble_size=variant['mbrl_kwargs']['ensemble_size'],
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=[M, M]
        )
        dynamics_model_trainer = MBRLTrainer(
            ensemble=dynamics_model,
            **variant['mbrl_kwargs'],
        )

        # ensemble somehow cant run in parallel evaluations
    elif dynamics_model_type == "det":
        from src.trainers.mbrl.mbrl_det import MBRLTrainer
        dynamics_model = DeterministicDynModel(obs_dim=obs_dim,
                                               action_dim=action_dim,
                                               hidden_size=500)
        dynamics_model_trainer = MBRLTrainer(
            model=dynamics_model,
            batch_size=512, )

    return dynamics_model, dynamics_model_trainer


def evaluate_likelihood_by_model_for_slist(s_list, models_path):
    """
    Evaluate the likelihood of observing the transitions from the s_list species members, for each available model
    Use this to check if still in same environment.

    Returns a dictionary with the models as keys, and values as a 1d array containing the median (across transitions) sum (across
    next_state dimensions) of log likelihoods, for each ensemble member (i.e. 4)

    Insert this function after the real evaluations have taken place; then use it to attribute datapoints

    models_path: usually 'models'

    """
    transition_list = []
    for sol in s_list:
        states = sol.obs_traj
        actions = sol.act_traj
        sol_transitions = convert_states_actions_to_transitions(states, actions)
        transition_list.append(sol_transitions)

    transitions = np.vstack(transition_list)

    likelihoods_dict = get_all_model_likelihoods(models_path, transitions)

    return likelihoods_dict


def evaluate_likelihood_for_slist(s_list, dynamics_model):
    """
    Evaluate the likelihood of observing the transitions from the s_list species members, for some model.
    Use this to check if still in same environment.

    Returns a float containing the mean (across transitions) sum (across
    next_state dimensions) of log likelihoods (adj to ignore -inf), averaged across ensemble members (i.e. 4).

    Use it to attribute datapoints to models.
    """
    transition_list = []
    for sol in s_list:
        states = sol.obs_traj
        actions = sol.act_traj
        sol_transitions = convert_states_actions_to_transitions(states, actions)
        transition_list.append(sol_transitions)

    transitions = np.vstack(transition_list)

    likelihood = evaluate_transition_likelihood(transitions, -1, dynamics_model, True)

    return likelihood.mean()


def make_random_int_list(list_length, num_envs, initialise=False):
    """
    Function to generate a random list of integers, used to later determine which environment is chosen.
    list_length: the length of the desired list, ie the number of environment changes
    num_envs: the number of environments we want to test
    initialise: if True, then the list will start with the environments in order, and random integers after
                This is useful to ensure that all envs are initialised and to keep the envs in order when we later
                check the list vs the list of identified environments in experiment 1.
    """
    if initialise is False:
        random_list = np.random.randint(num_envs, size=list_length).tolist()
        # random_list[1] = random_list[0]
        return random_list
    else:
        random_list = np.random.randint(num_envs, size=list_length - num_envs).tolist()
        starter = [i for i in range(num_envs)]
        return starter + random_list


def switch_env(env_id):
    if env_id == 0:
        shutil.copy2('src/envs/hexapod_dart/robot_model/alternative_models/hexapod_06791_0_0_0.urdf',
                     'src/envs/hexapod_dart/robot_model/hexapod_v2.urdf')
        shutil.copy2('src/envs/hexapod_dart/alternative_envs/even_floor.txt',
                     'src/envs/hexapod_dart/floor.txt')
        # shutil.copy2('src/envs/hexapod_dart/alternative_envs/hexapod_env_even.py',
        #              'src/envs/hexapod_dart/hexapod_env.py')
    elif env_id == 1:
        shutil.copy2('src/envs/hexapod_dart/robot_model/alternative_models/hexapod_10_-02_02_-03.urdf',
                     'src/envs/hexapod_dart/robot_model/hexapod_v2.urdf')
        shutil.copy2('src/envs/hexapod_dart/alternative_envs/slanted_floor_2.txt',
                     'src/envs/hexapod_dart/floor.txt')
        # shutil.copy2('src/envs/hexapod_dart/alternative_envs/hexapod_env_slanted_2.py',
        #              'src/envs/hexapod_dart/hexapod_env.py')
    elif env_id == 2:
        shutil.copy2('src/envs/hexapod_dart/robot_model/alternative_models/hexapod_06791_0_0_0.urdf',
                     'src/envs/hexapod_dart/robot_model/hexapod_v2.urdf')
        shutil.copy2('src/envs/hexapod_dart/alternative_envs/slanted_floor.txt',
                     'src/envs/hexapod_dart/floor.txt')
        # shutil.copy2('src/envs/hexapod_dart/alternative_envs/hexapod_env_slanted.py',
        #              'src/envs/hexapod_dart/hexapod_env.py')

    return


def get_strict_score(env_list, estimate):
    """
    Retrieves the mapping from identified env to dict, and whether the identified envs are correct.
    """
    map_dict = {}

    # Get the correct mapping
    for i in range(len(env_list)):
        real_env = env_list[i]
        est_env = estimate[i]
        if real_env not in map_dict.values():  # Check the env is not already mapped to
            map_dict.setdefault(est_env,
                                real_env)  # Check we are not overwriting an estimate -> env mapping where an estimate value has already been mapped

    # print(map_dict)
    scores = []

    for i in range(len(env_list)):
        if estimate[i] in map_dict.keys() and env_list[i] == map_dict[estimate[i]]:
            scores.append(1)
        else:
            scores.append(0)

    return map_dict, scores


def gen_random_sequence(length):
    """
  Generates a random sequence of environment estimates, that conforms to
  environments being built sequentially
  """
    sequence_list = []
    for i in range(length):
        if i == 0:
            sequence_list.append(0)
        else:
            sequence_list.append(np.random.randint(low=0, high=len(set(sequence_list)) + 1))
    return sequence_list


def random_strict_score():
    num_correct = []
    for i in range(100000):
        env_list = make_random_int_list(6, 3)
        visited = gen_random_sequence(6)
        corrects = 0
        for num in range(1, len(env_list)):
            if env_list[num] == visited[num]:
                corrects += 1
        num_correct.append(corrects)
    print('Mean correct (random)', np.mean(num_correct), '{:.1%}'.format(np.mean(num_correct) / (len(env_list) - 1)))
    print('Median correct (random)', np.median(num_correct),
          '{:.1%}'.format(np.median(num_correct) / (len(env_list) - 1)))
    print('25th percentile (random)', np.percentile(num_correct, 25),
          '{:.1%}'.format(np.percentile(num_correct, 25) / (len(env_list) - 1)))
    print('75th percentile (random)', np.percentile(num_correct, 75),
          '{:.1%}'.format(np.percentile(num_correct, 75) / (len(env_list) - 1)))


def make_switching_int_list(list_length, num_envs):
    """
    Make int list where next number is always different from previous number, ie always switching
    """
    random_list = np.random.randint(num_envs, size=1).tolist() # Generate first entry
    for i in range(1, list_length):
      choices = [k for k in range(num_envs)]
      choices.remove(random_list[i-1])
      random_list.append(int(np.random.choice(choices,1)))
    return random_list