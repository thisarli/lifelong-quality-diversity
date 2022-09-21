# Generating a random control and visualising it

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.envs.hexapod_dart.hexapod_env import HexapodEnv
from src.map_elites.llqd_utils import convert_states_actions_to_transitions, get_all_model_likelihoods, switch_env

env = HexapodEnv(dynamics_model=None,
                     render=False,
                     record_state_action=True,
                     ctrl_freq=100)

switch_env(0)

robot = env.init_robot()

# ctrl = [0.81030291, 0.45003475, 0.72721476, 0.44745025, 0.08874925, 0.67788591,
#  0.6622528,  0.38641183, 0.42289183, 0.94803084, 0.50470234, 0.57417548,
#  0.71014953, 0.91637582, 0.40512358, 0.97583656, 0.09130369, 0.0327558,
#  0.73088445, 0.166759,   0.71456177, 0.60875743, 0.62830832, 0.22145531,
#  0.76847295, 0.96027865, 0.79063379, 0.86092458, 0.22155352, 0.35859206,
#  0.42873686, 0.47116352, 0.63732902, 0.19752514, 0.83203628, 0.30700056]

# # Generating random controls and evaluating in specified env
# ctrl = [np.random.rand() for i in range(36)]
#
# # Shape of ctrl: list of 36
# # Shape of states_recorded: (n, 48), where n is the number of states visited in the seconds of sim
# # Shape of acctions_recoreded: (n, 18), where n is the number of actions taken in the seconds  of sim
# final_pos, states_recorded, actions_recorded = env.simulate(ctrl, 3, robot, True) # True to render
#
# # print(np.shape(states_recorded))
# # print(np.shape(actions_recorded))
# # 3 second simulations (like use in env.evaluate_solution provide 299 state, action, next_state transition points)
#
# simulation_transitions = convert_states_actions_to_transitions(states_recorded, actions_recorded)
# likelihoods_dict = get_all_model_likelihoods('models', simulation_transitions)
# print(likelihoods_dict)



# Get histogram of likelihoods for 50 controls for three models; different to llqd where not done by control, but all transitions from all controls grouped together

model_0_likelihoods = []
model_1_likelihoods = []
model_2_likelihoods = []

for _ in range(100):

    ctrl = [np.random.rand() for i in range(36)]

    final_pos, states_recorded, actions_recorded = env.simulate(ctrl, 3, robot, False)
    simulation_transitions = convert_states_actions_to_transitions(states_recorded, actions_recorded)

    likelihoods_dict = get_all_model_likelihoods('models', simulation_transitions)

    # Uncomment if want to visualise controls
    model_0_likelihoods.append(likelihoods_dict['model_env_0.pth'].mean())  # takes mean across ensemble members
    model_1_likelihoods.append(likelihoods_dict['model_env_1.pth'].mean())
    model_2_likelihoods.append(likelihoods_dict['model_env_2.pth'].mean())

    # Uncomment if want to visualise individual transitions
    # model_0_likelihoods.append(likelihoods_dict['model_env_0.pth'][0,:])
    # model_1_likelihoods.append(likelihoods_dict['model_env_1.pth'][0, :])
    # model_2_likelihoods.append(likelihoods_dict['model_env_2.pth'][0, :])

print('model_0', model_0_likelihoods)
print('model_1', model_1_likelihoods)
print('model_2', model_2_likelihoods)

numpy_data = np.hstack([np.reshape(model_0_likelihoods, (-1, 1)), np.reshape(model_1_likelihoods, (-1, 1)), np.reshape(model_2_likelihoods, (-1, 1))])
pd_data = pd.DataFrame(numpy_data, columns=["Model 0", "Model 1", "Model 2"])

print(pd_data)

# sns.color_palette("viridis", as_cmap=True)
# sns.set(rc={'figure.figsize':(8,5)})
plt.rcParams.update({'font.size': 14})
plt.rcParams['patch.edgecolor'] = 'none'
plt.rcParams["patch.force_edgecolor"] = False
plt.figure(figsize=(10, 5))
ax = sns.histplot(data=pd_data, bins=np.arange(-200, 0, 10), kde=True, alpha=0.4)  # bins=np.arange(-400, 0, 20)
plt.xlabel('Log likelihood per control', fontsize=14)
plt.ylabel('Count', fontsize=14)
# plt.legend(loc="upper left", fontsize=14)

plt.tight_layout()
plt.show()
#
# plt.style.use('seaborn-deep')
#
# plt.hist([model_0_likelihoods, model_1_likelihoods, model_2_likelihoods], bins = 'auto', label=['model_0', 'model_1', 'model_2'])
# plt.legend(loc='upper right')
# plt.show()



