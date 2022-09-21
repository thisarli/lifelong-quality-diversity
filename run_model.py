import numpy as np
import torch

import src.torch.pytorch_util as ptu

from src.map_elites.llqd_utils import evaluate_transition_likelihood, get_dynamics_model

dynamics_model, _ = get_dynamics_model("prob")
model_path = "models/20_05_0_0.pth"
dynamics_model = ptu.load_model(dynamics_model, model_path)

dynamics_model.eval()
# Load transitions
transitions_light_train = np.load('transitions_light_train.npy')
transitions_light_test = np.load('transitions_light_test.npy')

transitions_mid = np.load('transitions_mid.npy')
transitions_heavy = np.load('transitions_20_05_0_0.npy')

with torch.no_grad():
    for _ in range(10):
        print("-----------------------------------------")
        likelihood_light_train = evaluate_transition_likelihood(transitions_light_train, 1000, dynamics_model)
        print("Light Train:", likelihood_light_train)

        likelihood_light_test = evaluate_transition_likelihood(transitions_light_test, 1000, dynamics_model)
        print("Light Test:", likelihood_light_test)

        likelihood_mid = evaluate_transition_likelihood(transitions_mid, 1000, dynamics_model)
        print("Mid Test:", likelihood_mid)

        likelihood_heavy = evaluate_transition_likelihood(transitions_heavy, 1000, dynamics_model)
        print("Heavy Test:", likelihood_heavy)

