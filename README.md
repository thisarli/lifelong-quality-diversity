# Lifelong Quality-Diversity (LLQD)
Gitlab repository for Lifelong Quality-Diversity (LLQD) by Tilman Hisarli. Presentation slides describing the algorithm can be found in slides.pdf file in repository.

LLQD is a sample efficient model-based QD implementation that can effectively handle non-stationary environments by producing specialised repertoires of behaviours online for each distinct environment the robot encounters. LLQD uses probabilistic dynamics models to detect previously seen or newly encountered environments based on the robot's recent state and action trajectories. No assumptions about the distribution of environments the robot may encounter in the future need to be made.

## Running the code
1. Install Singularity locally

2. Clone the project repository

3. Navigate to the singularity folder in your terminal, and run 
```
./start_container.sh
```

4. Navigate to the root of the project folder

5. To perform LLQD for a random sequence of environments, run 
```
# python3 run_scripts/hexapod_omni_llqd_main.py --log_dir results/test_main_script --dump_period 500000 --epsilon 0.5 --min_model_add 200 --random_init_batch 200 --high_difficulty 0
```
The results will be stored in `results/test_main_script`.

3. To visualize archives and simulating their solutions on an interactive chart, run:
```
python3 vis_repertoire_hexapod.py --filename results/test_main_script/0/archive_200.dat --plot_type grid

```
By specifying "0/archive_200.dat", you choose to plot the archive for the first LLQD object, after 200 real evaluations.


## Additions to the codebase

The LLQD code is built on top of Lim et al's Dynamics Aware Quality Diversity repository. Our main additions to the codebase include

1. The LLQD script: the main script to perform LLQD on some randomized sequence of environments, containing the complex logic for the environment detection/handling and transfer learning features of LLQD. (`run_scripts/hexapod_omni_llqd_main.py`)
2. The LLQD class: the definition of the LLQD class, which stores all the relevant objects (e.g. dynamics model, buffers, archives) and contains the logic that splits evaluation of candidate controls from the assignment of trajectories to the buffer and training of the relevant model. (`src/map_elites/llqd.py`)
3. The LLQD utilities: contain all the utility functions needed to perform LLQD, such as computing log likelihoods of transitions, or creating randomized environment sequences and scoring. (`src/map_elites/llqd_utils.py`)
4. The Environment utilities: consist of the alternative environments used in our analysis, as well as the complex implementation logic to switch environments during script execution. (throughout `src/envs/hexapod_dart/`)
5. The Visualisation scripts: scripts to visualise various aspects of the LLQD algorithm, such as the hexapod gait, the distribution of the log likelihoods of the trajectories for the various models, or the interpolation plots for the development of Coverage and QD Score versus number of real evaluations. (`run_scripts/experiment_2_plots.py`, `random_visualisation.py`)
6. The Experiment scripts: scripts for all conducted experiments, both for LLQD and its DAQD comparison, ranging from the main evaluation of LLQD vs DAQD, to various ablation studies including the assessment of LLQD's transfer learning capabilities. (throughout `run_scripts/`)
7. The Results: contains the results for all experiments for verification purposes. (throughout `results/`)


## Code references
- The LLQD code is built on top of Lim et al's Dynamics Aware Quality Diversity Code <https://github.com/adaptive-intelligent-robotics/Dynamics-Aware_Quality-Diversity> repository.
- DAQD in turn is built on the QD implementation from the <https://github.com/resibots/pymap_elites> repository.
- As in DAQD, the code for the probabilistic dynamics models is adapted from the <https://github.com/kzl/lifelong_rl> repository.
- Simulations throughout the project are done using the [DART](https://dartsim.github.io) physics simulator and [RobotDART](https://github.com/resibots/robot_dart) python wrapper.

