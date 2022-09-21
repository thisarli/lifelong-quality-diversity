#! /usr/bin/env python

import os, sys
import time
import math
import numpy as np
import multiprocessing

from sklearn.neighbors import KDTree

from src.map_elites import common as cm
from src.map_elites import unstructured_container, cvt
from src.map_elites import model_condition_utils

import torch
import src.torch.pytorch_util as ptu

import cma

from multiprocessing import get_context

from os import listdir

from src.map_elites.llqd_utils import evaluate_holdout_likelihoods


def evaluate_(t):
    # evaluate a single vector (x) with a function f and return a species
    # evaluate z with function f - z is the genotype and f is the evalution function
    # t is the tuple from the to_evaluate list

    """
    This is a helper function that applies some evaluation funcction (e.g. f_real) to the first element of the
    tuple, which is a genotype. f_real for instaancce returns fitness, desc, obs_traj, act_traj; these are then
    used to create aand return a species member with genotype z, descriptor desc and so on.

    t: tuple containing (genotype, evaluation function)
    """

    z, f = t
    fit, desc, obs_traj, act_traj = f(z) 
    
    # because it somehow returns a list in a list (have to keep checking sometimes)
    desc = desc[0]  # important - if not it fails the KDtree for cvt and grid map elites
    disagr = 0  # no disagreement for real evaluation - but need to put to save archive
    # return a species object (containing genotype, descriptor and fitness)
    return cm.Species(z, desc, fit, obs_traj=obs_traj, act_traj=act_traj, model_dis=disagr)

def model_evaluate_(t):
    # same as the above evaluate but this takes in the disagreement also
    # - useful if you want to make use disargeement value
    # needs two types because no such thing as disagreemnt for real eval
    z, f = t
    fit, desc, obs_traj, act_traj, disagr = f(z) 
    
    # becasue it somehow returns a list in a list (have to keep checking sometimes)
    desc = desc[0] # important - if not it fails the KDtree for cvt and grid map elites
    
    # return a species object (containing genotype, descriptor and fitness)
    return cm.Species(z, desc, fit, obs_traj=obs_traj, act_traj=act_traj, model_dis=disagr)


class ModelBasedLLQD:
    def __init__(self,
                 dim_map, dim_x,
                 f_real, f_model,
                 model, model_trainer,
                 dynamics_model, dynamics_model_trainer,
                 replay_buffer,
                 n_niches=1000,
                 params=cm.default_params,
                 bins=None,
                 log_dir='./',):

        torch.set_num_threads(24)

        # init evaluation counters
        self.n_evals = 0
        self.gen = 0
        self.b_evals = 0
        self.n_model_evals = 0
        self.evals_since_last_train = 0

        # init likelihood trackers
        self.mean_likelihood = None
        self.std_likelihood = None

        # init evaluation lists
        self.to_evaluate = None
        self.to_model_evaluate = None
        self.s_list = None

        # init counters for model stats
        self.true_pos = None
        self.false_pos = None
        self.false_neg = None
        self.true_neg = None

        # init temporary archive used for stats
        self.tmp_archive = None
        self.add_list_model = None

        # init timers
        self.gen_start_time = None

        self.qd_type = params["type"]    # QD type - grid, cvt, unstructured
        self.dim_map = dim_map           # number of BD dimensions  
        self.dim_x = dim_x               # gemotype size (number of genotype dim)
        self.n_niches = n_niches         # estimated total population in archive
        self.bins = bins                 # grid shape - only for grid map elites
        self.params = params

        # 2 eval functions
        # 1 for real eval, 1 for model eval (imagination)
        self.f_real = f_real  # f_real is passed in as an argument
        
        if params["model_variant"] == "dynamics":
            self.f_model = f_model  # f_model is passed in as an argument
            print("Dynamics Model Variant")
        elif params["model_variant"] == "direct":
            self.f_model = self.evaluate_solution_surrogate_model 
            print("Direct Model Variant")
            
        # Model and Model trainer init -
        # initialize the classes outside this class and pass in
        self.model = model  # refers to the direct qd surrogate model
        self.model_trainer = model_trainer  # direct qd surrogate trainer

        self.dynamics_model = dynamics_model  # passed in as an argument
        self.dynamics_model_trainer = dynamics_model_trainer  # passed in as an argument

        self.replay_buffer = replay_buffer  # passed in as an argument
        self.all_real_evals = []
        
        # Init logging directory and log file
        self.log_dir = log_dir
        log_filename = self.log_dir + '/log_file.dat'
        self.log_file = open(log_filename, 'w')
    
        # path and filename to save model
        self.save_model_path = self.log_dir + '/trained_model.pth'
        
        # Initialise time logging
        if params['log_time_stats']:
            time_stats_filename = self.log_dir + '/time_log_file.dat'
            self.time_log_file = open(time_stats_filename, 'w')
            self.gen_time = 0
            self.model_eval_time = 0
            self.eval_time = 0
            self.model_train_time = 0 
        
        # Init cvt and grid - only if cvt and grid map elites used
        if (self.qd_type == "cvt") or (self.qd_type == "grid"):
            c = []
            if self.qd_type == "cvt":
                c = cm.cvt(self.n_niches,
                           self.dim_map,params['cvt_samples'],
                           params['cvt_use_cache'])
            else:
                c = cm.grid_centroids(self.bins)

            self.kdt = KDTree(c, leaf_size=30, metric='euclidean')
            cm.__write_centroids(c)

        if (self.qd_type == "cvt") or (self.qd_type=="grid"):
            self.archive = {}  # init archive as dict (empty)
            self.model_archive = {}
        elif self.qd_type == "unstructured":
            self.archive = []  # init archive as list
            self.model_archive = []

    def random_archive_init(self, to_evaluate):
        """
        Generates random genotypes and appends them to a passed in list of genotypes to evaluate, with the
        real evaluation function
        """
        for i in range(0, self.params['random_init_batch']):
            x = np.random.uniform(low=self.params['min'], high=self.params['max'], size=self.dim_x)
            to_evaluate += [(x, self.f_real)]

        return to_evaluate

    def random_archive_init_model(self, to_evaluate):
        """
        Generates random genotypes and appends them to a passed in list of genotypes to evaluate, with the
        model evaluation function
        """
        for i in range(0, self.params['random_init_batch']):
            x = np.random.uniform(low=self.params['min'], high=self.params['max'], size=self.dim_x)
            to_evaluate += [(x, self.f_model)]
        
        return to_evaluate

    def select_and_mutate(self, to_evaluate, archive, f, params, variation_operator=cm.variation, batch=False):
        """
        Randomly generates pairs of genotypes from archive, mutates them, returns a new genotype for each pair
        and adds this genotype to the to_evaluate list (together with the passed in eval function)
        
        archive: is a list or dict containing the species which contains the genotype, descriptor, etc.
        to_evaluate: list of genotypes (and their eval functions) to be evaluated; is passed in and updated
        """

        if (self.qd_type=="cvt") or (self.qd_type=="grid"):
            keys = list(archive.keys()) # in these cases the archive is a dict
        elif (self.qd_type=="unstructured"):
            keys = archive # in this case the archive is a list
                    
        # we select all the parents at the same time because randint is slow
        # generates an array of batch_size numbers between 0 and num of genotypes in the archive - 1
        # So really creates two lists of random indices, that are later used to mutate the genotypes
        rand1 = np.random.randint(len(keys), size=self.params['batch_size'])
        rand2 = np.random.randint(len(keys), size=self.params['batch_size'])
            
        for n in range(0, params['batch_size']):
            # parent selection - mutation operators like iso_dd/sbx require 2 gen parents
            if (self.qd_type == "cvt") or (self.qd_type=="grid"):
                x = archive[keys[rand1[n]]] # first random species member
                y = archive[keys[rand2[n]]] # second random species member
            elif (self.qd_type == "unstructured"):                    
                x = archive[rand1[n]]
                y = archive[rand2[n]]
                
            # copy & add variation (Species.x is the member's genotype)
            z = variation_operator(x.x, y.x, params) # applies some variation to the two members' genotypes
            # then spits out a new genotype z

            if batch:
                to_evaluate += [z]
            else: 
                to_evaluate += [(z, f)]

        return to_evaluate # list of genotypes (and their eval functions) to be evaluated
    
    def addition_condition(self, s_list, archive, params):
        """
        Checks if species in the s_list qualify for being added to the archive.
        Appends to the archive in here, and returns a list of species to be added
        
        s_list: list of Species members
        archive: is a list or dict containing the species which contains the genotype, descriptor, etc.

        returns
        add_list: list of species members
        """
        add_list = [] # list of species solutions that were added
        discard_list = []
        for s in s_list: # for each solution check if gets added to archive
            if self.qd_type == "unstructured":
                success = unstructured_container.add_to_archive(s, archive, params)
            else:
                success = cvt.add_to_archive(s, s.desc, self.archive, self.kdt)
            if success: # if success, then add it to the list of species to be added
                add_list.append(s)
            else: # if not, then add it to the list of species to be discarded
                discard_list.append(s) #not important for alogrithm but to collect stats
                
        return archive, add_list, discard_list

    def model_condition(self, s_list, archive, params):
        """
        Checks if species in the s_list qualify for being addded to the archive.
        Does not add to the archive in here yet, but returns a list of species to be added
        
        s_list: list of Species members
        archive: is a list or dict containing the species which contains the genotype, descriptor, etc.
        """
        add_list = [] # list of solutions that are worth evaluating in real life
        discard_list = []
        for s in s_list:
            success = model_condition_utils.add_to_archive(s, archive, params) # adds to archive if
            # more diverse or on average fitter than nearest neighbors by some margin specified in params  
            if success:
                add_list.append(s)
            else:
                discard_list.append(s) # not important for algorithm but to collect stats
                
        return archive, add_list, discard_list

    # model based lifelong map-elites algorithm
    def compute_s_list(self, num_cores_set, max_evals=1e5, params=None):

        if params is None:
            params = self.params

        # setup the parallel processing pool
        if num_cores_set == 0:
            num_cores = multiprocessing.cpu_count()  # use all cores
        else:
            num_cores = num_cores_set
            
        # pool = multiprocessing.Pool(num_cores)
        pool = get_context("spawn").Pool(num_cores)
        # pool = ThreadPool(num_cores)
        
        # gen = 0  # generation
        # self.n_evals = 0  # number of evaluations since the beginning
        # b_evals = 0  # number evaluation since the last dump
        # n_model_evals = 0  # number of evals done by model

        #evals_since_last_train = 0
        print("################# Starting QD algorithm #################")

        # lists of individuals we want to evaluate (list of tuples) for this gen
        # each entry in the list is a tuple of the genotype and the evaluation function
        self.to_evaluate = []
        self.to_model_evaluate = []

        ## initialize counter for model stats; was reset every loop
        self.true_pos = 0
        self.false_pos = 0
        self.false_neg = 0
        self.true_neg = 0

        ## intialize for time related stats ##
        self.gen_start_time = time.time()
        self.model_train_time = 0

        # random initialization of archive - start up
        # if num of species members in archive is less than min proportion x num of niches to be filled ...
        # If you move to new model, we try to add previous s_list to archive; if resulting add_list to small
        # we then enter the archive init below, which will evaluate random genotypes and try to add them, so adds
        # degree of exploration for the new archive; note that this does not override the archive but just creates
        # a more exploratory s_list; could have impact on transfer learning if we evaluate archive members from previous
        if len(self.archive) <= params['random_init']*self.n_niches:
            print("Not enough solutions in archive. Generating random genotypes to evaluate")
            self.to_evaluate = self.random_archive_init(self.to_evaluate)  # init real archive by generating random
            # genotypes and putting them into the to_evaluate list along with the real eval function
            # to_evaluate = self.random_archive_init_model(to_evaluate) # init synthetic archive

            start = time.time()
            self.s_list = cm.parallel_eval(evaluate_, self.to_evaluate, pool, params)  # applies evaluate_ function to
            # to_evaluate list which holds randomly generated genotypes and real eval function;
            # evaluate_ returns a species object for each genotype with info about bd, fitness, trajectories;
            # so s_list is then a list of species members for each genotype that was in to_evaluate list
            # s_list = cm.parallel_eval(model_evaluate_, to_evaluate, pool, params) # init model
            self.eval_time = time.time() - start
            # # Now we generate our list of species to add to the archive
            # self.archive, add_list, _ = self.addition_condition(s_list, self.archive, params)


        else: # we already have a fair number of members in the archive
            # variation/selection loop - select ind from archive to evolve
            self.model_archive = self.archive.copy()  # copy over the real archive to the model archive
            self.tmp_archive = self.archive.copy()  # tmp archive for stats of negatives
            #print("Real archive size start: ",len(self.archive))
            #print("Model archive size start: ",len(self.model_archive))

            '''
            # For serial or parallel evaluations
            to_model_evaluate = self.select_and_mutate(to_model_evaluate, self.model_archive, self.f_model, params)
            start = time.time()
            s_list_model = cm.parallel_eval(evaluate_, to_model_evaluate, pool, params)
            self.model_eval_time = time.time() - start
            #s_list_model = self.serial_eval(evaluate_, to_model_evaluate, params)
            
            ### MODEL CONDITIONS ###
            self.model_archive, add_list_model, discard_list_model = self.model_condition(s_list_model, self.model_archive, params)
            #self.model_archive, add_list_model = self.addition_condition(s_list_model, self.model_archive, params)
            #print("Model novel list: ", len(add_list_model))
            #print("Model discard list: ", len(discard_list_model))
            '''

            # uniform selection of emitter - other options is UCB
            # Spits out the solutions that should be evaluated in real (sim) environment (TH)
            # add_list_model are the species members that made it into model archive
            # to_model_evaluate is the list of tuples of genotypes and their eval funcs that were model
            # evaluated (ie some of these did not make it into model archive); latter is only used later
            # to count the number of evaluations and stop the algo
            # emitter generates 100 model archive species members
            emitter = params["emitter_selection"] #np.random.randint(3)
            if emitter == 0:
                self.add_list_model, self.to_model_evaluate = self.random_model_emitter(self.to_model_evaluate, pool, params)
            elif emitter == 1:
                self.add_list_model, self.to_model_evaluate = self.optimizing_emitter(self.to_model_evaluate, pool, params, self.gen)
            elif emitter == 2:
                self.add_list_model, self.to_model_evaluate = self.random_walk_emitter(self.to_model_evaluate, pool, params, self.gen)
            elif emitter == 3:
                self.add_list_model, self.to_model_evaluate = self.model_disagr_emitter(self.to_model_evaluate, pool, params, self.gen)


            ### REAL EVALUATIONS ###
            # if model finds novel solutions - evaluate in real setting
            if len(self.add_list_model) > 0:
                start = time.time()
                self.to_evaluate = []
                for z in self.add_list_model: # those species that qualified according to f_model
                    self.to_evaluate += [(z.x, self.f_real)] # add tuple of genotype and real eval function
                self.s_list = cm.parallel_eval(evaluate_, self.to_evaluate, pool, params)
                # to_evaluate list holds tuples of genotypes and real eval functions of promising solutions
                # as per the model.
                # evaluate_ returns a species object for each genotype with info about bd,
                # fitness, trajectories;
                # so s_list is then a list of species members for each genotype that was in to_evaluate list
                self.eval_time = time.time() - start

        # count evals
        # self.n_evals += len(self.to_evaluate)  # total number of real evals

        # self.b_evals += len(self.to_evaluate)  # number of real evals since last dump
        # remember that to_evaluate holds tuples of genotypes and real eval function for every species
        # member that the model evaluation said was interesting, and was thus evaluated in real life
        self.n_model_evals += len(self.to_model_evaluate)  # total number of model evals
        print(f"In total, for this model we have performed {self.n_model_evals} model evaluations")

        # Split here, as we need to check if model is appropriate before we add
        # to archive, as it could be a different environment
        return


    def compute_more_stuff(self, num_cores_set, max_evals=1e5, params=None):
        if params is None:
            params = self.params

        # setup the parallel processing pool
        if num_cores_set == 0:
            num_cores = multiprocessing.cpu_count()  # use all cores
        else:
            num_cores = num_cores_set

        # pool = multiprocessing.Pool(num_cores)
        pool = get_context("spawn").Pool(num_cores)

        self.n_evals += len(self.s_list)
        print(f"For this model, In total we have performed {self.n_evals} real evaluations")

        self.b_evals += len(self.s_list)  # number of real evals since last dump for this model; here, as should be model specific

        self.archive, add_list, discard_list = self.addition_condition(self.s_list, self.archive, params)
        self.true_pos = len(add_list)  # number of solutions the model evaluated as being added to archive
        # which are then truly added to the real archive
        self.false_pos = len(discard_list)
        print(f"We have {self.true_pos} true positives (added) and {self.false_pos} false positives (discarded)")

        ## FOR STATISTICS - EVALUATE MODEL DISCARD PILE ## NOT USED
        # this checks if the species members that the model evaluation deemed not good enough to be
        # added to the archive really aren't (when using the real eval function)
        if params["log_model_stats"]:
            if len(discard_list_model) > 0:
                to_evaluate_stats = []
                for z in discard_list_model:
                    to_evaluate_stats += [(z.x, self.f_real)]  # tuple of discarded species genotype and
                    # real eval function
                s_list_stats = cm.parallel_eval(evaluate_, to_evaluate_stats, pool, params)
                tmp_archive, add_list_stats, discard_list_stats = self.addition_condition(s_list_stats, tmp_archive, params)

                false_neg = len(add_list_stats)
                true_neg = len(discard_list_stats)
                #print("False negative: ", false_neg)
                #print("True negative: ", true_neg)



        ####### UPDATE MODEL - MODEL LEARNING ############
        print("Entering model training section")
        self.evals_since_last_train += len(self.s_list)  # len(self.to_evaluate) add number of real evaluated
        # solutions for this model; used to make sure we have sufficient new datapoints to warrant new training
        print(f"Since the last training, we have evaluated {self.evals_since_last_train} species instances")

        print("Adding s_list trajectories to buffer...")
        self.add_sa_to_buffer(self.s_list, self.replay_buffer)  # retrieves the state and action trajectories
        print("Successfully added the s_list trajectories to the replay buffer")
        # from the species that the model qualified as interesting, and were then evaluated in real
        # life using f_real; state action next_state pairs are added to the buffer

        print("The replay buffer size is now: ", self.replay_buffer._size)

        # Shd always work for first (ie 0th gen in new model, as 0/anything has remainder 0
        if (((self.gen % params["train_freq"]) == 0) or (self.evals_since_last_train > params["evals_per_train"])) and params["train_model_on"]:
            # s_list are solutions/species that have been evaluated in the real setting
            print("Training model...")
            start = time.time()
            if params["model_variant"] == "dynamics":
                # FOR DYNAMICS MODEL
                torch.set_num_threads(24)
                self.dynamics_model_trainer.train_from_buffer(self.replay_buffer,
                                                              holdout_pct=0.1,
                                                              max_grad_steps=10000)

            elif params["model_variant"] == "direct":
                # FOR DIRECT QD SURROGATE MODEL
                # track all real evals - keep only always 50k of latest data
                # RAM problems if too many is kept
                #self.all_real_evals = s_list + self.all_real_evals
                #if len(self.all_real_evals) > 50000:
                #    self.all_real_evals = self.all_real_evals[:50000]
                #dataset = self.get_gen_bd_fit_dataset(self.all_real_evals)
                dataset = self.get_gen_bd_fit_dataset(self.archive)
                self.model_trainer.train_from_dataset(dataset,
                                                      holdout_pct=0.2,
                                                      max_grad_steps=10000)

            self.model_train_time = time.time() - start
            print("Training finished. Model train time: ", self.model_train_time)
            self.evals_since_last_train = 0 # after training reset to 0

            x_test = self.dynamics_model_trainer.x_test
            y_test = self.dynamics_model_trainer.y_test
            self.mean_likelihood, self.std_likelihood = evaluate_holdout_likelihoods(x_test, y_test, self.dynamics_model)
            print('Mean likelihoods of holdout set:', self.mean_likelihood)
            print('Std of likelihoods of holdout set:', self.std_likelihood)

        self.gen += 1  # generations
        # # remember that to_evaluate holds tuples of genotypes and real eval function for every species
        # # member that the model evaluation said was interesting, and was thus evaluated in real life
            
        #print("n_evals: ", n_evals)
        print("Number of evals since last dump (b_evals): ", self.b_evals)

        # write archive during dump period
        if self.b_evals >= params['dump_period'] and params['dump_period'] != -1:
            save_start = time.time()
            print("Entering dump...")

            # write archive
            #print("[{}/{}]".format(n_evals, int(max_evals)), end=" ", flush=True)
            print("[{}/{}]".format(self.n_evals, int(max_evals)))

            # Write fitness, BD and genotype of every species member in archive to a file
            cm.save_archive(self.archive, self.n_evals, params, self.log_dir)

            self.b_evals = 0  # dump happened so reset number of evals since last dump to 0

            # Save models
            #ptu.save_model(self.model, self.save_model_path)
            print("Saving torch model")
            ptu.save_model(self.dynamics_model, self.save_model_path)
            print("Done saving torch model")

            save_end = time.time() - save_start
            print("Save archive and model time: ", save_end)



        # write log -  write log every generation
        if (self.qd_type == "cvt") or (self.qd_type == "grid"):
            fit_list = np.array([x.fitness for x in self.archive.values()])
            self.log_file.write("{} {} {} {} {} {} {} {} {} {}\n".format(self.gen,
                                     self.n_evals,  # num of real evals
                                     self.n_model_evals,
                                     len(self.archive.keys()),  # num of species members in archive
                                     fit_list.max(),  # highest fitness of species members in archive
                                     np.sum(fit_list),  # total sum of fitness of archive members
                                     np.mean(fit_list),
                                     np.median(fit_list),
                                     np.percentile(fit_list, 5),
                                     np.percentile(fit_list, 95)))

        elif self.qd_type == "unstructured":
            fit_list = np.array([x.fitness for x in self.archive])
            self.log_file.write("{} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(
                self.gen,
                self.n_evals,
                self.n_model_evals,
                len(self.archive),
                fit_list.max(),
                np.sum(fit_list),
                np.mean(fit_list),
                np.median(fit_list),
                np.percentile(fit_list, 5),
                np.percentile(fit_list, 95),
                self.true_pos,
                self.false_pos,
                self.false_neg,
                self.true_neg))
                
        self.log_file.flush()  # writes to file but does not close stream

        cm.save_archive(self.archive, self.n_evals, self.params, self.log_dir)

        self.gen_time = time.time() - self.gen_start_time
        if params['log_time_stats']:
            self.time_log_file.write("{} {} {} {} {} {} {}\n".format(self.gen,
                                                               self.gen_time,
                                                               self.model_eval_time,
                                                               self.eval_time,
                                                               self.model_train_time,
                                                               len(self.to_evaluate),
                                                               len(self.to_model_evaluate),))
            self.time_log_file.flush()

        return


        # print("==========================================")
        # print("End of QD algorithm - saving final archive")
        # cm.save_archive(self.archive, self.n_evals, params, self.log_dir)
        #
        # # retrieve the transition dataset (TH)
        # transitions = self.replay_buffer.get_transitions()
        # # print(transitions)
        #
        # # save the transitions dataset; these come from the trajectories of the species members that
        # # the model predicted were interesting and were thus later evaluated in real life
        # with open('transitions_20_05_0_0.npy', 'wb') as f:
        #     np.save(f, transitions)


    ##################### Emitters ##############################
    def random_model_emitter(self, to_model_evaluate, pool, params):
        """
        Makes random candidates from the model archive, model evaluates them to turn them into species members,
        checks if these species members make it into the model archive, and if so adds them to the list of
        species members to be added to the model archive

        The input to_model_evaluate is a list of species members, but not sure why passed in

        When that list exceeds 100, then finish
        """

        start = time.time()
        add_list_model_final = []
        all_model_eval = []
        max_num_add_list = 100
        if 'min_model_add' in params:
            max_num_add_list = params["min_model_add"]
        while len(add_list_model_final) < max_num_add_list:
        #for i in range(5000): # 600 generations (500 gens = 100,000 evals)
            to_model_evaluate=[] # will hold tuples of genotypes and their eval functions
            # Make new genotypes from random combinations of members of archive and add those
            # and their eval function to the to_model_evaluate list
            to_model_evaluate = self.select_and_mutate(to_model_evaluate, self.model_archive, self.f_model, params) # f_model is passed into the DAQD class, so defined outsiide
            if params["model_variant"]=="dynamics":
                #s_list_model = cm.parallel_eval(evaluate_, to_model_evaluate, pool, params)

                # Generate a list of species members by applying eval function for each tuple in the
                # to_model_evaluate list to the genotype in the tuple; these members now contain model
                # evaluated fitness, descriptor, disagreement as attributes
                s_list_model = cm.parallel_eval(model_evaluate_, to_model_evaluate, pool, params)
            elif params["model_variant"]=="direct":
                s_list_model = self.serial_eval(evaluate_, to_model_evaluate, params)
            
            #self.model_archive, add_list_model, discard_list_model = self.model_condition(s_list_model, self.model_archive, params)

            # Generate the add_list for the model evaluated species members (ie which of the candidates 
            # will later be added to the model archive)
            self.model_archive, add_list_model, discard_list_model = self.addition_condition(s_list_model, self.model_archive, params)

            add_list_model_final += add_list_model
            all_model_eval += to_model_evaluate # count all inds evaluated by model
            #print("to model eval length: ",len(to_model_evaluate)) 
            #print("s list length: ",len(s_list_model)) 
            #print("model list length: ",len(add_list_model_final)) 
            #print("all model evals length: ", len(all_model_eval))

            #if i%20 ==0: 
            #    cm.save_archive(self.model_archive, "model_gen_"+str(i), params, self.log_dir)
            #    print("Model gen: ", i)
            #    print("Model archive size: ", len(self.model_archive))
            
        self.model_eval_time = time.time() - start         
        return add_list_model_final, all_model_eval # list of species that made it into model archive and 
        # list of genotypes and eval functions that were evaluated

    def optimizing_emitter(self, to_model_evaluate, pool, params, gen):
        '''
        uses CMA - no mutations
        '''
        start = time.time()
        add_list_model_final = []
        all_model_eval = []

        rand1 = np.random.randint(len(self.model_archive))
        mean_init = (self.model_archive[rand1]).x
        sigma_init = 0.01
        popsize = 50
        max_iterations = 100
        es = cma.CMAEvolutionStrategy(mean_init,
                                      sigma_init,
                                      {'popsize': popsize,
                                       'bounds': [0,1]})
        
        for i in range(max_iterations):
        #while not es.stop():    
            to_model_evaluate = []
            solutions = es.ask()
            for sol in solutions:
                to_model_evaluate += [(sol, self.f_model)]
            s_list_model = cm.parallel_eval(model_evaluate_, to_model_evaluate, pool, params)

            self.model_archive, add_list_model, discard_list_model = self.model_condition(s_list_model, self.model_archive, params)
            add_list_model_final += add_list_model
            all_model_eval += to_model_evaluate # count all inds evaluated by model
            #print("model list length: ",len(add_list_model_final)) 
            #print("all model evals length: ", len(all_model_eval))

            # convert maximize to minimize
            # for optimizing emitter fitness of CMAES is fitness of the ind
            reward_list = []
            for s in s_list_model:
                reward_list.append(s.fitness)

            cost_arr = -np.array(reward_list)
            es.tell(solutions, list(cost_arr))
            #es.disp()
            
            #if i%10==0:
            #    cm.save_archive(self.model_archive, str(gen)+"_"+str(i), params, self.log_dir)
            #i +=1

                        
        self.model_eval_time = time.time() - start
        
        return add_list_model_final, all_model_eval

    def random_walk_emitter(self, to_model_evaluate, pool, params, gen):
        start = time.time()
        add_list_model_final = []
        all_model_eval = []

        # sample an inidivudal from the archive to init cmaes
        rand1 = np.random.randint(len(self.model_archive))
        ind_init = self.model_archive[rand1]
        mean_init = ind_init.x
        sigma_init = 0.01
        popsize = 50
        max_iterations = 100
        es = cma.CMAEvolutionStrategy(mean_init,
                                      sigma_init,
                                      {'popsize': popsize,
                                       'bounds': [0,1]})

        # sample random vector/direction in the BD space to compute CMAES fitness on
        # BD space is 2 dim
        desc_init = ind_init.desc
        target_dir = np.random.uniform(-1,1,size=2)

        
        for i in range(max_iterations):
        #i = 0 
        #while not es.stop():
            to_model_evaluate = []
            solutions = es.ask()
            for sol in solutions:
                to_model_evaluate += [(sol, self.f_model)]
            s_list_model = cm.parallel_eval(model_evaluate_, to_model_evaluate, pool, params)
            #s_list_model = self.serial_eval(model_evaluate_, to_model_evaluate, params)

            #self.model_archive, add_list_model, discard_list_model = self.model_condition(s_list_model, self.model_archive, params)
            self.model_archive, add_list_model, discard_list_model = self.addition_condition(s_list_model, self.model_archive, params)
            add_list_model_final += add_list_model
            all_model_eval += to_model_evaluate # count all inds evaluated by model
            #print("model list length: ",len(add_list_model_final)) 
            #print("all model evals length: ", len(all_model_eval))

            # convert maximize to minimize
            # for random walk emitter, fitnes of CMAES is the magnitude of vector in the target_direction
            reward_list = []
            for s in s_list_model:
                s_dir = s.desc - desc_init
                comp_proj = (np.dot(s_dir, target_dir))/np.linalg.norm(target_dir)
                reward_list.append(comp_proj)

            cost_arr = -np.array(reward_list)
            es.tell(solutions, list(cost_arr))
            #es.disp()

            if i%10==0:
                cm.save_archive(self.model_archive, str(gen)+"_"+str(i), params, self.log_dir)
            #i +=1
            
            self.model_eval_time = time.time() - start
            #print("model_eval_time", self.model_eval_time)
            
        return add_list_model_final, all_model_eval

    def improvement_emitter(self):
    
        return 1
    
    def model_disagr_emitter(self, to_model_evaluate, pool, params, gen):
        '''
        emitter which maximises model dissagreement
        CMAES fitness function is disagreement
        '''
        start = time.time()
        add_list_model_final = []
        all_model_eval = []

        # sample an inidivudal from the archive to init cmaes
        rand1 = np.random.randint(len(self.model_archive))
        ind_init = self.model_archive[rand1]
        mean_init = ind_init.x
        sigma_init = 0.01
        popsize = 50
        max_iterations = 100
        es = cma.CMAEvolutionStrategy(mean_init,
                                      sigma_init,
                                      {'popsize': popsize,
                                       'bounds': [0,1]})
        for i in range(max_iterations):
        #i = 0 
        #while not es.stop():
            to_model_evaluate = []
            solutions = es.ask()
            for sol in solutions:
                to_model_evaluate += [(sol, self.f_model)]
            s_list_model = cm.parallel_eval(model_evaluate_, to_model_evaluate, pool, params)
            #s_list_model = self.serial_eval(model_evaluate_, to_model_evaluate, params)

            #self.model_archive, add_list_model, discard_list_model = self.model_condition(s_list_model, self.model_archive, params)
            self.model_archive, add_list_model, discard_list_model = self.addition_condition(s_list_model, self.model_archive, params)
            add_list_model_final += add_list_model
            all_model_eval += to_model_evaluate # count all inds evaluated by model
            #print("model list length: ",len(add_list_model_final)) 
            #print("all model evals length: ", len(all_model_eval))

            # convert maximize to minimize
            # for disagr_emitter, fitness of CMA-ES is model disagreement
            reward_list = []
            for s in s_list_model:
                s_dis = s.model_dis
                reward_list.append(s_dis)

            cost_arr = -np.array(reward_list)
            es.tell(solutions, list(cost_arr))
            es.disp()

            if i%10==0:
                cm.save_archive(self.model_archive, str(gen)+"_"+str(i), params, self.log_dir)
            #i +=1
            
        self.model_eval_time = time.time() - start    
        return add_list_model_final, all_model_eval

    
    ################## Custom functions for Model Based QD ####################
    def serial_eval(self, evaluate_function, to_evaluate, params):
        s_list = map(evaluate_function, to_evaluate)
        return list(s_list)
    
    def evaluate_solution_surrogate_model(self, gen):
        #torch.set_num_threads(1)
        with torch.no_grad():
            x = ptu.from_numpy(gen)
            x = x.view(1, -1)
        
            pred = self.model.output_pred(x)            
            fitness = pred[0,0]
            desc = [pred[0,1:3]]
            
        obs_traj = None
        act_traj = None
        
        return fitness, desc, obs_traj, act_traj
    
    def evaluate_batch(self, z_batch):
        '''
        # For batch evaluations of NN models
        #to_model_evaluate = self.select_and_mutate(to_model_evaluate, self.model_archive, self.f_model, params, batch=True)
        #s_list_model = self.evaluate_batch(to_model_evaluate)
        '''
        f = self.f_model
        fit_batch, desc_batch, obs_traj_b, act_traj_b = f(z_batch) 
        # this is a batch eval functions so returns everything as [batch_size, dim]
        s_list = []
        for i in range(len(z_batch)):
            z = z_batch[i]
            desc = desc_batch[i] 
            desc_ground = desc
            fit = fit_batch[i]
            obs_traj = obs_traj_b[i]
            act_traj = act_traj_b[i]
            
            s = cm.Species(z, desc, desc_ground, fit, obs_traj, act_traj)
            s_list.append(s) 
        
        return s_list
    
    def get_gen_bd_fit_dataset(self, s_list):
        x = []
        y = []
        for s in s_list:
            x.append(s.x)
            y.append(np.concatenate((s.fitness, s.desc), axis=None))

        x = np.array(x)
        y = np.array(y)
        dataset = [x,y]

        return dataset
    
    def add_sa_to_buffer(self, s_list, replay_buffer):
        """
        s_list: list of species members

        """
        # s_a_ns_list = []
        for sol in s_list:
            s = sol.obs_traj[:-1] # state trajectory excluding the last, as last won't have next state
            a = sol.act_traj[:-1] # action trajectory excluding the last
            ns = sol.obs_traj[1:] # state trajectory excluding the first

            reward = 0
            done = 0
            info = {}
            for i in range(len(s)): # each trajectory consists of multiple s, a, next_state pairs
                replay_buffer.add_sample(s[i], a[i], reward, done, ns[i], info)
                # s_a_ns_list.append([s[i], a[i], reward, done, ns[i], info])
        # int(s_a_ns_list)
        return 1
