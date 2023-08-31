import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import differential_evolution
import os
import csv
from itertools import product


def simulate_model(params, *args):
    # simulation params
    n_trials = args[2]
    n_steps = 2000

    # init params
    v_mean = params[0]
    v_sd = params[1]
    a = params[2]
    z = params[3]
    ndt = params[4]

    t = np.arange(0, n_steps, 1)
    evidence = np.zeros((n_trials, n_steps))
    evidence[:, 0] = z

    r_choice = np.zeros(n_trials)
    r_time = np.zeros(n_trials)

    for i in range(n_trials):
        for j in range(1, n_steps):
            if j > ndt:
                # accumulate new evidence
                new_evidence = np.random.normal(v_mean, v_sd)
                evidence[i, j] = evidence[i, j - 1] + new_evidence

                # make a response if a threshold is crossed
                if evidence[i, j] > a:
                    r_choice[i] = 0
                    r_time[i] = j
                    break

                if evidence[i, j] < -a:
                    r_choice[i] = 1
                    r_time[i] = j
                    break

    #fig, ax = plt.subplots(1, 1, squeeze=False)
    # ax[0, 0].plot([0, n_steps], [a, a], '--k')
    # ax[0, 0].plot([0, n_steps], [-a, -a], '--k')
    # ax[0, 0].plot(t, evidence[0, :])
    # plt.show()

    return r_choice, r_time


def obj_func(params, *args):
    """
    This function should accept a given set of parameters, simulate the model
    with those parameters, and then compare the result of that simulation with
    your objective (e.g., a person)
    """

    # simnulate the model
    r_choice_pred, rt_pred = simulate_model(params, *args)

    r_choice_obs = args[0]
    rt_obs = args[1] * 1000

    # NOTE: should cost factor in correct vs incorrect
    sse_choice = (np.nanmean(r_choice_pred) - np.nanmean(r_choice_obs)) ** 2
    sse_rt = (np.nanmean(rt_pred) - np.nanmean(rt_obs)) ** 2

    # NOTE: left in just to think about if we ever want trial order to matter
    # compare against objective
    # sse_choice = np.sum((r_choice_pred - r_choice_obs)**2)
    # sse_rt = np.sum((rt_pred - rt_obs)**2)

    sse = sse_choice + sse_rt

    return sse


def fit_model():

    dir_data = "../data_Stroop_PRmanipulation/"
    d = pd.read_csv(dir_data + "2response_trimmed_combined.csv")

    loop_output = []
    dir_output = "../fits/"
    csv_file_path = os.path.join(
        dir_output, "loop_output.csv"
    )  # name of the loop output file and path

    header_row = [
        "participant",
        "congruency",
        "v_mean",
        "v_sd",
        "a",
        "z",
        "ndt",
        "value",
    ]  # create the heading names

    # Writing loop_output to a CSV file. This first one creates the files and add the header rows above
    with open(csv_file_path, "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header_row)

        for s in d["participant"].unique():
            for c in d["Congruency"].unique():
                print(s, c)
    
                dd = d.loc[(d["participant"] == s) & (d["Congruency"] == c)]
    
                n_trials = dd.shape[0]
    
                r_choice_obj = dd["Correct"].to_numpy()
                r_time_obj = dd["RT"].to_numpy()
                args = (r_choice_obj, r_time_obj, n_trials)
    
                # TODO: choose reasonable values for bounds
                bounds = [(0, 1), (0, 5), (0, 1000), (-a * 0.9, a * 0.9), (0, 500)]
    
                # search parameter space and find the best set of params
                result = differential_evolution(
                    obj_func,
                    bounds,
                    args,
                    tol=1e1,
                    maxiter=10,
                    polish=False,
                    # updating='deferred',
                    # workers=-1,
                    disp=True,
                )
    
                print(result.x, result.fun)
                loop_output = np.concatenate((result["x"], [result["fun"]]))
    
                # Writing loop_output to a CSV file
                with open(csv_file_path, "a", newline="") as csvfile:
                    csv_writer = csv.writer(csvfile)
                    
    
                    # Writing data into each row with multiple columns
                    for i in range(1):
                        row_data = [
                            s,
                            c,
                            loop_output[0],
                            loop_output[1],
                            loop_output[2],
                            loop_output[3],
                            loop_output[4],
                            loop_output[5],
                        ]
                        csv_writer.writerow(row_data)


def inspect_fits():
    d = pd.read_csv("../fits/loop_output.csv")

    dd = d.groupby(["congruency"])[["v_mean", "v_sd", "a", "z", "ndt"]].mean()

    fig, ax = plt.subplots(1, 5, squeeze=False)
    sns.barplot(data=dd, x=dd.index, y="v_mean", ax=ax[0, 0])
    sns.barplot(data=dd, x=dd.index, y="v_sd", ax=ax[0, 1])
    sns.barplot(data=dd, x=dd.index, y="a", ax=ax[0, 2])
    sns.barplot(data=dd, x=dd.index, y="z", ax=ax[0, 3])
    sns.barplot(data=dd, x=dd.index, y="ndt", ax=ax[0, 4])
    plt.show()


def fit_validate():
    # set up a grid of parameter values we are interested in exploring
    a = 100
    
    bounds = [
        np.arange(0, 1, 0.1),
        np.arange(0, 5, 1),
        np.arange(0, 1000, 1),
        np.arange(-a * 0.9, a * 0.9, 1),
        np.arange(0, 500, 10)]

    param_combinations = list(product(*bounds))

    loop_output = []
    dir_output = "../sim/"
    csv_file_path = os.path.join(
        dir_output, "sim_output.csv"
    )  # name of the loop output file and path

    header_row = [
        "v_mean_recovered",
        "v_sd_recovered",
        "a_recovered",
        "z_recovered",
        "ndt_recovered",
        "value_recovered",
        "v_mean",
        "v_sd",
        "a",
        "z",
        "ndt",
    ]  # create the heading names
    
    #I just want it to create the header_row one time...
    with open(csv_file_path, "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header_row)

    # Writing loop_output to a CSV file. This first one creates the files and add the header rows above
    with open(csv_file_path, "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
    
        for p in param_combinations:
    
            print(p)
            n_trials = 150
            args = ([],[],n_trials)
            r_choice_pred, rt_pred = simulate_model(p, *args)
            args = (r_choice_pred, rt_pred, n_trials)
       
            bounds = [(0, 1), (0, 5), (0, 1000), (-a * 0.9, a * 0.9), (0, 500)]
    
             # search parameter space and find the best set of params
            result = differential_evolution(
                 obj_func,
                 bounds,
                 args,
                 tol=1e1,
                 maxiter=10,
                 polish=False,
                 # updating='deferred',
                 # workers=-1,
                 disp=True,)  
             
             
            print(result.x, result.fun)
            loop_output = np.concatenate((result["x"], [result["fun"]]))
    
            # Writing loop_output to a CSV file
            with open(csv_file_path, "a", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)
    
                # Writing data into each row with multiple columns
                for i in range(1):
                    row_data = [
                        loop_output[0],
                        loop_output[1],
                        loop_output[2],
                        loop_output[3],
                        loop_output[4],
                        loop_output[5],p[0],p[1],p[2],p[3],p[4]
                    ]
                    csv_writer.writerow(row_data)
             


fit_validate()
# fit_model()
# inspect_fits()

# v_mean = 0.1  # drift rate
# v_sd = 2.0  # drift rate
# a = 100.0  # threshold
# z = 0.5  # starting point
# ndt = 200  # nondecision time
# params = (v_mean, v_sd, a, z, ndt)
# r_choice_obj, r_time_obj = simulate_model(params)
