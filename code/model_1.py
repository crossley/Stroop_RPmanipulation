import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution


def simulate_model(params):

    # simulation params
    n_trials = 100
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

            # TODO: is this okay?
            if j > ndt:

                # TODO: remind ourselves v_mean vs v_sd
                new_evidence = np.random.normal(v_mean, v_sd)
                evidence[i, j] = evidence[i, j - 1] + new_evidence

                # make a response if a threshold is crossed
                if evidence[i, j] > a:
                    r_choice[i] = 1
                    r_time[i] = j
                    break

                if evidence[i, j] < -a:
                    r_choice[i] = 2
                    r_time[i] = j
                    break

    # fig, ax = plt.subplots(1, 1, squeeze=False)
    # ax[0, 0].plot([0, n_steps], [a, a], '--k')
    # ax[0, 0].plot([0, n_steps], [-a, -a], '--k')
    # ax[0, 0].plot(t, evidence[0, :])
    # plt.show()

    return r_choice, r_time


def obj_func(params, *args):
    '''
    This function should accept a given set of parameters, simulate the model
    with those parameters, and then compare the result of that simulation with
    your objective (e.g., a person)
    '''

    # simnulate the model
    r_choice_pred, r_time_pred = simulate_model(params)

    # TODO: need trial-by-trial choice and rt data from a target participant
    r_choice_obs = args[0]
    r_time_obs = args[1]

    # compare against objective
    sse_choice = np.sum((r_choice_pred - r_choice_obs)**2)
    sse_time = np.sum((r_time_pred - r_time_obs)**2)

    sse = sse_choice + sse_time

    return sse


def fit_model():

    v_mean = 0.1  # drift rate
    v_sd = 2.0  # drift rate
    a = 100.0  # threshold
    z = 0.5  # starting point
    ndt = 200  # nondecision time

    params = (v_mean, v_sd, a, z, ndt)

    r_choice_obj, r_time_obj = simulate_model(params)
    args = (r_choice_obj, r_time_obj)

    # TODO: choose reasonable values for bounds
    bounds = [(0, 1), (0, 5), (0, 1000), (-a * 0.9, a * 0.9), (0, 500)]

    # search parameter space and find the best set of params
    result = differential_evolution(obj_func,
                                    bounds,
                                    args,
                                    tol=1e-5,
                                    disp=True)

    print(result.x, result.fun)


fit_model()

# v_mean = 0.1  # drift rate
# v_sd = 2.0  # drift rate
# a = 100.0  # threshold
# z = 0.5  # starting point
# ndt = 200  # nondecision time
# params = (v_mean, v_sd, a, z, ndt)
# r_choice_obj, r_time_obj = simulate_model(params)
