import numpy as np
import itertools
from scipy.optimize import linprog
import pandas as pd
import datetime
import scipy.stats as st
import math
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, DateFormatter
import numpy as np
import os
import numpy as np
import random
import matplotlib.dates as mdates
import time
import copy
from process_daily_data_controled_eval import process_daily_data
from prob_being_caught_controled_eval import prob_being_caught
from bayesian_OSSP_controled_eval import bayesian_OSSP
from bounded_rationality_M1_controled_eval import bounded_rationality_epsilon
from bounded_rationality_M2_controled_eval import bounded_rationality_beta
from bayesian_OSSP_controled_eval_qr import bayesian_OSSP_qr
from bounded_rationality_M1_controled_eval_qr import bounded_rationality_epsilon_qr
from bounded_rationality_M2_controled_eval_qr import bounded_rationality_beta_qr

np.random.seed(2021)
random.seed(a=2021, version=2)

ATTACK_TYPE_SAMPLE = 2
ALERT_TYPE = 7
PROB_SIMU_TIMES = 25000
ATTACK_TYPE_PROB = [1/7]*ALERT_TYPE
NUM_ALERT_COUNT = [196.09756098,  28.90243902, 140.97560976,   9.75609756, 25.43902439,  15.36585366,  42.7804878 ]

# NUM_ALERT_COUNT = [200] * ALERT_TYPE

ALPHA = 0.01    # the proportion of resources used to coping with repeated access requests.
# B = 50.0

# Beta_list = [[0.125, 0.125], [0.25, 0.25], [0.5, 0.5], [1.0, 1.0], [2.0, 2.0], [4.0, 4.0], [8.0, 8.0]]

#Epsilon_list = [[50.0, 50.0], [100.0, 100.0], [200.0, 200.0], [400.0, 400.0], [800.0, 800.0], [1600.0, 1600.0], [3200.0, 3200.0]]
Epsilon_list = [[0.0, 0.0], [50.0, 50.0], [100.0, 100.0], [150.0, 150.0], [200.0, 200.0], [250.0, 250.0], [300.0, 300.0], [350.0, 350.0], [400.0, 400.0], [450.0, 450.0], [500.0, 500.0], [550.0, 550.0], [600.0, 600.0], [650.0, 650.0], [700.0, 700.0], [750.0, 750.0],[800.0, 800.0], [850.0, 850.0], [900.0, 900.0], [950.0, 950.0], [1000.0, 1000.0], [1050.0, 1050.0], [1100.0, 1100.0], [1150.0, 1150.0], [1200.0, 1200.0], [1250.0, 1250.0], [1300.0, 1300.0], [1350.0, 1350.0], [1400.0, 1400.0], [1450.0, 1450.0], [1500.0, 1500.0],[1550,1550], [1600, 1600], [1650, 1650], [1700, 1700], [1750, 1750], [1800,1800]]

V_cost = 1
SIMU_ATTACK_TIME = 1
ATTACKER_TYPE_PROB = [0.2, 0.8]

lambda_qr = [0.0001]
# lambda_qr = [1.0]

# prob_coeff = [1.0]
noise_variance = [0.04]

directory = 'E_list_1_optimal_T_' + str(ALERT_TYPE) + '_A_' + str(ATTACKER_TYPE_PROB[0]) + '_' + str(ATTACKER_TYPE_PROB[1]) + '_N_' + str(PROB_SIMU_TIMES) + '_n_' + str(noise_variance[0])+ '_lambda_' + str(lambda_qr[0]) + '_new'
parent_dir = '../../result/control_B/'
path = os.path.join(parent_dir, directory)

if not os.path.exists(path):
    os.mkdir(path)

# start_index = 0
# end_index = 41
# # end_index = 52
# iterate_time = 15
# time_compute = []

# all_time_OSSP = []
all_time_epsilon_OSSP = []
# all_time_beta_OSSP = []

# all_prac_Bayesian_OSSP = []
all_prac_Bayesian_epsilon_OSSP = []
# all_prac_Bayesian_beta_OSSP = []

all_attacker_type_sampled = []

# all_type_match_opt_type_list = []
all_epsilon_signaling_active = []
all_epsilon_type_match_opt_type = []

all_epsilon_type_within_epsilon_types = []
all_epsilon_range_covered_attack = []
# all_beta_type_match_opt_type = []
# all_beta_signaling_active = []


# online_SSE_value = []
# online_SAG_value = []

for lambda_p in lambda_qr:
    # lambda_time_OSSP = []
    lambda_time_epsilon_OSSP = []
    # lambda_time_beta_OSSP = []

    lambda_prac_Bayesian_OSSP = []
    lambda_prac_Bayesian_epsilon_OSSP = []
    lambda_prac_Bayesian_beta_OSSP = []

    lambda_type_match_opt_type_list = []

    lambda_epsilon_type_within_epsilon_types = []
    lambda_epsilon_range_covered_attack = []
    lambda_epsilon_signaling_active = []
    lambda_epsilon_type_match_opt_type = []

    lambda_beta_signaling_active = []
    lambda_beta_type_match_opt_type = []

    for attacker_type in [-1]:

        # time_OSSP = []
        time_epsilon_OSSP = []
        # time_beta_OSSP = []

        # prac_Bayesian_OSSP = []
        prac_Bayesian_epsilon_OSSP = []
        # prac_Bayesian_beta_OSSP = []

        type_match_opt_type_list = []

        epsilon_type_within_epsilon_types = []
        epsilon_range_covered_attack = []
        epsilon_signaling_active = []
        epsilon_type_match_opt_type = []

        beta_signaling_active = []
        beta_type_match_opt_type = []

        for noise in noise_variance:
            # prob_being_caught_list_all = np.random.dirichlet(np.ones(ALERT_TYPE), size=PROB_SIMU_TIMES)*prob_sum
            # prob_being_caught_list_all = np.random.random((PROB_SIMU_TIMES, ALERT_TYPE)) * prob_sum
            prob_being_caught_list_all = np.array([[0.083, 0.075, 0.084, 0.129, 0.123, 0.109, 0.173]] * PROB_SIMU_TIMES)
            noise = np.random.normal(0, noise, prob_being_caught_list_all.shape)
            prob_being_caught_list_all = prob_being_caught_list_all + noise
            prob_being_caught_list_all = np.clip(prob_being_caught_list_all, 0.05, 0.5)

            # sub_time_OSSP = []
            sub_time_epsilon_OSSP = []
            # sub_time_beta_OSSP = []

            sub_prac_Bayesian_OSSP = []
            sub_prac_Bayesian_epsilon_OSSP = []
            sub_prac_Bayesian_beta_OSSP = []

            sub_type_match_opt_type_list = []

            sub_epsilon_type_within_epsilon_types = []
            sub_epsilon_range_covered_attack = []
            sub_epsilon_signaling_active = []
            sub_epsilon_type_match_opt_type = []

            sub_beta_signaling_active = []
            sub_beta_type_match_opt_type = []

            index = 0
            for epsilon_beta_idx in range(len(Epsilon_list)):
                Epsilon = Epsilon_list[epsilon_beta_idx]
                print('Epsilon = %f' % Epsilon[0])
                # Beta = Beta_list[epsilon_beta_idx]

                # prob_being_caught_list = [0.07013789070433528, 0.06352121472863086, 0.07401066995702825, 0.21688707065820345, 0.19680268919597624, 0.2116098610775148, 0.17954370984322163]
                # prob_being_caught_list = [0.08300539784086894, 0.07517469993135312, 0.08447896773494429, 0.1287783725219631, 0.12307204241591382, 0.10931351209516271, 0.17279092313691877]

                # attack_type_sampled = list(np.random.choice(ALERT_TYPE, ATTACK_TYPE_SAMPLE, p=ATTACK_TYPE_PROB))

                # sub_sub_time_OSSP = []
                sub_sub_time_epsilon_OSSP = []
                # sub_sub_time_beta_OSSP = []

                # sub_sub_prac_Bayesian_OSSP = []
                sub_sub_prac_Bayesian_epsilon_OSSP = []
                # sub_sub_prac_Bayesian_beta_OSSP = []

                sub_sub_type_match_opt_type_list = []

                sub_sub_epsilon_type_within_epsilon_types = []
                sub_sub_epsilon_range_covered_attack = []
                sub_sub_epsilon_signaling_active = []
                sub_sub_epsilon_type_match_opt_type = []

                sub_sub_beta_signaling_active = []
                sub_sub_beta_type_match_opt_type = []

                for prob_being_caught_list in prob_being_caught_list_all:

                    # sub_sub_sub_time_OSSP = []
                    sub_sub_sub_time_epsilon_OSSP = []
                    # sub_sub_sub_time_beta_OSSP = []

                    sub_sub_sub_prac_Bayesian_OSSP = []
                    sub_sub_sub_prac_Bayesian_epsilon_OSSP = []
                    sub_sub_sub_prac_Bayesian_beta_OSSP = []

                    sub_sub_sub_type_match_opt_type_list = []

                    sub_sub_sub_epsilon_type_within_epsilon_types = []
                    sub_sub_sub_epsilon_range_covered_attack = []
                    sub_sub_sub_epsilon_signaling_active = []
                    sub_sub_sub_epsilon_type_match_opt_type = []

                    sub_sub_sub_beta_signaling_active = []
                    sub_sub_sub_beta_type_match_opt_type = []

                    for mmm in range(SIMU_ATTACK_TIME):

                        attacker_type = random.choices([0, 1], weights=(0.2, 0.8), k=1)[0]

                        # ####### Bayesian OSSP
                        # solution_B_OSSP, z_OSSP, obj_B_OSSP, practice_obj_OSSP, opt_target_OSSP, type_match_opt_type_OSSP, run_time_OSSP, type_Bayesian, usefulness_Bayesian = bayesian_OSSP_qr(prob_being_caught_list, attacker_type, NUM_ALERT_COUNT, lambda_p)



                        ####### Bayesian Epsilon OSSP
                        solution_B_epsilon_OSSP, z_epsilon_OSSP, epsilon_range, type_within_epsilon_types, obj_B_epsilon_OSSP, practice_obj_epsilon, opt_target_epsilon_OSSP, type_match_opt_type_epsilon, run_time_epsilon_OSSP, type_epsilon, usefulness_epsilon = bounded_rationality_epsilon_qr(
                            prob_being_caught_list, attacker_type, NUM_ALERT_COUNT, Epsilon, lambda_p)



                        # ####### Bayesian Beta OSSP
                        # solution_B_beta_OSSP, z_beta_OSSP, obj_B_beta_OSSP, practice_obj_beta, opt_target_beta_OSSP, type_match_opt_type_beta, run_time_beta_OSSP, type_beta, usefulness_beta = bounded_rationality_beta_qr(
                        #     prob_being_caught_list, attacker_type, NUM_ALERT_COUNT, Beta, lambda_p)



                        # if usefulness_Bayesian is True and usefulness_epsilon is True and usefulness_beta is True:
                        if usefulness_epsilon is True:

    #                         ####### Bayesian OSSP
    #
    #                         p0_Bayesian, q0_Bayesian, p1_Bayesian, q1_Bayesian = solution_B_OSSP[
    #                                                                                  'p0_t' + str(opt_target_OSSP)], \
    #                                                                              solution_B_OSSP[
    #                                                                                  'q0_t' + str(opt_target_OSSP)], \
    #                                                                              solution_B_OSSP[
    #                                                                                  'p1_t' + str(opt_target_OSSP)], \
    #                                                                              solution_B_OSSP[
    #                                                                                  'q1_t' + str(opt_target_OSSP)]
    #
    #                         sub_sub_sub_time_OSSP.append(run_time_OSSP)
    #                         sub_sub_sub_prac_Bayesian_OSSP.append(practice_obj_OSSP)
    #                         # update available budget
    #                         if p1_Bayesian + q1_Bayesian <= 1e-6:
    # #                             print('Bayesian OSSP signaling NOT active.')
    #                             a = 1
    #                         else:
    # #                             print('Bayesian OSSP signaling Active.')
    #                             a = 1
    #                         if type_match_opt_type_OSSP is True:
    # #                             print('******************* ' + str(practice_obj) + '    OPT MATCH')
    #                             sub_sub_sub_type_match_opt_type_list.append(1)
    #                         else:
    # #                             print('******************* ' + str(practice_obj))
    #                             sub_sub_sub_type_match_opt_type_list.append(0)
    #
    # #                         print('p0 q0 p1 q1 %.4f   %.4f   %.4f   %.4f' % (p0, q0, p1, q1))



                            ####### Bayesian Epsilon OSSP

                            p0_epsilon, q0_epsilon, p1_epsilon, q1_epsilon = solution_B_epsilon_OSSP[
                                                                                 'p0_t' + str(type_epsilon)], \
                                                                             solution_B_epsilon_OSSP[
                                                                                 'q0_t' + str(type_epsilon)], \
                                                                             solution_B_epsilon_OSSP[
                                                                                 'p1_t' + str(type_epsilon)], \
                                                                             solution_B_epsilon_OSSP[
                                                                                 'q1_t' + str(type_epsilon)]

                            sub_sub_sub_epsilon_type_within_epsilon_types.append(type_within_epsilon_types)
                            sub_sub_sub_epsilon_range_covered_attack.append(epsilon_range)
                            sub_sub_sub_time_epsilon_OSSP.append(run_time_epsilon_OSSP)
                            sub_sub_sub_prac_Bayesian_epsilon_OSSP.append(practice_obj_epsilon)

                            if p0_epsilon < 0:
    #                             print('################## p0 = %f' % p0)
                                p0_epsilon = 0.0
                            if q0_epsilon < 0:
    #                             print('################## q0 = %f' % q0)
                                q0_epsilon = 0.0
                            if p1_epsilon < 0:
    #                             print('################## p1 = %f' % p1)
                                p1_epsilon = 0.0
                            if q1_epsilon < 0:
    #                             print('################## q1 = %f' % q1)
                                q1_epsilon = 0.0

                            # update available budegt
                            if p1_epsilon + q1_epsilon <= 1e-6:
    #                             print('       Epsilon signaling NOT active.')
                                sub_sub_sub_epsilon_signaling_active.append(0)

                            else:
    #                             print('       Epsilon signaling Active.')
                                sub_sub_sub_epsilon_signaling_active.append(1)


                            if type_match_opt_type_epsilon is True:
    #                             print('       ******************* ' + str(practice_obj) + '    OPT MATCH')
                                sub_sub_sub_epsilon_type_match_opt_type.append(1)
                            else:
    #                             print('       ******************* ' + str(practice_obj))
                                sub_sub_sub_epsilon_type_match_opt_type.append(0)
    #                         print('       p0 q0 p1 q1 %.4f   %.4f   %.4f   %.4f' % (p0, q0, p1, q1))

    #                         ####### Bayesian Beta OSSP
    #
    #                         p0_beta, q0_beta, p1_beta, q1_beta = solution_B_beta_OSSP['p0_t' + str(type_beta)], \
    #                                                              solution_B_beta_OSSP[
    #                                                                  'q0_t' + str(type_beta)], \
    #                                                              solution_B_beta_OSSP['p1_t' + str(type_beta)], \
    #                                                              solution_B_beta_OSSP[
    #                                                                  'q1_t' + str(type_beta)]
    #
    #                         sub_sub_sub_time_beta_OSSP.append(run_time_beta_OSSP)
    #                         sub_sub_sub_prac_Bayesian_beta_OSSP.append(practice_obj_beta)
    #
    #                         if p0_beta < 0:
    # #                             print('################## p0 = %f' % p0)
    #                             p0_beta = 0.0
    #                         if q0_beta < 0:
    # #                             print('################## q0 = %f' % q0)
    #                             q0_beta = 0.0
    #                         if p1_beta < 0:
    # #                             print('################## p1 = %f' % p1)
    #                             p1_beta = 0.0
    #                         if q1_beta < 0:
    # #                             print('################## q1 = %f' % q1)
    #                             q1_beta = 0.0
    #
    #                         # update available budegt
    #                         if p1_beta + q1_beta <= 1e-6:
    # #                             print('            Beta signaling NOT active.')
    #                             sub_sub_sub_beta_signaling_active.append(0)
    #                         else:
    # #                             print('            Beta signaling Active.')
    #                             sub_sub_sub_beta_signaling_active.append(1)
    #
    #                         if type_match_opt_type_beta is True:
    # #                             print('              ******************* ' + str(practice_obj) + '    OPT MATCH')
    #                             sub_sub_sub_beta_type_match_opt_type.append(1)
    #                         else:
    # #                             print('              ******************* ' + str(practice_obj))
    #                             sub_sub_sub_beta_type_match_opt_type.append(0)
    #
    # #                         print('              p0 q0 p1 q1 %.4f   %.4f   %.4f   %.4f\n' % (p0, q0, p1, q1))
    #
    #
    #                         if abs(obj_B_beta_OSSP - obj_B_OSSP) > 1 or abs(obj_B_epsilon_OSSP - obj_B_OSSP) > 1:
    #                             a = 1

                    # sub_sub_time_OSSP.append(sub_sub_sub_time_OSSP)
                    # sub_sub_prac_Bayesian_OSSP.append(sub_sub_sub_prac_Bayesian_OSSP)
                    # sub_sub_type_match_opt_type_list.append(sub_sub_sub_type_match_opt_type_list)

                    sub_sub_epsilon_type_within_epsilon_types.append(sub_sub_sub_epsilon_type_within_epsilon_types)
                    sub_sub_epsilon_range_covered_attack.append(sub_sub_sub_epsilon_range_covered_attack)
                    sub_sub_epsilon_signaling_active.append(sub_sub_sub_epsilon_signaling_active)
                    sub_sub_epsilon_type_match_opt_type.append(sub_sub_sub_epsilon_type_match_opt_type)
                    sub_sub_time_epsilon_OSSP.append(sub_sub_sub_time_epsilon_OSSP)
                    sub_sub_prac_Bayesian_epsilon_OSSP.append(sub_sub_sub_prac_Bayesian_epsilon_OSSP)

                    # sub_sub_beta_signaling_active.append(sub_sub_sub_beta_signaling_active)
                    # sub_sub_prac_Bayesian_beta_OSSP.append(sub_sub_sub_prac_Bayesian_beta_OSSP)
                    # sub_sub_beta_type_match_opt_type.append(sub_sub_sub_beta_type_match_opt_type)
                    # sub_sub_time_beta_OSSP.append(sub_sub_sub_time_beta_OSSP)

                # sub_time_OSSP.append(sub_sub_time_OSSP)
                # sub_prac_Bayesian_OSSP.append(sub_sub_prac_Bayesian_OSSP)
                # sub_type_match_opt_type_list.append(sub_sub_type_match_opt_type_list)

                sub_epsilon_type_within_epsilon_types.append(sub_sub_epsilon_type_within_epsilon_types)
                sub_epsilon_range_covered_attack.append(sub_sub_epsilon_range_covered_attack)
                sub_epsilon_signaling_active.append(sub_sub_epsilon_signaling_active)
                sub_epsilon_type_match_opt_type.append(sub_sub_epsilon_type_match_opt_type)
                sub_time_epsilon_OSSP.append(sub_sub_time_epsilon_OSSP)
                sub_prac_Bayesian_epsilon_OSSP.append(sub_sub_prac_Bayesian_epsilon_OSSP)

                # sub_beta_signaling_active.append(sub_sub_beta_signaling_active)
                # sub_prac_Bayesian_beta_OSSP.append(sub_sub_prac_Bayesian_beta_OSSP)
                # sub_beta_type_match_opt_type.append(sub_sub_beta_type_match_opt_type)
                # sub_time_beta_OSSP.append(sub_sub_time_beta_OSSP)



            # time_OSSP.append(sub_time_OSSP)
            # prac_Bayesian_OSSP.append(sub_prac_Bayesian_OSSP)
            # type_match_opt_type_list.append(sub_type_match_opt_type_list)

            epsilon_type_within_epsilon_types.append(sub_epsilon_type_within_epsilon_types)
            epsilon_range_covered_attack.append(sub_epsilon_range_covered_attack)
            epsilon_signaling_active.append(sub_epsilon_signaling_active)
            epsilon_type_match_opt_type.append(sub_epsilon_type_match_opt_type)
            prac_Bayesian_epsilon_OSSP.append(sub_prac_Bayesian_epsilon_OSSP)
            time_epsilon_OSSP.append(sub_time_epsilon_OSSP)

            # beta_signaling_active.append(sub_beta_signaling_active)
            # prac_Bayesian_beta_OSSP.append(sub_prac_Bayesian_beta_OSSP)
            # beta_type_match_opt_type.append(sub_beta_type_match_opt_type)
            # time_beta_OSSP.append(sub_time_beta_OSSP)

        # lambda_time_OSSP.append(time_OSSP)
        # lambda_prac_Bayesian_OSSP.append(prac_Bayesian_OSSP)
        # lambda_type_match_opt_type_list.append(type_match_opt_type_list)

        lambda_epsilon_type_within_epsilon_types.append(epsilon_type_within_epsilon_types)
        lambda_epsilon_range_covered_attack.append(epsilon_range_covered_attack)
        lambda_epsilon_signaling_active.append(epsilon_signaling_active)
        lambda_epsilon_type_match_opt_type.append(epsilon_type_match_opt_type)
        lambda_prac_Bayesian_epsilon_OSSP.append(prac_Bayesian_epsilon_OSSP)
        lambda_time_epsilon_OSSP.append(time_epsilon_OSSP)

        # lambda_beta_signaling_active.append(beta_signaling_active)
        # lambda_prac_Bayesian_beta_OSSP.append(prac_Bayesian_beta_OSSP)
        # lambda_beta_type_match_opt_type.append(beta_type_match_opt_type)
        # lambda_time_beta_OSSP.append(time_beta_OSSP)

    # all_time_OSSP.append(lambda_time_OSSP)
    # all_prac_Bayesian_OSSP.append(lambda_prac_Bayesian_OSSP)
    # all_type_match_opt_type_list.append(lambda_type_match_opt_type_list)

    all_epsilon_type_within_epsilon_types.append(lambda_epsilon_type_within_epsilon_types)
    all_epsilon_range_covered_attack.append(lambda_epsilon_range_covered_attack)
    all_epsilon_signaling_active.append(lambda_epsilon_signaling_active)
    all_epsilon_type_match_opt_type.append(lambda_epsilon_type_match_opt_type)
    all_prac_Bayesian_epsilon_OSSP.append(lambda_prac_Bayesian_epsilon_OSSP)
    all_time_epsilon_OSSP.append(lambda_time_epsilon_OSSP)

    # all_beta_signaling_active.append(lambda_beta_signaling_active)
    # all_prac_Bayesian_beta_OSSP.append(lambda_prac_Bayesian_beta_OSSP)
    # all_beta_type_match_opt_type.append(lambda_beta_type_match_opt_type)
    # all_time_beta_OSSP.append(lambda_time_beta_OSSP)


# for i in range(len(lambda_qr)):
#     print('lambda = %f' % lambda_qr[i])

#     for attacker_type in [0,1]:
#         print('    Attacker type = %d' % attacker_type)

#         for noise_idx in range(len(noise_variance)):
#             print('        prob_coeff = %f' % noise_variance[noise_idx])

#             for epsilon_beta_idx in range(len(Epsilon_list)):
#                 print('            epsilon = %f, beta = %f' %  (Epsilon_list[epsilon_beta_idx][0], Beta_list[epsilon_beta_idx][0]))

#                 print(np.mean(np.array(all_prac_Bayesian_OSSP[i][attacker_type][noise_idx][epsilon_beta_idx]).ravel()))
#                 print(np.mean(np.array(all_prac_Bayesian_epsilon_OSSP[i][attacker_type][noise_idx][epsilon_beta_idx]).ravel()))
#                 print(np.mean(np.array(all_prac_Bayesian_beta_OSSP[i][attacker_type][noise_idx][epsilon_beta_idx]).ravel()))

#                 print('\n')



# np.save(path + '/all_time_OSSP.npy', all_time_OSSP)
# np.save(path + '/all_prac_Bayesian_OSSP.npy', all_prac_Bayesian_OSSP)
# np.save(path + '/all_type_match_opt_type_list.npy', all_type_match_opt_type_list)
np.save(path + '/all_epsilon_type_within_epsilon_types.npy', all_epsilon_type_within_epsilon_types)
np.save(path + '/all_epsilon_range_covered_attack.npy', all_epsilon_range_covered_attack)
np.save(path + '/all_epsilon_signaling_active.npy', all_epsilon_signaling_active)
np.save(path + '/all_epsilon_type_match_opt_type.npy', all_epsilon_type_match_opt_type)
np.save(path + '/all_prac_Bayesian_epsilon_OSSP.npy', all_prac_Bayesian_epsilon_OSSP)
np.save(path + '/all_time_epsilon_OSSP.npy', all_time_epsilon_OSSP)
# np.save(path + '/all_beta_signaling_active.npy', all_beta_signaling_active)
# np.save(path + '/all_prac_Bayesian_beta_OSSP.npy', all_prac_Bayesian_beta_OSSP)
# np.save(path + '/all_beta_type_match_opt_type.npy', all_beta_type_match_opt_type)
# np.save(path + '/all_time_beta_OSSP.npy', all_time_beta_OSSP)

