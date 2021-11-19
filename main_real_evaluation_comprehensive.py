
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
import matplotlib.dates as mdates
import time
import copy
from process_daily_data import process_daily_data
from prob_being_caught import prob_being_caught
from bayesian_OSSP_eval_qr_real import bayesian_OSSP_qr
from bounded_rationality_M1_eval_qr_real import bounded_rationality_epsilon_qr
from bounded_rationality_M2_eval_qr_real import bounded_rationality_beta_qr

np.random.seed(2021)

ALERT_TYPE = 7
ALPHA = 0.01    # the proportion of resources used to coping with repeated access requests.
B = 60.0
# Beta = [1000000000, 1000000000]
Beta_list = [[0.0, 0.0], [0.125, 0.125], [0.25, 0.25], [0.5, 0.5], [1.0, 1.0], [2.0, 2.0], [4.0, 4.0], [8.0, 8.0]]
Epsilon_list = [[0.0, 0.0], [50.0, 50.0], [100.0, 100.0], [200.0, 200.0], [400.0, 400.0], [800.0, 800.0], [1600.0, 1600.0], [3200.0, 3200.0]]  # epsilon
# Epsilon = [700, 70]  # epsilon
V_cost = 1
ATTACKER_TYPE_PROB = [0.2,0.8]

lambda_p = 1.0

for epsilon_beta_idx in range(len(Beta_list)):
    
    Epsilon = Epsilon_list[epsilon_beta_idx]
    Beta = Beta_list[epsilon_beta_idx]
    print('\n Epsilon = %f, Beta = %f' % (Epsilon[0], Beta[0]))
    directory = 'REAL_B_'+str(int(B)) + '_E_' + str(Epsilon[0]) + '_Beta_' + str(Beta[0]) + '_T_' + str(ALERT_TYPE) + '_A_' + str(ATTACKER_TYPE_PROB[0]) + '_' + str(ATTACKER_TYPE_PROB[1]) + '_C10'
    parent_dir = '../../result/comprehensive/'
    path = os.path.join(parent_dir, directory)

    if not os.path.exists(path):
        os.mkdir(path)

    start_index = 0
    end_index = 41
    # end_index = 52
    iterate_time = 15
    time_compute = []

    time_OSSP = []
    time_epsilon_OSSP = []
    time_beta_OSSP = []

    all_prac_Bayesian_OSSP = []
    all_prac_Bayesian_epsilon_OSSP = []
    all_prac_Bayesian_beta_OSSP = []

    all_diff_epsilon_vs_Bayesian = []
    all_diff_beta_vs_Bayesian = []
    all_diff_beta_vs_epsilon = []

    all_attacker_type_sampled = []

    all_type_match_opt_type = []

    all_epsilon_signaling_active = []
    all_epsilon_type_match_opt_type = []

    all_epsilon_type_within_epsilon_types = []
    all_epsilon_range_covered_attack = []
    all_beta_type_match_opt_type = []
    all_beta_signaling_active = []

    B_cons_OSSP = []
    B_cons_epsilon_OSSP = []
    B_cons_beta_OSSP = []

    # online_SSE_value = []
    # online_SAG_value = []

    for iter in range(iterate_time):
    # for iter in [2]:

        print("\niter = "+ str(iter))

        # prepare historical data and run-time data
        training_data_separate, test_data, test_data_time, test_data_index = process_daily_data(start_index, end_index, iterate_time, iter)

        attacker_type_sampled = list(np.random.choice(2, len(test_data_index), p=ATTACKER_TYPE_PROB))
        all_attacker_type_sampled.append(attacker_type_sampled)

        # set up residual budget for the end of each day
        residual = [B * 0.02, B * 0.02, B * 0.02, B * 0.02, B * 0.02, B * 0.02, B * 0.01]
        B_avail_online_sig_OSSP = B * (1 - ALPHA) - np.sum(residual)
        B_avail_online_sig_epsilon_OSSP = B * (1 - ALPHA) - np.sum(residual)
        B_avail_online_sig_beta_OSSP = B * (1 - ALPHA) - np.sum(residual)

        type_list = test_data['type']
        gate_for_insert_OSSP = 1
        gate_for_insert_epsilon_OSSP = 1
        gate_for_insert_beta_OSSP = 1

        prob_coeff_being_caught_last_iter_SAG = [0] * ALERT_TYPE
        prob_coeff_being_caught_online_sig_last_iter_SAG = [0] * ALERT_TYPE
        prob_coeff_being_caught_online_sig_last_iter_epsilon_SAG = [0] * ALERT_TYPE
        prob_coeff_being_caught_online_sig_last_iter_beta_SAG = [0] * ALERT_TYPE

        Bayesian_SAG_auditor_practice_utility = []
        Bayesian_epsilon_SAG_auditor_practice_utility = []
        Bayesian_beta_SAG_auditor_practice_utility = []

        diff_epsilon_vs_Bayesian = []
        diff_beta_vs_Bayesian = []
        diff_beta_vs_epsilon = []

        type_match_opt_type_list = []

        epsilon_signaling_active = []
        epsilon_type_match_opt_type = []
        epsilon_type_within_epsilon_types = []
        epsilon_range_covered_attack = []

        beta_type_match_opt_type = []
        beta_signaling_active = []

        sub_B_cons_OSSP = []
        sub_B_cons_epsilon_OSSP = []
        sub_B_cons_beta_OSSP = []

        for i in test_data_index:

            ###### Bayesian OSSP

            # compute prob of being caught for all types of attacks
            prob_coeff_being_caught_online_sig_last_iter_SAG, prob_coeff_being_caught_online_sig_SAG, num_alerts_count, B_avail_online_sig_OSSP, type_SAG, gate_for_insert_OSSP = prob_being_caught(
                training_data_separate, test_data_time, type_list, i, prob_coeff_being_caught_online_sig_last_iter_SAG,
                gate_for_insert_OSSP, B_avail_online_sig_OSSP, residual)

            assert B_avail_online_sig_OSSP > 0
            sub_B_cons_OSSP.append(B_avail_online_sig_OSSP)

            # compute solutions: compute the optimal solution of Bayesian OSSP, and sample an attacker type
            # solution_B_OSSP: the whole solution set of Bayesian OSSP
            # z_OSSP: solution for attack types (2 subvector, each represents an attacker type)
            # obj_B_OSSP: the optimized objective function value for Bayesian OSSP
            # practice_obj: expected utility of the auditor given the assigned/sampled attacker type, as well as the actual attack type in the test day
            # opt_target_OSSP: the optimal attack type given the sampled attacker type
            # run_time_OSSP: solving time
            solution_B_OSSP, z_OSSP, obj_B_OSSP, practice_obj_OSSP, opt_target_OSSP, type_match_opt_type_OSSP, run_time_OSSP, usefulness_OSSP = bayesian_OSSP_qr(B_avail_online_sig_OSSP, prob_coeff_being_caught_online_sig_SAG, type_SAG, num_alerts_count, attacker_type_sampled[i-1], lambda_p)


            if usefulness_OSSP is True:
                time_OSSP.append(run_time_OSSP)
                p0_OSSP, q0_OSSP, p1_OSSP, q1_OSSP = solution_B_OSSP['p0_t' + str(opt_target_OSSP)], solution_B_OSSP[
                    'q0_t' + str(opt_target_OSSP)], solution_B_OSSP['p1_t' + str(opt_target_OSSP)], solution_B_OSSP['q1_t' + str(opt_target_OSSP)]

                # update available budget
                if p1_OSSP+q1_OSSP <= 1e-6:
                    print('    Bayesian OSSP signaling NOT active.')
                else:
                    print('    Bayesian OSSP signaling Active.')

                warning = np.random.choice([0, 1], p=[p0_OSSP + q0_OSSP, 1 - p0_OSSP - q0_OSSP])
                if warning == 0:
                    B_avail_online_sig_OSSP = B_avail_online_sig_OSSP - p0_OSSP / (p0_OSSP + q0_OSSP) * V_cost
                else:
                    B_avail_online_sig_OSSP = B_avail_online_sig_OSSP - p1_OSSP / (p1_OSSP + q1_OSSP) * V_cost


                Bayesian_SAG_auditor_practice_utility.append(practice_obj_OSSP)

                if type_match_opt_type_OSSP is True:
                    print('******************* ' + str(practice_obj_OSSP) + '    OPT MATCH')
                    type_match_opt_type_list.append(1)
                else:
                    print('******************* ' + str(practice_obj_OSSP))
                    type_match_opt_type_list.append(0)

                print('p0 q0 p1 q1 %.4f   %.4f   %.4f   %.4f' % (p0_OSSP, q0_OSSP, p1_OSSP, q1_OSSP))

                # p0, q0, p1, q1 = [], [], [], []
                # for attacker_type in range(len(z_OSSP)):
                #     opt_target_OSSP = z_OSSP[attacker_type].index(1.0)
                #     p0.append(solution_B_OSSP['p0_t' + str(opt_target_OSSP)])
                #     p1.append(solution_B_OSSP['p1_t' + str(opt_target_OSSP)])
                #     q0.append(solution_B_OSSP['q0_t' + str(opt_target_OSSP)])
                #     q1.append(solution_B_OSSP['q1_t' + str(opt_target_OSSP)])

            ###### Bayesian epsilon OSSP

            # compute prob of being caught for all types of attacks
            prob_coeff_being_caught_online_sig_last_iter_epsilon_SAG, prob_coeff_being_caught_online_sig_epsilon_SAG, num_alerts_count, B_avail_online_sig_epsilon_OSSP, type_epsilon_SAG, gate_for_insert_epsilon_OSSP = prob_being_caught(
                training_data_separate, test_data_time, type_list, i, prob_coeff_being_caught_online_sig_last_iter_epsilon_SAG,
                gate_for_insert_epsilon_OSSP, B_avail_online_sig_epsilon_OSSP, residual)

            assert B_avail_online_sig_epsilon_OSSP > 0

            sub_B_cons_epsilon_OSSP.append(B_avail_online_sig_epsilon_OSSP)

            # compute solutions
            solution_B_epsilon_OSSP, z_epsilon_OSSP, epsilon_range, type_within_epsilon_types, obj_B_epsilon_OSSP, practice_obj_epsilon, opt_target_epsilon_OSSP, type_match_opt_type_epsilon, run_time_epsilon_OSSP, usefulness_epsilon = bounded_rationality_epsilon_qr(
                B_avail_online_sig_epsilon_OSSP, prob_coeff_being_caught_online_sig_epsilon_SAG, type_epsilon_SAG, num_alerts_count, attacker_type_sampled[i-1], Epsilon, lambda_p)

            if usefulness_epsilon is True:
                time_epsilon_OSSP.append(run_time_epsilon_OSSP)

                epsilon_type_within_epsilon_types.append(type_within_epsilon_types)
                epsilon_range_covered_attack.append(epsilon_range)

                p0_epsilon, q0_epsilon, p1_epsilon, q1_epsilon = solution_B_epsilon_OSSP['p0_t' + str(type_epsilon_SAG)], solution_B_epsilon_OSSP['q0_t' + str(type_epsilon_SAG)], \
                                 solution_B_epsilon_OSSP['p1_t' + str(type_epsilon_SAG)], solution_B_epsilon_OSSP['q1_t' + str(type_epsilon_SAG)]

                if p0_epsilon < 0:
                    print('################## p0 = %f' % p0_epsilon)
                    p0_epsilon = 0.0
                if q0_epsilon < 0:
                    print('################## q0 = %f' % q0_epsilon)
                    q0_epsilon = 0.0
                if p1_epsilon < 0:
                    print('################## p1 = %f' % p1_epsilon)
                    p1_epsilon = 0.0
                if q1_epsilon < 0:
                    print('################## q1 = %f' % q1_epsilon)
                    q1_epsilon = 0.0

                # update available budegt
                if p1_epsilon+q1_epsilon <= 1e-6:
                    print('        Epsilon signaling NOT active.')
                    epsilon_signaling_active.append(0)
                    B_avail_online_sig_epsilon_OSSP = B_avail_online_sig_epsilon_OSSP - p0_epsilon / (p0_epsilon + q0_epsilon)
                else:
                    print('        Epsilon signaling Active.')
                    epsilon_signaling_active.append(1)
                    warning = np.random.choice([0, 1], p=[p0_epsilon + q0_epsilon, 1 - p0_epsilon - q0_epsilon])
                    if warning == 0:
                        B_avail_online_sig_epsilon_OSSP = B_avail_online_sig_epsilon_OSSP - p0_epsilon / (p0_epsilon + q0_epsilon) * V_cost
                    else:
                        B_avail_online_sig_epsilon_OSSP = B_avail_online_sig_epsilon_OSSP - p1_epsilon / (p1_epsilon + q1_epsilon) * V_cost

                Bayesian_epsilon_SAG_auditor_practice_utility.append(practice_obj_epsilon)

                if type_match_opt_type_epsilon is True:
                    print('       ******************* ' + str(practice_obj_epsilon) + '    OPT MATCH')
                    epsilon_type_match_opt_type.append(1)
                else:
                    print('       ******************* ' + str(practice_obj_epsilon))
                    epsilon_type_match_opt_type.append(0)
                print('       p0 q0 p1 q1 %.4f   %.4f   %.4f   %.4f' % (p0_epsilon, q0_epsilon, p1_epsilon, q1_epsilon))


            ###### Bayesian beta OSSP

            # compute prob of being caught for all types of attacks
            prob_coeff_being_caught_online_sig_last_iter_beta_SAG, prob_coeff_being_caught_online_sig_beta_SAG, num_alerts_count, B_avail_online_sig_beta_OSSP, type_beta_SAG, gate_for_insert_beta_OSSP = prob_being_caught(
                training_data_separate, test_data_time, type_list, i, prob_coeff_being_caught_online_sig_last_iter_beta_SAG,
                gate_for_insert_beta_OSSP, B_avail_online_sig_beta_OSSP, residual)

            assert B_avail_online_sig_beta_OSSP > 0

            sub_B_cons_beta_OSSP.append(B_avail_online_sig_beta_OSSP)

            # compute solutions
            solution_B_beta_OSSP, z_beta_OSSP, obj_B_beta_OSSP, practice_obj_beta, opt_target_beta_OSSP, type_match_opt_type_beta, run_time_beta_OSSP, usefulness_beta = bounded_rationality_beta_qr(
                B_avail_online_sig_beta_OSSP, prob_coeff_being_caught_online_sig_beta_SAG, type_beta_SAG, num_alerts_count, attacker_type_sampled[i-1], Beta, lambda_p)

            if usefulness_beta is True:

                time_beta_OSSP.append(run_time_beta_OSSP)
                p0_beta, q0_beta, p1_beta, q1_beta = solution_B_beta_OSSP['p0_t' + str(type_beta_SAG)], solution_B_beta_OSSP['q0_t' + str(type_beta_SAG)], \
                                 solution_B_beta_OSSP['p1_t' + str(type_beta_SAG)], solution_B_beta_OSSP['q1_t' + str(type_beta_SAG)]

                if p0_beta < 0:
                    print('################## p0 = %f' % p0_beta)
                    p0_beta = 0.0
                if q0_beta < 0:
                    print('################## q0 = %f' % q0_beta)
                    q0_beta = 0.0
                if p1_beta < 0:
                    print('################## p1 = %f' % p1_beta)
                    p1_beta = 0.0
                if q1_beta < 0:
                    print('################## q1 = %f' % q1_beta)
                    q1_beta = 0.0


                # update available budegt
                if p1_beta+q1_beta <= 1e-6:
                    print('            Beta signaling NOT active.')
                    beta_signaling_active.append(0)
                    B_avail_online_sig_beta_OSSP = B_avail_online_sig_beta_OSSP - p0_beta / (p0_beta + q0_beta)
                else:
                    print('            Beta signaling Active.')
                    beta_signaling_active.append(1)
                    warning = np.random.choice([0, 1], p=[p0_beta + q0_beta, 1 - p0_beta - q0_beta])
                    if warning == 0:
                        B_avail_online_sig_beta_OSSP = B_avail_online_sig_beta_OSSP - p0_beta / (p0_beta + q0_beta) * V_cost
                    else:
                        B_avail_online_sig_beta_OSSP = B_avail_online_sig_beta_OSSP - p1_beta / (p1_beta + q1_beta) * V_cost

                Bayesian_beta_SAG_auditor_practice_utility.append(practice_obj_beta)

                if type_match_opt_type_beta is True:
                    print('              ******************* ' + str(practice_obj_beta) + '    OPT MATCH')
                    beta_type_match_opt_type.append(1)
                else:
                    print('              ******************* ' + str(practice_obj_beta))
                    beta_type_match_opt_type.append(0)

                print('              p0 q0 p1 q1 %.4f   %.4f   %.4f   %.4f\n' % (p0_beta, q0_beta, p1_beta, q1_beta))


        all_prac_Bayesian_OSSP.append(list(Bayesian_SAG_auditor_practice_utility))
        all_prac_Bayesian_epsilon_OSSP.append(list(Bayesian_epsilon_SAG_auditor_practice_utility))
        all_prac_Bayesian_beta_OSSP.append(list(Bayesian_beta_SAG_auditor_practice_utility))

        mean_OSSP, std_OSSP = np.mean(Bayesian_SAG_auditor_practice_utility), np.std(Bayesian_SAG_auditor_practice_utility)
        mean_epsilon_OSSP, std_epsilon_OSSP = np.mean(Bayesian_epsilon_SAG_auditor_practice_utility), np.std(Bayesian_epsilon_SAG_auditor_practice_utility)
        mean_beta_OSSP, std_beta_OSSP = np.mean(Bayesian_beta_SAG_auditor_practice_utility), np.std(
            Bayesian_beta_SAG_auditor_practice_utility)

        print(mean_OSSP, std_OSSP)
        print(mean_epsilon_OSSP, std_epsilon_OSSP)
        print(mean_beta_OSSP, std_beta_OSSP)

        diff_epsilon_vs_Bayesian = np.array(Bayesian_epsilon_SAG_auditor_practice_utility) - np.array(Bayesian_SAG_auditor_practice_utility)
        diff_beta_vs_Bayesian = np.array(Bayesian_beta_SAG_auditor_practice_utility) - np.array(
            Bayesian_SAG_auditor_practice_utility)
        diff_beta_vs_epsilon = np.array(Bayesian_beta_SAG_auditor_practice_utility) - np.array(
            Bayesian_epsilon_SAG_auditor_practice_utility)

        mean_diff_epsilon_vs_Bayesian, std_diff_epsilon_vs_Bayesian = np.mean(diff_epsilon_vs_Bayesian), np.std(diff_epsilon_vs_Bayesian)
        mean_diff_beta_vs_Bayesian, std_diff_beta_vs_Bayesian = np.mean(diff_beta_vs_Bayesian), np.std(diff_beta_vs_Bayesian)
        mean_diff_beta_vs_epsilon, std_diff_beta_vs_epsilon = np.mean(diff_beta_vs_epsilon), np.std(diff_beta_vs_epsilon)

        print(mean_diff_epsilon_vs_Bayesian, std_diff_epsilon_vs_Bayesian)
        print(mean_diff_beta_vs_Bayesian, std_diff_beta_vs_Bayesian)
        print(mean_diff_beta_vs_epsilon, std_diff_beta_vs_epsilon)

        all_diff_epsilon_vs_Bayesian.append(list(diff_epsilon_vs_Bayesian))
        all_diff_beta_vs_Bayesian.append(list(diff_beta_vs_Bayesian))
        all_diff_beta_vs_epsilon.append(list(diff_beta_vs_epsilon))

        all_type_match_opt_type.append(type_match_opt_type_list)

        all_epsilon_signaling_active.append(epsilon_signaling_active)
        all_epsilon_type_match_opt_type.append(epsilon_type_match_opt_type)
        all_epsilon_type_within_epsilon_types.append(epsilon_type_within_epsilon_types)
        all_epsilon_range_covered_attack.append(epsilon_range_covered_attack)

        all_beta_type_match_opt_type.append(beta_type_match_opt_type)
        all_beta_signaling_active.append(beta_signaling_active)

        B_cons_OSSP.append(sub_B_cons_OSSP)
        B_cons_epsilon_OSSP.append(sub_B_cons_epsilon_OSSP)
        B_cons_beta_OSSP.append(sub_B_cons_beta_OSSP)

        # # draw fitures
        # result_single = [test_data_time, Bayesian_SAG_auditor_practice_utility]
        # result_comb = pd.DataFrame(result_single)
        # result_comb = result_comb.T
        # result_comb = result_comb.rename(columns={0: 'time'})
        # result_comb = result_comb.rename(columns={1: 'Bayesian OSSP'})
        #
        # # file_name = "Result_new_final_mul_4_groups_" + str(iter + 1) + "_.csv"
        # # result_comb.to_csv(file_name)
        #
        # # read_file_name = "./" + file_name
        # # df = pd.read_csv(read_file_name)
        # time_column = result_comb.time
        # B_OSSP_column = result_comb['Bayesian OSSP']
        # time_axis = []
        # for time_point in time_column:
        #     hour_min = time_point[11:16]
        #     hour_min_list = time_point[11:16].split(':')
        #     # time_axis.append(datetime.time(int(hour_min_list[0]), int(hour_min_list[1])))
        #     time_axis.append(datetime.datetime.strptime(hour_min, '%H:%M'))
        #
        # B_OSSP = [x for x in B_OSSP_column]
        #
        # fig, ax = plt.subplots()
        # ax.plot(time_axis, B_OSSP, 'go:', label='Bayesian OSSP', markersize=4)
        # # ax.plot(time_axis, SSG_online, 'rx:', label='online SSE', markersize=5)
        # # ax.plot(time_axis, SSG_offline, 'b--', label='offline SSE', markersize=4)
        #
        # # ax.plot(time_axis, SAG, 'go', label='OOSP', markersize=4)
        # # ax.plot(time_axis, golden, 'r*', label='Prophet OOSP', markersize=5)
        # # ax.plot(time_axis, SSG_online, 'bx', label='Online SSE', markersize=5)
        # # ax.plot(time_axis, SSG_offline, 'm', label='Offline SSE', markersize=4)
        #
        # ax.legend(loc='best', prop={'size': 10})
        # myFmt = mdates.DateFormatter('%H:%M')
        # plt.ylim(-400, 50)
        # plt.xlim(xmax=datetime.datetime.strptime("23:59", '%H:%M'))
        # # ax = plt.plot(time_axis,golden)
        #
        #
        # ax.xaxis.set_major_formatter(myFmt)
        # axis_font = {'fontname': 'Arial', 'size': '10'}
        # plt.ylabel('Expected utility of the auditor', **axis_font)
        # plt.xlabel('Time', **axis_font)
        #
        # plt.ylabel('Expected utility of the auditor')
        # plt.xlabel('Time')
        #
        # ax.xaxis.set_major_formatter(myFmt)
        # plt.ylabel('Expected utility of the auditor')
        # plt.xlabel('Time')

        #################################################################
        # online SSG  &  online signaling

        # print(Bayesian_SAG_auditor_practice_utility)




    np.save(path + '/all_prac_Bayesian_OSSP.npy', all_prac_Bayesian_OSSP)
    np.save(path + '/all_prac_Bayesian_epsilon_OSSP.npy', all_prac_Bayesian_epsilon_OSSP)
    np.save(path + '/all_prac_Bayesian_beta_OSSP.npy', all_prac_Bayesian_beta_OSSP)

    np.save(path + '/all_diff_epsilon_vs_Bayesian.npy', all_diff_epsilon_vs_Bayesian)
    np.save(path + '/all_diff_beta_vs_Bayesian.npy', all_diff_beta_vs_Bayesian)
    np.save(path + '/all_diff_beta_vs_epsilon.npy', all_diff_beta_vs_epsilon)

    np.save(path + '/all_attacker_type_sampled.npy', all_attacker_type_sampled)

    np.save(path + '/time_OSSP.npy', time_OSSP)
    np.save(path + '/time_epsilon_OSSP.npy', time_epsilon_OSSP)
    np.save(path + '/time_beta_OSSP.npy', time_beta_OSSP)

    np.save(path + '/all_type_match_opt_type.npy', all_type_match_opt_type)

    np.save(path + '/all_epsilon_signaling_active.npy', all_epsilon_signaling_active)
    np.save(path + '/all_epsilon_type_match_opt_type.npy', all_epsilon_type_match_opt_type)
    np.save(path + '/all_epsilon_type_within_epsilon_types.npy', all_epsilon_type_within_epsilon_types)
    np.save(path + '/all_epsilon_range_covered_attack.npy', all_epsilon_range_covered_attack)

    np.save(path + '/all_beta_type_match_opt_type.npy', all_beta_type_match_opt_type)
    np.save(path + '/all_beta_signaling_active.npy', all_beta_signaling_active)

    np.save(path + '/B_cons_OSSP.npy', B_cons_OSSP)
    np.save(path + '/B_cons_epsilon_OSSP.npy', B_cons_epsilon_OSSP)
    np.save(path + '/B_cons_beta_OSSP.npy', B_cons_beta_OSSP)
