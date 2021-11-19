"""
    ------ Input file ------
    No. of defender strategies (X)
    No. of attackers (L)
    | Probability for an attacker (p_l)
    | No. of attack strategies for an attacker (Q_l)
    | [
    |       Matrix ( X * Q_l) with
    |       values r, c
    | ]
    | where r,c are rewards for defender and attacker respectively
"""

import gurobipy
import sys
import numpy as np
from scipy.special import softmax
from datetime import datetime


D = [0, 0] # delta
Target_num = 7
Attacker_type = 2

const_P = [0.186] * Target_num # prob of being deterred
# const_E = [10,10,2,7,4,3,2] # expectation of # of future false positive alerts
const_C = [-10] * Target_num # loss for each quit of a normal user


def bounded_rationality_epsilon_qr(B_now, expected_prob_catch, type, num_alerts_count, attacker_type, E, lambda_p):

    # expected_prob_catch = [0.1, 0.3, 0.05, 0.03, 0.07, 0.1, 0.1]
    usefulness = True
    try:
        # Create a new model
        model = gurobipy.Model("MIQP")

        # if len(sys.argv) < 3:
        file = open(str(sys.argv[1]), "r")

        # Add resource allocation
        b_list = []
        con = gurobipy.LinExpr()
        for i in range(Target_num):
            b_list.append(model.addVar(lb=0, ub=B_now, vtype=gurobipy.GRB.CONTINUOUS, name='B_now_t'+str(i)))
            con.add(b_list[i])
        model.addConstr(con <= B_now)
        model.update()

        # Add defender strategies (p0 p1 q0 q1)
        # num_defender_strategy = Target_num * 4
        x = []
        for i in range(Target_num):
            name_ = ['p0_t', 'q0_t', 'p1_t', 'q1_t']
            con = gurobipy.LinExpr()
            x_inside = []
            for j in range(len(name_)):
                x_inside.append(model.addVar(lb=0, ub=1, vtype=gurobipy.GRB.CONTINUOUS, name=name_[j]+str(i)))
                con.add(x_inside[-1])
            x.append(x_inside)
            model.addConstr(con == 1)
            model.update()
            con = gurobipy.LinExpr()
            con.add(x_inside[0])
            con.add(x_inside[2])
            con.add(-1 * b_list[i] * expected_prob_catch[i])  ## ********
            model.addConstr(con == 0)  ## ********
            model.update()

        # # Add defender strategy constraints
        # for i in range(len(x)):
        #     con = gurobipy.LinExpr()
        #     for j in range(len(x[0])):
        #         con.add(x[i][j])
        #     model.addConstr(con == 1)
        #     con = gurobipy.LinExpr()
        #     con.add(x[i][0])
        #     con.add(x[i][2])
        #     model.addConstr(con == expected_prob_catch)
        # model.update()

        """ Start processing for attacker types """

        obj = gurobipy.QuadExpr()
        M = 100000000
        obj_update_ind = True

        R_a = []
        R_d = []

        for l in range(Attacker_type):

            # Probability of l-th attacker
            theta = float(file.readline().strip())

            # Add A^l
            A_l = model.addVar(lb=-gurobipy.GRB.INFINITY, ub=gurobipy.GRB.INFINITY, vtype=gurobipy.GRB.CONTINUOUS, name='A_l' + str(l))

            # Add gamma_l
            gamma_l = model.addVar(lb=-gurobipy.GRB.INFINITY, ub=gurobipy.GRB.INFINITY, vtype=gurobipy.GRB.CONTINUOUS, name='gamma_l' + str(l))

            model.update()

            # build obj
            obj.add(theta * gamma_l)


            # Add l-th attacker info to the model
            num_attacher_strategy = int(file.readline())
            z_l_list = []
            y_l_list = []
            cve_names = file.readline().strip().split("|")
            con_z = gurobipy.LinExpr()
            con_y = gurobipy.LinExpr()
            for i in range(num_attacher_strategy):
                n = "z_l" + str(l) + "_" + cve_names[i]
                z_l_list.append(model.addVar(lb=0, ub=1, vtype=gurobipy.GRB.INTEGER, name=n))
                con_z.add(z_l_list[-1])
                n = "y_l" + str(l) + "_" + cve_names[i]
                y_l_list.append(model.addVar(lb=0, ub=1, vtype=gurobipy.GRB.INTEGER, name=n))
                con_y.add(y_l_list[-1])
                con_z_y = gurobipy.LinExpr()
                con_z_y.add(z_l_list[-1])
                con_z_y.add(-1*y_l_list[-1])
                model.addConstr(con_z_y <= 0)
                model.update()
            model.addConstr(con_z == 1)
            model.update()
            model.addConstr(con_y >= 1)
            model.update()

            # Get reward for attacker and defender
            # the first subvector corresponds to utility of attacker not being caught;  the second subvector corresponds to utility of attacker being caught
            R_l_d = []
            R_l_a = []
            for i in range(2): # 1st:
                rewards = file.readline().split()
                r = []
                c = []
                for j in range(Target_num):
                    r_and_c = rewards[j].split(",")
                    r.append(float(r_and_c[0]))
                    c.append(float(r_and_c[1]))
                R_l_d.append(r)
                R_l_a.append(c)

            R_a.append(R_l_a)
            R_d.append(R_l_d)

            for target in range(Target_num):
                # p_1^t U_ac^tl + q_1^t U_au^tl <= delta
                con_force = gurobipy.LinExpr()
                con_force.add(R_l_a[1][target] * x[target][2])
                con_force.add(R_l_a[0][target] * x[target][3])
                con_force.add(-D[l])
                model.addConstr(con_force <= 0)
                model.update()

                # A^l - (p_0^t U_ac^tl + q_0^t U_au^tl ) >= 0
                con_A_ll = gurobipy.LinExpr()
                con_A_ll.add(A_l)
                con_A_ll.add(-1 * R_l_a[1][target] * x[target][0])
                con_A_ll.add(-1 * R_l_a[0][target] * x[target][1])
                model.addConstr(con_A_ll >= 0)
                model.update()

                # A^l - (p_0^t U_ac^tl + q_0^t U_au^tl ) <= (1 - z_t^l) M
                con_A_lr = gurobipy.LinExpr()
                con_A_lr.add(A_l)
                con_A_lr.add(-1 * R_l_a[1][target] * x[target][0])
                con_A_lr.add(-1 * R_l_a[0][target] * x[target][1])
                con_A_lr.add(-1 * M)
                con_A_lr.add(M * z_l_list[target])
                model.addConstr(con_A_lr <= 0)
                model.update()

                # A^l - (p_0^t U_ac^tl + q_0^t U_au^tl ) >= \epsilon (1-y_t^l)
                con_A_lly = gurobipy.LinExpr()
                con_A_lly.add(A_l)
                con_A_lly.add(-1 * R_l_a[1][target] * x[target][0])
                con_A_lly.add(-1 * R_l_a[0][target] * x[target][1])
                con_A_lly.add(-1 * E[l])
                con_A_lly.add(E[l] * y_l_list[target])
                model.addConstr(con_A_lly >= 0)
                model.update()

                # A^l - (p_0^t U_ac^tl + q_0^t U_au^tl ) <= \epsilon + (1 - y_t^l) M
                con_A_lry = gurobipy.LinExpr()
                con_A_lry.add(A_l)
                con_A_lry.add(-1 * R_l_a[1][target] * x[target][0])
                con_A_lry.add(-1 * R_l_a[0][target] * x[target][1])
                con_A_lry.add(-1 * M)
                con_A_lry.add(-1 * E[l])
                con_A_lry.add(M * y_l_list[target])
                model.addConstr(con_A_lry <= 0)
                model.update()

                # M (1 - y_t^l) + p_0^t U_dc^tl + q_0^t U_du^tl >= gamma^l
                con_M = gurobipy.LinExpr()
                con_M.add(M)
                con_M.add(-M * y_l_list[target])
                con_M.add(R_l_d[1][target] * x[target][0])
                con_M.add(R_l_d[0][target] * x[target][1])
                con_M.add(-1 * gamma_l)
                model.addConstr(con_M >= 0)
                model.update()

                # # M (1 - z_t^l) + p_0^t U_dc^tl + q_0^t U_du^tl >= gamma^l
                # con_M = gurobipy.LinExpr()
                # con_M.add(M)
                # con_M.add(-M * z_l_list[target])
                # con_M.add(R_l_d[1][target] * x[target][0])
                # con_M.add(R_l_d[0][target] * x[target][1])
                # con_M.add(-1 * gamma_l)
                # model.addConstr(con_M >= 0)
                # model.update()


                # build obj
                if obj_update_ind == True:
                    obj.add(x[target][2] * const_P[target] * np.mean(num_alerts_count, axis=1)[target] * const_C[target])
                    obj.add(x[target][3] * const_P[target] * np.mean(num_alerts_count, axis=1)[target] * const_C[target])

            obj_update_ind = False

        # Set objective funcion as all attackers have now been considered
        model.setObjective(obj, gurobipy.GRB.MAXIMIZE)
        model.setParam('IntFeasTol', 1e-9)

        model.update()

        start = datetime.now()

        # Solve MIQP
        model.Params.LogToConsole = 0
        model.optimize()

        run_time = (datetime.now() - start).total_seconds()

        # print("\nRuning time: ", (datetime.now() - start))

        solution = dict()
        for v in model.getVars():
            # print("%s -> %g" % (v.varName, v.x))
            solution[v.varName] = v.x

    except gurobipy.GurobiError:
        print("Error reported")

    z_list = [[solution['z_l0_T0'], solution['z_l0_T1'], solution['z_l0_T2'], solution['z_l0_T3'], solution['z_l0_T4'],
               solution['z_l0_T5'], solution['z_l0_T6']],
              [solution['z_l1_T0'], solution['z_l1_T1'], solution['z_l1_T2'], solution['z_l1_T3'], solution['z_l1_T4'],
               solution['z_l1_T5'], solution['z_l1_T6']]]

    z_list = np.round(z_list).tolist()

    y_list = [[solution['y_l0_T0'], solution['y_l0_T1'], solution['y_l0_T2'], solution['y_l0_T3'], solution['y_l0_T4'],
               solution['y_l0_T5'], solution['y_l0_T6']],
              [solution['y_l1_T0'], solution['y_l1_T1'], solution['y_l1_T2'], solution['y_l1_T3'], solution['y_l1_T4'],
               solution['y_l1_T5'], solution['y_l1_T6']]]

    y_list = np.round(y_list)

    name = ['p0_t', 'q0_t', 'p1_t', 'q1_t']

    type_within_epsilon_types = False

    if attacker_type == 0:
        if 1.0 in z_list[0]:
            opt_target = z_list[0].index(1.0)
            epsilon_range = len(np.where(y_list[0] == 1.0)[0])
            if type in np.where(y_list[0] == 1.0)[0]:
                type_within_epsilon_types = True
        else:
            usefulness = False
            opt_target = -1
    else:
        if 1.0 in z_list[1]:
            opt_target = z_list[1].index(1.0)
            epsilon_range = len(np.where(y_list[1] == 1.0)[0])
            if type in np.where(y_list[1] == 1.0)[0]:
                type_within_epsilon_types = True
        else:
            usefulness = False
            opt_target = -1

    auditor_util = []
    attacker_util = []
    for type_ in range(Target_num):
        practice_obj_left = solution[name[0] + str(type_)] * R_d[attacker_type][1][type_] + solution[
                                                                                              name[1] + str(type_)] * \
                                                                                          R_d[attacker_type][0][type_]
        practice_obj_right = 0
        for ele in range(Target_num):
            practice_obj_right += (solution[name[2] + str(ele)] + solution[name[3] + str(ele)]) * const_P[ele] * \
                                  np.mean(num_alerts_count, axis=1)[ele] * const_C[ele]
        practice_obj = practice_obj_left + practice_obj_right
        auditor_util.append(practice_obj)

        attacker_util_ele = solution[name[0] + str(type_)] * R_a[attacker_type][1][type_] + solution[
                                                                                              name[1] + str(type_)] * \
                                                                                          R_a[attacker_type][0][type_]
        attacker_util.append(attacker_util_ele)

    e_attacker_util = softmax(np.array(attacker_util) * lambda_p)
    # if e_attacker_util[type] >= 0.1 and usefulness is True:
    #     usefulness = True
    # else:
    #     usefulness = False

    # type = int(random.choices(list(np.linspace(0, Target_num - 1, Target_num)), e_attacker_util)[0])
    practice_obj = auditor_util[type]

    type_match_opt_type = False
    if opt_target == type:
        type_match_opt_type = True

    if epsilon_range <= 6:
        a = 1

    return solution, z_list, epsilon_range, type_within_epsilon_types, model.objVal, practice_obj, opt_target, type_match_opt_type, run_time, usefulness