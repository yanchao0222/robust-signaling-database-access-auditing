import numpy as np
import datetime
import math
import scipy.stats as st


def prob_being_caught(training_data_separate, test_data_time, type_list, i, prob_coeff_being_caught_online_sig_last_iter, gate_for_insert, B_avail_online_sig, residual):

    alert_time = str(test_data_time[i - 1])
    alert_time = alert_time[11:19]
    hour_current = alert_time[0:2]
    minute_current = alert_time[3:5]
    # seconds = alert_time[6:8]
    type = type_list[i - 1]

    # attacker_type = np.random.choice(2, 1, p=[0.5, 0.5])[0]

    print(alert_time)
    current_time = datetime.time(hour=int(hour_current), minute=int(minute_current), second=0, microsecond=0)

    # print("i = " + str(i))
    num_alerts_count = []
    prob_coeff_being_caught_online_sig = []
    type_index = -1

    for type_data in training_data_separate:
        type_index = type_index + 1
        num_alerts_count_type = []
        for day in type_data:
            temp = day.index._values
            temp = str(temp[0])
            date = temp[0:10]
            # alert_time
            time_to_start = date + ' ' + alert_time

            time_to_end = datetime.datetime.strptime(date, '%Y-%m-%d') + datetime.timedelta(days=1)
            block = day[time_to_start:time_to_end]
            num_alerts_count_type.append(block.shape[0])

        num_alerts_count.append(num_alerts_count_type)
        num_range_upper = np.mean(num_alerts_count_type)

        if num_range_upper <= 4:
            prob_coeff_being_caught_online_sig.append(prob_coeff_being_caught_online_sig_last_iter[type_index])
        else:
            upper_point = math.ceil(2 * np.mean(num_alerts_count_type))
            num_range = range(1, upper_point)
            rv = st.poisson(np.mean(num_alerts_count_type))
            # prob_num_range = st.norm.pdf(num_range,np.mean(num_alerts_count), np.std(num_alerts_count))

            prob_num_range_type = [rv.pmf(j) for j in num_range]
            prob_coeff_being_caught_online_sig.append(
                np.dot(np.array([1 / float(ele) for ele in num_range]), np.array(prob_num_range_type)))

    prob_coeff_being_caught_online_sig_last_iter = prob_coeff_being_caught_online_sig

    if current_time >= datetime.time(hour=15, minute=0, second=0, microsecond=0) and gate_for_insert == 1:
        # B_avail = B_avail + residual1
        B_avail_online_sig = B_avail_online_sig + residual[0]
        print("The 1st insertion.")
        gate_for_insert = 2

    if current_time >= datetime.time(hour=17, minute=0, second=0, microsecond=0) and gate_for_insert == 2:
        # B_avail = B_avail + residual2
        B_avail_online_sig = B_avail_online_sig + residual[1]
        print("The 2nd insertion.")
        gate_for_insert = 3

    if current_time >= datetime.time(hour=18, minute=30, second=0, microsecond=0) and gate_for_insert == 3:
        # B_avail = B_avail + residual3
        B_avail_online_sig = B_avail_online_sig + residual[2]
        print("The 3rd insertion.")
        gate_for_insert = 4

    if current_time >= datetime.time(hour=19, minute=0, second=0, microsecond=0) and gate_for_insert == 4:
        # B_avail = B_avail + residual4
        B_avail_online_sig = B_avail_online_sig + residual[3]
        print("The 4th insertion.")
        gate_for_insert = 5

    if current_time >= datetime.time(hour=20, minute=0, second=0, microsecond=0) and gate_for_insert == 5:
        # B_avail = B_avail + residual5
        B_avail_online_sig = B_avail_online_sig + residual[4]
        print("The 5th insertion.")
        gate_for_insert = 6

    if current_time >= datetime.time(hour=21, minute=20, second=0, microsecond=0) and gate_for_insert == 6:
        # B_avail = B_avail + residual6
        B_avail_online_sig = B_avail_online_sig + residual[5]
        print("The 6th insertion.")
        gate_for_insert = 7

    if current_time >= datetime.time(hour=22, minute=20, second=0, microsecond=0) and gate_for_insert == 7:
        # B_avail = B_avail + residual7
        B_avail_online_sig = B_avail_online_sig + residual[6]
        print("The 7th insertion.")
        gate_for_insert = 8


    for ele in prob_coeff_being_caught_online_sig:
        assert ele >= 0

    return prob_coeff_being_caught_online_sig_last_iter, prob_coeff_being_caught_online_sig, num_alerts_count, B_avail_online_sig, type, gate_for_insert #, attacker_type