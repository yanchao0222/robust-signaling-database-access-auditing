import pandas as pd
import numpy as np



def process_daily_data(start_index, end_index, iterate_time, iter):

    f = open("../date_list", "r")
    date_list = f.read().split(',')

    ################################################################
    # type 0:  5

    df0 = pd.read_csv('../../data/time_5.csv', header = None)
    df0['type'] = 0
    df0.columns = ['day_index', 'time' ,'type']
    df0['time'] = pd.to_datetime(df0['time'])
    df0 = df0.set_index('time')

    # train_data0 = df0[date_list[start_index + iter]:date_list[end_index + iter - 1]]
    test_data0 = df0[date_list[end_index + iter]:date_list[end_index + iter]]
    # test_data_time0 = np.squeeze(test_data0.index._values)
    # test_data_index0 = np.squeeze(test_data0._values)


    training_data_separate0 = []
    time_index0 = np.squeeze([i[0] for i in df0._values])
    # time0 = np.squeeze(df0.index._values)
    starting_point0 = [i for i, e in enumerate(time_index0) if e == 1]
    # separate training data by date
    for i in range(0, len(starting_point0) - iterate_time):
        training_data_separate0.append(df0[starting_point0[i + iter]:starting_point0[i + iter + 1]])
    # for i in range(0,len(starting_point0)-1):
    #     training_data_separate0.append(df0[starting_point0[i]:starting_point0[i+1]])

    ################################################################

    ################################################################
    # type 1:  13

    df1 = pd.read_csv('../../data/time_13.csv', header = None)
    df1['type'] = 1
    df1.columns = ['day_index', 'time', 'type']
    df1['time'] = pd.to_datetime(df1['time'])
    df1 = df1.set_index('time')
    # train_data1 = df1[date_list[start_index + iter]:date_list[end_index + iter - 1]]
    test_data1 = df1[date_list[end_index + iter]:date_list[end_index + iter]]
    # test_data_time1 = np.squeeze(test_data1.index._values)
    # test_data_index1 = np.squeeze(test_data1._values)


    training_data_separate1 = []
    time_index1 = np.squeeze([i[0] for i in df1._values])
    # time1 = np.squeeze(df1.index._values)
    starting_point1 = [i for i, e in enumerate(time_index1) if e == 1]
    # separate training data by date
    for i in range(0, len(starting_point1) - iterate_time):
        training_data_separate1.append(df1[starting_point1[i + iter]:starting_point1[i + iter + 1]])
    # for i in range(0,len(starting_point1)-1):
    #     training_data_separate1.append(df1[starting_point1[i]:starting_point1[i+1]])
    ################################################################

    ################################################################
    # type 2:  22

    df2 = pd.read_csv('../../data/time_22.csv', header = None)
    df2['type'] = 2
    df2.columns = ['day_index', 'time', 'type']
    df2['time'] = pd.to_datetime(df2['time'])
    df2 = df2.set_index('time')
    # train_data2 = df2[date_list[start_index + iter]:date_list[end_index + iter - 1]]
    test_data2 = df2[date_list[end_index + iter]:date_list[end_index + iter]]
    # test_data_time2 = np.squeeze(test_data2.index._values)
    # test_data_index2 = np.squeeze(test_data2._values)

    training_data_separate2 = []
    time_index2 = np.squeeze([i[0] for i in df2._values])
    # time2 = np.squeeze(df2.index._values)
    starting_point2 = [i for i, e in enumerate(time_index2) if e == 1]
    # separate training data by date
    for i in range(0, len(starting_point2) - iterate_time):
        training_data_separate2.append(df2[starting_point2[i + iter]:starting_point2[i + iter + 1]])
    # for i in range(0,len(starting_point2)-1):
    #     training_data_separate2.append(df2[starting_point2[i]:starting_point2[i+1]])
    ################################################################

    ################################################################
    # type 3:  18

    df3 = pd.read_csv('../../data/time_18.csv', header = None)
    df3['type'] = 3
    df3.columns = ['day_index', 'time', 'type']
    df3['time'] = pd.to_datetime(df3['time'])
    df3 = df3.set_index('time')
    # train_data3 = df3[date_list[start_index + iter]:date_list[end_index + iter - 1]]
    test_data3 = df3[date_list[end_index + iter]:date_list[end_index + iter]]
    # test_data_time3 = np.squeeze(test_data3.index._values)
    # test_data_index3 = np.squeeze(test_data3._values)

    training_data_separate3 = []
    time_index3 = np.squeeze([i[0] for i in df3._values])
    # time3 = np.squeeze(df3.index._values)
    starting_point3 = [i for i, e in enumerate(time_index3) if e == 1]
    # separate training data by date
    for i in range(0, len(starting_point3) - iterate_time):
        training_data_separate3.append(df3[starting_point3[i + iter]:starting_point3[i + iter + 1]])
    # for i in range(0,len(starting_point3)-1):
    #     training_data_separate3.append(df3[starting_point3[i]:starting_point3[i+1]])
    ################################################################

    ################################################################
    # tyoe 4:  5, 22

    df4 = pd.read_csv('../../data/time_5_22.csv', header = None)
    df4['type'] = 4
    df4.columns = ['day_index', 'time', 'type']
    df4['time'] = pd.to_datetime(df4['time'])
    df4 = df4.set_index('time')
    # train_data4 = df4[date_list[start_index + iter]:date_list[end_index + iter - 1]]
    test_data4 = df4[date_list[end_index + iter]:date_list[end_index + iter]]
    # test_data_time4 = np.squeeze(test_data4.index._values)
    # test_data_index4 = np.squeeze(test_data4._values)

    training_data_separate4 = []
    time_index4 = np.squeeze([i[0] for i in df4._values])
    # time4 = np.squeeze(df4.index._values)
    starting_point4 = [i for i, e in enumerate(time_index4) if e == 1]
    # separate training data by date
    for i in range(0, len(starting_point4) - iterate_time):
        training_data_separate4.append(df4[starting_point4[i + iter]:starting_point4[i + iter + 1]])
    # for i in range(0,len(starting_point4)-1):
    #     training_data_separate4.append(df4[starting_point4[i]:starting_point4[i+1]])
    ################################################################

    ################################################################
    # type 5:  5, 18

    df5 = pd.read_csv('../../data/time_5_18.csv', header = None)
    df5['type'] = 5
    df5.columns = ['day_index', 'time', 'type']
    df5['time'] = pd.to_datetime(df5['time'])
    df5 = df5.set_index('time')
    # train_data5 = df5[date_list[start_index + iter]:date_list[end_index + iter - 1]]
    test_data5 = df5[date_list[end_index + iter]:date_list[end_index + iter]]
    # test_data_time5 = np.squeeze(test_data5.index._values)
    # test_data_index5 = np.squeeze(test_data5._values)

    training_data_separate5 = []
    time_index5 = np.squeeze([i[0] for i in df5._values])
    # time5 = np.squeeze(df5.index._values)
    starting_point5 = [i for i, e in enumerate(time_index5) if e == 1]
    # separate training data by date
    for i in range(0, len(starting_point5) - iterate_time):
        training_data_separate5.append(df5[starting_point5[i + iter]:starting_point5[i + iter + 1]])
    # for i in range(0,len(starting_point5)-1):
    #     training_data_separate5.append(df5[starting_point5[i]:starting_point5[i+1]])
    ################################################################

    ################################################################
    # type 6:  5, 18, 22

    df6 = pd.read_csv('../../data/time_5_18_22.csv', header = None)
    df6['type'] = 6
    df6.columns = ['day_index', 'time', 'type']
    df6['time'] = pd.to_datetime(df6['time'])
    df6 = df6.set_index('time')
    # train_data6 = df6[date_list[start_index + iter]:date_list[end_index + iter - 1]]
    test_data6 = df6[date_list[end_index + iter]:date_list[end_index + iter]]
    # test_data_time6 = np.squeeze(test_data6.index._values)
    # test_data_index6 = np.squeeze(test_data6._values)

    training_data_separate6 = []
    time_index6 = np.squeeze([i[0] for i in df6._values])
    # time6 = np.squeeze(df6.index._values)
    starting_point6 = [i for i, e in enumerate(time_index6) if e == 1]
    # separate training data by date
    for i in range(0, len(starting_point6) - iterate_time):
        training_data_separate6.append(df6[starting_point6[i + iter]:starting_point6[i + iter + 1]])
    # for i in range(0,len(starting_point6)-1):
    #     training_data_separate6.append(df6[starting_point6[i]:starting_point6[i+1]])
    ################################################################


    training_data_separate = [training_data_separate0, training_data_separate1, training_data_separate2,
                              training_data_separate3, training_data_separate4, training_data_separate5,
                              training_data_separate6]
    test_data = pd.concat([test_data0, test_data1, test_data2, test_data3, test_data4, test_data5, test_data6],
                          ignore_index=False)
    test_data.sort_index(inplace=True)
    test_data_time = np.squeeze(test_data.index._values)
    test_data_index = np.squeeze(np.arange(1, test_data.shape[0] + 1))

    return training_data_separate, test_data, test_data_time, test_data_index