from qcodes.dataset.measurements import Measurement
import numpy as np
from qcodes.instrument.specialized_parameters import ElapsedTimeParameter
from time import sleep
from tqdm import tqdm


def do1d(param_set, start, stop, num_points, delay, *param_meas):
    meas = Measurement()
    meas.register_parameter(param_set)  # register the first independent parameter
    output = []
    param_set.post_delay = delay
    # do1D enforces a simple relationship between measured parameters
    # and set parameters. For anything more complicated this should be reimplemented from scratch
    for parameter in param_meas:
        meas.register_parameter(parameter, setpoints=(param_set,))
        output.append([parameter, None])

    with meas.run() as datasaver:
        for set_point in tqdm(np.linspace(start, stop, num_points)):
            param_set.set(set_point)
            for i, parameter in enumerate(param_meas):
                output[i][1] = parameter.get()
            datasaver.add_result((param_set, set_point),
                                 *output)
    dataid = datasaver.run_id  # convenient to have for plotting
    return dataid


def do2d(param_set1, start1, stop1, num_points1, delay1,
         param_set2, start2, stop2, num_points2, delay2,
         *param_meas):
    # And then run an experiment

    meas = Measurement()
    meas.register_parameter(param_set1)
    param_set1.post_delay = delay1
    meas.register_parameter(param_set2)
    param_set2.post_delay = delay2
    output = []
    for parameter in param_meas:
        meas.register_parameter(parameter, setpoints=(param_set1,param_set2))
        output.append([parameter, None])

    with meas.run() as datasaver:
        for set_point1 in tqdm(np.linspace(start1, stop1, num_points1), desc='first parameter'):
            param_set1.set(set_point1)
            for set_point2 in  tqdm(np.linspace(start2, stop2, num_points2), desc='nested  parameter', leave=False):
                param_set2.set(set_point2)
                for i, parameter in enumerate(param_meas):
                    output[i][1] = parameter.get()
                datasaver.add_result((param_set1, set_point1),
                                     (param_set2, set_point2),
                                     *output)
            param_set2.set(start2)
    dataid = datasaver.run_id  # convenient to have for plotting
    return dataid

def do2ddyn(param_set1, start1, stop1, num_points1, dyn_param, dyn_value1,dyn_value2, delay1,
         param_set2, start2, stop2, num_points2, delay2,
         *param_meas):
    # And then run an experiment

    meas = Measurement()
    meas.register_parameter(param_set1)
    meas.register_parameter(dyn_param)
    param_set1.post_delay = delay1
    meas.register_parameter(param_set2)
    param_set2.post_delay = delay2
    output = []
    for parameter in param_meas:
        meas.register_parameter(parameter, setpoints=(param_set1,param_set2))
        output.append([parameter, None])

    with meas.run() as datasaver:
        for set_point1 in tqdm(np.linspace(start1, stop1, num_points1), desc='first parameter'):
            param_set1.set(set_point1)
            dyn_set = set_point1*dyn_value1+dyn_value2
            dyn_param.set(dyn_set)
            for set_point2 in  tqdm(np.linspace(start2, stop2, num_points2), desc='nested  parameter', leave=False):
                param_set2.set(set_point2)
                for i, parameter in enumerate(param_meas):
                    output[i][1] = parameter.get()
                datasaver.add_result((param_set1, set_point1),
                                     (dyn_param, dyn_set),
                                     (param_set2, set_point2),
                                     *output)
            param_set2.set(start2)
    dataid = datasaver.run_id  # convenient to have for plotting
    return dataid


def time_sweep(num_points, delay, *param_meas):
    print('ENTERED TIME SWEEP')
    time = ElapsedTimeParameter('time')
    meas = Measurement()
    meas.register_parameter(time)
    output = []
    for parameter in param_meas:
        meas.register_parameter(parameter, setpoints=(time,))
        output.append([parameter, None])

    with meas.run() as datasaver:
        time.reset_clock()
        for point in tqdm(range(num_points),position=0,leave=False):
            for i, parameter in enumerate(param_meas):
                output[i][1] = parameter.get()
            now = time()
            datasaver.add_result((time, now),
                                 *output)
            sleep(delay)

    dataid = datasaver.run_id  # convenient to have for plotting
    return dataid

def measure_until(param_ind, exit_condition, delay, *param_meas):

    meas = Measurement()
    meas.register_parameter(param_ind)
    output = []

    for parameter in param_meas:
        meas.register_parameter(parameter, setpoints=(param_ind,))
        output.append([parameter, None])

    with meas.run() as datasaver:
        while not exit_condition(param_ind, param_meas):
            for i, parameter in enumerate(param_meas):
                output[i][1] = parameter.get()
            datasaver.add_result((param_ind, param_ind.get()),
                                 *output)
            sleep(delay)

    dataid = datasaver.run_id  # convenient to have for plotting
    return dataid

def do1d_until(param_ind,  exit_condition, delay1,
         param_set, start, stop, num_points, delay2,
         *param_meas):
    meas = Measurement()
    meas.register_parameter(param_ind)
    meas.register_parameter(param_set)
    param_set.post_delay = delay2
    output = []
    for parameter in param_meas:
        meas.register_parameter(parameter, setpoints=(param_ind, param_set))
        output.append([parameter, None])

    with meas.run() as datasaver:
         while not exit_condition(param_ind, param_set, param_meas):
            set_point1 = param_ind.get()
            for set_point2 in np.linspace(start, stop, num_points):
                param_set.set(set_point2)
                for i, parameter in enumerate(param_meas):
                    output[i][1] = parameter.get()
                datasaver.add_result((param_ind, set_point1),
                                     (param_set, set_point2),
                                     *output)
            sleep(delay1)
    dataid = datasaver.run_id  # convenient to have for plotting
    return dataid