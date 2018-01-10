import numpy as np
import pickle
import argparse
import ConfigSpace as CS
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.ERROR)
from copy import deepcopy

from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario

import hpbandster.distributed.utils
from hpbandster.distributed.worker import Worker


def create_config_space():
    output_activation = CS.CategoricalHyperparameter("out_activation", ["softmax"], default_value='softmax')

    optimizer = CS.CategoricalHyperparameter("optimizer", ["Adam", "SGD"], default_value='Adam')

    layers = CS.UniformIntegerHyperparameter("layers", lower=1, default_value=2, upper=4)

    num_units_0 = CS.UniformIntegerHyperparameter("num_units_0", lower=16, default_value=32, upper=256, log=True)

    num_units_1 = CS.UniformIntegerHyperparameter("num_units_1", lower=16, default_value=32, upper=256, log=True)

    cond_num_units1 = CS.GreaterThanCondition(num_units_1, layers, 2)

    num_units_2 = CS.UniformIntegerHyperparameter("num_units_2", lower=16, default_value=32, upper=256, log=True)

    cond_num_units2 = CS.GreaterThanCondition(num_units_2, layers, 3)

    num_units_3 = CS.UniformIntegerHyperparameter("num_units_3", lower=16, default_value=32, upper=256, log=True)

    cond_num_units3 = CS.EqualsCondition(num_units_3, layers, 4)

    loss_function = CS.CategoricalHyperparameter("loss_function", ["categorical_crossentropy"], default_value='categorical_crossentropy')

    learning_rate_schedule = CS.CategoricalHyperparameter("learning_rate_schedule", ["ExponentialDecay", "StepDecay"], default_value='ExponentialDecay')

    l2_reg_0 = CS.UniformFloatHyperparameter('l2_reg_0', lower=1e-6, upper=1e-2, default_value=1e-4, log=True)

    l2_reg_1 = CS.UniformFloatHyperparameter('l2_reg_1', lower=1e-6, upper=1e-2, default_value=1e-4, log=True)

    cond_l2_reg1 = CS.GreaterThanCondition(l2_reg_1, layers, 2)

    l2_reg_2 = CS.UniformFloatHyperparameter('l2_reg_2', lower=1e-6, upper=1e-2, default_value=1e-4, log=True)

    cond_l2_reg2 = CS.GreaterThanCondition(l2_reg_2, layers, 3)

    l2_reg_3 = CS.UniformFloatHyperparameter('l2_reg_3', lower=1e-6, upper=1e-2, default_value=1e-4, log=True)

    cond_l2_reg3 = CS.EqualsCondition(l2_reg_3, layers, 4)

    dropout_0 = CS.UniformFloatHyperparameter('dropout_0', lower=0.0, upper=0.5, default_value=0.0, log=False)

    dropout_1 = CS.UniformFloatHyperparameter('dropout_1', lower=0.0, upper=0.5, default_value=0.0, log=False)

    cond_dropout_1 = CS.GreaterThanCondition(dropout_1, layers, 2)

    dropout_2 = CS.UniformFloatHyperparameter('dropout_2', lower=0.0, upper=0.5, default_value=0.0, log=False)

    cond_dropout_2 = CS.GreaterThanCondition(dropout_2, layers, 2)

    dropout_3 = CS.UniformFloatHyperparameter('dropout_3', lower=0.0, upper=0.5, default_value=0.0, log=False)

    cond_dropout_3 = CS.EqualsCondition(dropout_3, layers, 4)

    batch_size = CS.UniformIntegerHyperparameter("batch_size", lower=8, default_value=16, upper=256, log=True)

    activation = CS.CategoricalHyperparameter("activation", ["relu", "tanh"], default_value='relu')

    step_decay_epochs = CS.UniformIntegerHyperparameter("StepDecay epochs per step", lower=1, default_value=16, upper=128, log=True)

    cond_step_decay_epochs = CS.EqualsCondition(step_decay_epochs, learning_rate_schedule, "StepDecay")

    sgd_momentum = CS.UniformFloatHyperparameter('sgd_momentum', lower=0.0, upper=0.99, default_value=0.9, log=False)

    cond_sgd_momentum = CS.EqualsCondition(sgd_momentum, optimizer, "SGD")

    sgd_initial_lr = CS.UniformFloatHyperparameter('sgd_initial_lr', lower=1e-3, upper=0.5, default_value=1e-1, log=True)

    cond_sgd_initial_lr = CS.EqualsCondition(sgd_initial_lr, optimizer, "SGD")

    sgd_final_lr_fraction = CS.UniformFloatHyperparameter('sgd_final_lr_fraction', lower=1e-4, upper=1.0, default_value=1e-2, log=True)

    cond_sgd_final_lr_fraction = CS.EqualsCondition(sgd_final_lr_fraction, optimizer, "SGD")

    adam_initial_lr = CS.UniformFloatHyperparameter('adam_initial_lr', lower=1e-4, upper=1e-2, default_value=1e-3, log=True)

    cond_adam_initial_lr = CS.EqualsCondition(adam_initial_lr, optimizer, "Adam")

    adam_final_lr_fraction = CS.UniformFloatHyperparameter('adam_final_lr_fraction', lower=1e-4, upper=1.0, default_value=1e-2, log=True)

    cond_adam_final_lr_fraction = CS.EqualsCondition(adam_final_lr_fraction, optimizer, "Adam")


    cs = CS.ConfigurationSpace()

    cs.add_hyperparameter(output_activation)

    cs.add_hyperparameter(optimizer)

    cs.add_hyperparameter(layers)

    cs.add_hyperparameter(num_units_0)

    cs.add_hyperparameter(num_units_1)

    cs.add_condition(cond_num_units1)

    cs.add_hyperparameter(num_units_2)

    cs.add_condition(cond_num_units2)

    cs.add_hyperparameter(num_units_3)

    cs.add_condition(cond_num_units3)

    cs.add_hyperparameter(loss_function)

    cs.add_hyperparameter(learning_rate_schedule)

    cs.add_hyperparameter(l2_reg_0)

    cs.add_hyperparameter(l2_reg_1)

    cs.add_condition(cond_l2_reg1)

    cs.add_hyperparameter(l2_reg_2)

    cs.add_condition(cond_l2_reg2)

    cs.add_hyperparameter(l2_reg_3)

    cs.add_condition(cond_l2_reg3)

    cs.add_hyperparameter(dropout_0)

    cs.add_hyperparameter(dropout_1)

    cs.add_condition(cond_dropout_1)

    cs.add_hyperparameter(dropout_2)

    cs.add_condition(cond_dropout_2)

    cs.add_hyperparameter(dropout_3)

    cs.add_condition(cond_dropout_3)

    cs.add_hyperparameter(batch_size)

    cs.add_hyperparameter(activation)

    cs.add_hyperparameter(step_decay_epochs)

    cs.add_condition(cond_step_decay_epochs)

    cs.add_hyperparameter(sgd_momentum)

    cs.add_condition(cond_sgd_momentum)

    cs.add_hyperparameter(sgd_initial_lr)

    cs.add_condition(cond_sgd_initial_lr)

    cs.add_hyperparameter(sgd_final_lr_fraction)

    cs.add_condition(cond_sgd_final_lr_fraction)

    cs.add_hyperparameter(adam_initial_lr)

    cs.add_condition(cond_adam_initial_lr)

    cs.add_hyperparameter(adam_final_lr_fraction)

    cs.add_condition(cond_adam_final_lr_fraction)

    return cs


def objective_function(config, epoch=127, **kwargs):
    # Cast the config to an array such that it can be forwarded to the surrogate
    x = deepcopy(config.get_array())
    x[np.isnan(x)] = -1
    lc = rf.predict(x[None, :])[0]
    c = cost_rf.predict(x[None, :])[0]

    return lc[epoch], {"cost": c, "learning_curve": lc[:epoch].tolist()}


class WorkerWrapper(Worker):
    def compute(self, config, budget, *args, **kwargs):
        cfg = CS.Configuration(cs, values=config)
        loss, info = objective_function(cfg, epoch=int(budget))

        return ({
            'loss': loss,
            'info': {"runtime": info["cost"],
                     "lc": info["learning_curve"]}
        })

def plot_graph(x, y, graph_label='', x_lbl='', y_lbl='', title=''):
    plt.plot(x, y, label=graph_label)
    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.legend(loc='lower right')
    plt.title(title)
    plt.xlim(xmin=0)


def show_graph():
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_smac', action='store_true')
    parser.add_argument('--run_hyperband', action='store_true')
    parser.add_argument('--n_iters', default=50, type=int)
    args = vars(parser.parse_args())

    n_iters = args['n_iters']

    cs = create_config_space()
    rf = pickle.load(open("rf_surrogate_paramnet_mnist.pkl", "rb"))
    cost_rf = pickle.load(open("rf_cost_surrogate_paramnet_mnist.pkl", "rb"))

    if args["run_smac"]:
        scenario = Scenario({"run_obj": "quality",
                             "runcount-limit": n_iters,
                             "cs": cs,
                             "deterministic": "true",
                             "output_dir": ""})

        smac = SMAC(scenario=scenario, tae_runner=objective_function)
        smac.optimize()

        # The following lines extract the incumbent strategy and the estimated wall-clock time of the optimization
        rh = smac.runhistory
        incumbents = []
        incumbent_performance = []
        inc = None
        inc_value = 1
        idx = 1
        t = smac.get_trajectory()

        wall_clock_time = []
        cum_time = 0
        for d in rh.data:
            cum_time += rh.data[d].additional_info["cost"]
            wall_clock_time.append(cum_time)
        for i in range(n_iters):

            if idx < len(t) and i == t[idx].ta_runs - 1:
                inc = t[idx].incumbent
                inc_value = t[idx].train_perf
                idx += 1

            incumbents.append(inc)
            incumbent_performance.append(inc_value)

        # plot validation error against wall clock time
        plot_graph(x=wall_clock_time, y=incumbent_performance, x_lbl='time',
                   y_lbl='validation error', title='SMAC')
        show_graph()

        lc_smac = []
        for i, d in enumerate(rh.data):
            data = rh.data[d].additional_info["learning_curve"]
            lc_smac.append(data)
            x_axis = range(len(data))

            # plot all learning curves
            plot_graph(x=x_axis, y=data, y_lbl='error', x_lbl='iterations',
                       graph_label='Curve '+ str(i+1), title='Learning Curves SMAC')
        show_graph()

    if args["run_hyperband"]:
        nameserver, ns_port = hpbandster.distributed.utils.start_local_nameserver()

        # starting the worker in a separate thread
        w = WorkerWrapper(nameserver=nameserver, ns_port=ns_port)
        w.run(background=True)

        CG = hpbandster.config_generators.RandomSampling(cs)

        # instantiating Hyperband with some minimal configuration
        HB = hpbandster.HB_master.HpBandSter(
            config_generator=CG,
            run_id='0',
            eta=2,  # defines downsampling rate
            min_budget=1,  # minimum number of epochs / minimum budget
            max_budget=127,  # maximum number of epochs / maximum budget
            nameserver=nameserver,
            ns_port=ns_port,
            job_queue_sizes=(0, 1),
        )
        # runs one iteration if at least one worker is available
        res = HB.run(10, min_n_workers=1)

        # shutdown the worker and the dispatcher
        HB.shutdown(shutdown_workers=True)

        # extract incumbent trajectory and all evaluated learning curves
        traj = res.get_incumbent_trajectory()
        wall_clock_time = []
        cum_time = 0

        for c in traj["config_ids"]:
            cum_time += res.get_runs_by_id(c)[-1]["info"]["runtime"]
            wall_clock_time.append(cum_time)

        lc_hyperband = []
        for i, r in enumerate(res.get_all_runs()):
            c = r["config_id"]
            data = res.get_runs_by_id(c)[-1]["info"]["lc"]
            lc_hyperband.append(data)
            x_axis = range(len(data))

            # plot all learning curves
            plot_graph(x=x_axis, y=data, y_lbl='error', x_lbl='iterations',
                       graph_label='Curve ' + str(i + 1), title='Learning Curves Hyperband')
        show_graph()


        # plot validation error
        incumbent_performance = traj["losses"]
        plot_graph(x=wall_clock_time, y=incumbent_performance,
                   x_lbl='time', y_lbl='validation error', title='Hyperband')
        show_graph()