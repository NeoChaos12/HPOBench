import logging

import time
from typing import Union, Tuple, Dict, List, Optional, Sequence
import ConfigSpace as CS
import numpy as np
import cocoex

from hpobench.abstract_benchmark import AbstractBenchmark

__version__ = '0.0.1'
logger = logging.getLogger('COCOBenchmark')


class BBOBBenchmark(AbstractBenchmark):
    """ Wraps up a single problem instance from either the "bbob-mixint" or the "bbob-biobj-mixint" BBOB test function
    suites into an HPOBench-like benchmark object. Due to the dynamic nature of the test functions' configuration
    space, it is not possible to access the Configuration Space without first initializing the benchmark object. """

    def __init__(self, suite: str, func_idx: int, instance: Optional[str] = "", suite_options: Optional[str] = "",
                 fake_fidelity: bool = False, rng: Union[np.random.RandomState, int, None] = None):
        logger.debug("Initializing BBOB Benchmark suite %s, choosing problem at index %d, filtering using instance %s "
                     "and suite_options %s. Fake fidelity is %s." % (suite, func_idx, instance, suite_options,
                                                                     fake_fidelity))
        self.suite = suite
        self.func_idx = func_idx
        self.instance = instance
        self.suite_options = suite_options
        self.suite_object = cocoex.Suite(self.suite, self.instance, self.suite_options)
        self.problem = self.suite_object[self.func_idx]
        self._config_params = self._generate_parameters(self.problem)
        self._fidelity_params = [CS.CategoricalHyperparameter("FakeFidelity", [True,])] if fake_fidelity else None

        super(BBOBBenchmark, self).__init__(rng)
        logger.info("Initialized BBOB Synthetic Benchmark.")

    def _generate_parameters(self, problem) -> Sequence[CS.hyperparameters.Hyperparameter]:
        n = problem.dimension
        k = problem.number_of_integer_variables
        lb = problem.lower_bounds
        ub = problem.upper_bounds
        default = problem.initial_solution_proposal()
        logger.debug("Generating %d total parameters, of which the first %d will be integers." % (n, k))

        params = [None] * n
        # k integer parameters followed by n-j continuous ones
        for i in range(k):
            logger.debug("Generating parameter %d, an integer parameter with lower bound %d, upper bound %d and "
                         "default value %d." % (i+1, lb[i], ub[i], default[i]))
            params[i] = CS.UniformIntegerHyperparameter(f"Int_{i}", lower=lb[i], upper=ub[i], default_value=default[i])

        for i in range(k, n):
            logger.debug("Generating parameter %d, a continuous parameter with lower bound %f, upper bound %f and "
                         "default value %f." % (i+1, lb[i], ub[i], default[i]))
            params[i] = CS.UniformFloatHyperparameter(f"Cont_{i}", lower=lb[i], upper=ub[i], default_value=default[i])

        return params

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        csp = CS.ConfigurationSpace(name="BBOB Problem Instance Configuration Space", seed=seed)
        csp.add_hyperparameters(self._config_params)
        return csp

    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fsp = CS.ConfigurationSpace(name="BBOB Problem Instance Fidelity Space", seed=seed)
        if self._fidelity_params is not None:
            fsp.add_hyperparameters(self._fidelity_params)
        return fsp

    @AbstractBenchmark.check_parameters
    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:
        """ Translates a call to the benchmark into a call to the underlying problem instance. Note that the
        fidelity, rng and kwargs are present only to comply with the API and are not actually used.  """

        if isinstance(configuration, CS.Configuration):
            configuration = self.problem(configuration.get_dictionary().values())

        c = [configuration.get(p.name) for p in self._config_params]
        val = self.problem(c)
        return {
            'function_value': val,
            'cost': 1.,  # Constant cost of evaluation, always
            'info': {
                'rng': rng,
                'config': configuration,
                'fidelity': fidelity
            }
        }


    @AbstractBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[Dict, CS.Configuration, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None,
                                **kwargs) -> Dict:
        return self.objective_function(self, configuration, fidelity, rng, **kwargs)


    def get_meta_information(self):
        """ Returns the meta information for the benchmark """
        return {'name': 'BBOB Test Suite',
                'references': [],
                'initial random seed': self.rng,
                'suite': self.suite,
                'func_idx': self.task_id,
                'instance': self.instance,
                'suite_options': self.suite_options
                }


class BBOBMixIntBenchmark(BBOBBenchmark):
    def __init__(self, func_idx: int, instance: Optional[str] = "", suite_options: Optional[str] = "",
                 fake_fidelity: bool = False, rng: Union[np.random.RandomState, int, None] = None):
        super(BBOBMixIntBenchmark, self).__init__("bbob-mixint", func_idx, instance, suite_options, fake_fidelity, rng)


class BBOBBiobjMixIntBenchmark(BBOBBenchmark):
    def __init__(self, func_idx: int, instance: Optional[str] = "", suite_options: Optional[str] = "",
                 fake_fidelity: bool = False, rng: Union[np.random.RandomState, int, None] = None):
        super(BBOBBiobjMixIntBenchmark, self).__init__("bbob-biobj-mixint", func_idx, instance, suite_options,
                                                       fake_fidelity, rng)