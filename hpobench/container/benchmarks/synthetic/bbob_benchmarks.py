#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Container client for the BBOB Benchmarks from hpobench/benchmarks/synthetic/bbob_benchmarks """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


class BBOBMixIntBenchmark(AbstractBenchmarkClient):
    def __init__(self, func_idx: int, **kwargs):
        kwargs['func_idx'] = func_idx
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'BBOBMixIntBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'bbob_benchmarks')
        super(BBOBMixIntBenchmark, self).__init__(**kwargs)


class BBOBBiobjMixIntBenchmark(AbstractBenchmarkClient):
    def __init__(self, func_idx: int, **kwargs):
        kwargs['func_idx'] = func_idx
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'BBOBBiobjMixIntBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'bbob_benchmarks')
        super(BBOBBiobjMixIntBenchmark, self).__init__(**kwargs)
