import os


def pytest_cmdline_main(config):
    args = list(config.invocation_params.args)
    if args or os.environ.get('PFDSIM_PARALLEL_DEFAULT_ACTIVE'):
        return None

    if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
        from .tests.default_parallel import run_parallel_default
    else:
        from tests.default_parallel import run_parallel_default

    return run_parallel_default()
