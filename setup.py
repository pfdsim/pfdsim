from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        if package != "pfdsim":
            return modules

        excluded = {"conftest", "gunicorn.conf", "setup"}
        return [
            module
            for module in modules
            if module[1] not in excluded
        ]


setup(cmdclass={"build_py": build_py})
