from setuptools import setup, find_namespace_packages

setup(name='dynamic_network_architectures',
      packages=find_namespace_packages(include=["dynamic_network_architectures", "dynamic_network_architectures.*"]),
      version='0.3.1',
      description='none',
      install_requires=[
            "torch>=1.6.0a",
            "numpy"
            ],
      zip_safe=False)
