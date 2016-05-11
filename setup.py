from setuptools import setup
#from distutils.core import setup


config = {
    'name': 'transport',  # Replace with project name
    'version': '0.0',  # Replace with module_name.__version__
    'url': '',  # Replace with url to github
    'description': 'EE 546 Project',  # Replace with project description
    'author': 'Christopher Aicher',
    'license': 'license',
    'packages': ['transport'],  # Replace with package names
    'ext_modules': [], # Cythonized Packages
    'scripts': [], # Scripts with #!/usr/bin/env python
}

setup(**config)
# Develop: python setup.py develop
# Remove: python setup.py develop --uninstall
