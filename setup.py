from setuptools import setup

setup(name = 'Furby_p3',
    version = '1.0',
    packages = ['Furby_p3',
                'Furby_p3.tests'],
    scripts = ['bin/gen_furby.py', 'bin/check_furby.py'],
    install_requires = ['numpy', 'matplotlib', 'scipy'],
    )
