from setuptools import setup

setup(name='purepos',
      version='0.1',
      description='Python wrapper for PurePos',
      long_description=open('README.md').read(),
      license="LGPL 3",
      author='György Orosz',
      author_email='oroszgy@gmail.com',
      url='https://github.com/ppke-nlpg/purepos.py',
      packages=['purepos'],
      package_dir={'purepos': 'src/purepos'},
      #eager_resources = ["src/purepos/purepos-2.0.one-jar.jar"],
      package_data = {'':['*.jar']},
      entry_points = {
        'console_scripts': ['purepos=purepos.purepos:main'],
        }
)