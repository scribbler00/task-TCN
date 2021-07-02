from setuptools import setup,find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='fore',
      version='0.1.0a0',
      description='Preprocessing for Renewable Energie.',
      long_description=readme(),
      url='https://git.ies.uni-kassel.de/jens/renewable_energie_preprocessing',
      keywords='machine learning energy timeseries',
      author='Jens Schreiber, Janosch Henze',
      author_email='iescloud@uni-kassel.de',
      packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
      install_requires=[
          'numpy',
          'sklearn',
          'scipy',
	      'pandas',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      classifiers=[
      'Development Status :: 3 - Alpha',
      'Programming Language :: Python :: 3',
      'Intended Audience :: Developers',
      ]
      )

