from setuptools import setup, find_packages

setup(
    name="plauthor",
    version="0.1.2b",
    description="Plauthor: Machine Learning Focused Interface for Dataframe-related Visualizations",
    url="https://github.com/shayanfazeli/plauthor",
    author="Shayan Fazeli",
    author_email="shayan@cs.ucla.edu",
    license="Apache",
    classifiers=[
          'Intended Audience :: Science/Research',
          'Development Status :: 1',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ],
    keywords="plauthor machine learning dataframe label plot graphics grammer ggplot matplotlib",
    packages=find_packages(),
    python_requires='>3.6.0',
    install_requires=[
        'numpy',
        'pandas',
        'sklearn',
        'plotnine',
    ],
    include_package_data=True,
    zip_safe=False
)