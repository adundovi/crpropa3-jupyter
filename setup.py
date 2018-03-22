from setuptools import find_packages, setup

setup(name="crpropa3jupyter",
      version="0.1",
      description="CRPropa addons and helpers for Jupyter notebooks",
      author="Andrej Dundovic",
      author_email='andrej@dundovic.com.hr',
      platforms=["linux"],
      license="MIT",
      url="",
      keywords="crpropa jupyter",
      packages=find_packages(),
      install_requires=[
            "jupyter",
            "matplotlib",
            "numpy",
            "ipyvolume",
            "pint",
            "healpy",
        ],
     )
