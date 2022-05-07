import setuptools

with open("README.md", "r") as readme:
    long_desc = readme.read()

setuptools.setup(
    name="meta_act",
    version="2.0.0",
    author="Vinicius E. Martins",
    author_email="vini9x@gmail.com",
    description="Z-Value Meta-Recommender for Threshold based Active Learning",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    include_package_data=True,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Operating System :: GNU/Linux",
    ],
    install_requires=[
        "scipy == 1.5.*",
        "scikit-learn == 0.23.*",
        "scikit-multiflow == 0.5.*",
        "tsfel == 0.1.*",
        "pymfe == 0.*",
        "pandas == 1.1.2",
        "numpy == 1.19.2",
        "imbalanced-learn == 0.7.0",
    ],
    python_requires=">=3.5",
)
