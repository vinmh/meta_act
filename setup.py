import setuptools

with open("README.md", "r") as readme:
    long_desc = readme.read()

setuptools.setup(
    name="meta_act",
    version="0.0.3",
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
    python_requires=">=3.5",
)
