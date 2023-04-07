from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = [
                "numpy==1.23.5",
                "pandas==1.5.3",
                "scikit-learn==1.2.1",
                "umap-learn==0.5.3 ",
                "pickleshare==0.7.5",
                "openpyxl==3.1.2"
                ]

setup(
        name = "TEPaat",
        version = "1.0",
        author = "Amanda Ana & Teslim",
        author_email = 'tolayi1@lsu.edu',
        description = 'TEPaat: A python framework to perform dimensionality reduction and classification of a given process data',
        long_description=readme,
        long_description_content_type = 'text/markdown',
        url = 'https://github.com/anabenitoterreros/TEPaat',
        packages = find_packages(),
        include_package_data = True,
        install_requires = requirements,
        classifiers = [
            'Programming Language :: Python :: 3.8.5'
            ]
    )
