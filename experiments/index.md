---
file_format: mystnb
kernelspec:
  name: python3
---

# Experiments

Experiments on various topics, mainly machine learning exploration and software. No specific order, just a collection of topics that I needed specific refreshers on or wanted to explore further.

To run a lot of the code in these experiments, it is useful to know the development environment they were created in and the dependencies involved. Here is a list of the main tools and libraries used in my environment:

```{code-cell} ipython3
:tags: [hide-input]
import platform
import sys
import pkg_resources

def get_system_info():
    print("Operating System:", platform.system())
    print("OS Version:", platform.version())
    print("Machine:", platform.machine())
    print("Processor:", platform.processor())

def get_python_version():
    print("Python Version:", sys.version)

def get_pip_packages():
    # List of common machine learning and data science packages
    ml_ds_packages = [
        'numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn', 
        'sklearn', 'scikit-learn', 'tensorflow', 'keras', 'pytorch', 
        'torch', 'xgboost', 'lightgbm', 'catboost', 'statsmodels', 
        'nltk', 'spacy', 'gensim', 'theano', 'plotly', 'dash', 
        'flask', 'streamlit', 'jupyter', 'ipython', 'sympy', 'tensorflow-metal',
        'kaggle', 'jupyterlab', 'jupyter-book'
    ]
    
    installed_packages = pkg_resources.working_set
    installed_packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages if i.key in ml_ds_packages])
    print("Installed Machine Learning and Data Science packages:")
    for package in installed_packages_list:
        print(package)

get_system_info()
get_python_version()
get_pip_packages()
```

## Table of contents

```{tableofcontents}
```