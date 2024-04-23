# Experiments

Experiments on various topics, mainly machine learning exploration and software. No specific order, just a collection of topics that I needed specific refreshers on or wanted to explore further.

To run a lot of the code in these experiments, it is useful to know the development environment they were created in and the dependencies involved. Here is a list of the main tools and libraries used in my environment:

```python
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
```
Operating System: Darwin
OS Version: Darwin Kernel Version 23.4.0: Fri Mar 15 00:19:22 PDT 2024; root:xnu-10063.101.17~1/RELEASE_ARM64_T8112
Machine: arm64
Processor: arm
Python Version: 3.9.18 (main, Mar  8 2024, 18:13:35) 
[Clang 15.0.0 (clang-1500.3.9.4)]
Installed Machine Learning and Data Science packages:
gensim==4.3.2
ipython==8.18.1
jupyter-book==1.0.0
jupyterlab==4.1.4
kaggle==1.6.12
keras==3.0.5
matplotlib==3.8.3
nltk==3.8.1
numpy==1.26.4
pandas==2.2.1
plotly==5.20.0
scikit-learn==1.4.1.post1
scipy==1.12.0
seaborn==0.13.2
tensorflow-metal==1.1.0
tensorflow==2.16.1
xgboost==2.0.3
```

## Table of contents

```{tableofcontents}
```