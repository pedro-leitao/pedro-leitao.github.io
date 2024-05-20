---
myst:
    html_meta:
        description: "A guide on setting up a machine learning environment on Apple Silicon, specifically for Python and TensorFlow. In an attempt to keep things simple, this guide relies only on the standard Python distribution rather than Anaconda, Miniconda or other Python distributions."
        keywords: "Machine Learning, Apple Silicon, M2, M1, M3, Python, TensorFlow, TensorFlow-Metal, Jupyter Lab, Homebrew, Python 3.9, virtual environment, UMA, NUMA, GPU, Apple, Silicon, M1, M2, M3, Pedro, Pedro Leitao"
---

# Setting up a Machine Learning environment on Apple Silicon

All the kids now use cloud machine learning platforms like Google Colab, Kaggle, or AWS. But if you want to do some machine learning on your local machine,
you might find it a bit tricky to set up a proper environment without burning your pocket on GPUs. Anyone with a Mac with Apple Silicon (M1, M2, M3) has access
to reasonably nifty hardware, at least for some basic machine learning tasks.

Here's a guide on setting up a machine learning environment on Apple Silicon, specifically for Python and TensorFlow.
In an attempt to keep things simple, this guide relies only on the standard Python distribution rather than Anaconda, Miniconda or other Python distributions.

## Install Homebrew

Homebrew is a package manager for macOS. It's the easiest way to install and manage software on your Mac. You can install Homebrew by running the following command in your terminal:

```bash

    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

```

(most likely you already have it installed)

## Install Python

You can install Python using Homebrew. Run the following command in your terminal:

```bash

    brew install python
```

This will install a recent version of Python, but to use Apple Silicon at this moment in time, you will need to install an older version of Python. You can do this by running the following command:

```bash

    brew install python@3.9
```

## Create a virtual environment

You want to separate your machine learning environment from your system Python. You can do this by creating a virtual environment. Run the following command in your terminal:

```bash

    python3.9 -m venv ml-env
```

This will create a virtual environment in a folder called `ml-env`. You can activate the virtual environment by running the following command:

```bash

    source ml-env/bin/activate
```

## Install TensorFlow

You can install TensorFlow using pip. Run the following command in your terminal:

```bash

    pip install tensorflow
```

This will install the latest version of TensorFlow available for Python 3.9.

## Install TensorFlow-Metal

Apple has released a Metal backend for TensorFlow, which allows you to use the GPU on Apple Silicon. You can install TensorFlow-Metal using pip. Run the following command in your terminal:

```bash

    pip install tensorflow-metal
```

## Test your installation

You can test your installation by running the following Python code in your terminal:

```python

    import tensorflow as tf
    print(tf.__version__)
```

This will print the version of TensorFlow you have installed. If you see a version number, then congratulations! You have successfully set up a machine learning environment on Apple Silicon.

:::{warning}
You will see the following message in Tensorflow, this is normal as Apple Silicon uses a UMA and not NUMA architecture. You can ignore this:

    `tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:306] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.`
:::

## Install Jupyter Lab

You can install Jupyter Lab using pip. Run the following command in your terminal:

```bash

    pip install jupyterlab
```

You can start Jupyter Lab by running the following command in your terminal:

```bash

    jupyter lab
```

This will start Jupyter Lab in your default web browser. You can now create a new notebook and start doing some machine learning on your local machine.

That's it! You now have a machine learning environment set up on Apple Silicon. You can start doing some basic machine learning tasks on your local machine
without burning your pocket on GPUs.
 
Happy coding! 