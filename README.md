# Task Embedding Temporal Convolution Networks for Transfer Learning Problems in Renewable Power Time-Series Forecast

This README is currently a place holder. 
After the double blind review at ECML, this repository will be published through github.
We will include the trained models, plots, tables and so forth.
For the current submission we limite the repository to the source code, due to the restricted size.


## Project Structure:

- doc: contains various visalizations
- results: where all results will be stored including csvs, it also includes the trained models
- dies: contains the models such as the task embedding mlp and the task-TCN
- rep: is used for preprocessing and as a framework for transfer learning in renewable power forecasts
- confer/experiments: contains all the scripts to execute the experiment:
    1. `confer/experiments/create_splits.py` creates a cross validation splits of source and target data.
    2. `confer/experiments/source.py` trains on those splits a source model. Can be either mlp or cnn.
    3. `confer/experiments/mtl.py`  executes the mtl experiment   
    4. `confer/experiments/zero_shot.py`  executes the zero shot learning experiment
    5. `confer/experiments/target.py` executes the inductive TL experiment



## Setup Python environment:

```
projectpath$ sudo apt-get install python3-pip
projectpath$ pip3 install virtualenv
projectpath$ virtualenv tcn
projectpath$ source tcn/bin/activate
projectpath$ pip3 install -r requirements.txt
```