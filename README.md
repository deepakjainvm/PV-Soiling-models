# Prediction of Soiling on PV Modules

## Overview
The original Kimber and HSU models assume that the soiling ratio returns to 1 (completely clean) after a rain event. However, in reality, this is not always the case. This repository contains modified Kimber and modified HSU functions, offering improved versions of the original models. These modifications consider that the cleaning effect depends on both the soiling ratio and rainfall.

### Models Included
- Three modified and improved versions of the original Kimber model
- Two modified and improved versions of the original HSU model

### Features
- Improved accuracy in predicting losses due to soiling on PV panels
- Consideration of realistic soiling recovery patterns after rain events


### Required Packages
- Python >= 3.7
- NumPy
- Pandas
- SciPy

### Acknowledgements
Our model is improved and modified based on [Kimber model](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.soiling.kimber.html) and [hsu model](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.soiling.hsu.html) from the PVLIB Python library.

