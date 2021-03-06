#!/usr/bin/env python
"""
Plotting Utilities for TransportMapOptimizer

"""

# Import Modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Code Implementation


# Kernel Density Plot (given samples + coordinate)
def kde_plot(samples, index1, index2=None, **kwargs):
    if index2 is None:
        return sns.distplot(samples[:,index1], **kwargs)
    else:
        return sns.jointplot(samples[:,index1], samples[:,index2], **kwargs)



# Code To Execute
if __name__ == '__main__':
    print "_plot_utils.py"





#EOF
