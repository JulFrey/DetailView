# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 10:14:32 2023

@author: Zoe
"""

# import packages
import pdal
import numpy as np

#%%

def read_las(path_las):
    
    """
    
    Parameters
    ----------
    path_las : str
        Path to a las file.

    Returns
    -------
    data : np.array
        XYZ point coordinates in np.array.
    """
    
    # read in data
    pipeline = pdal.Pipeline([pdal.Reader(path_las, type = "readers.las")])
    pipeline.execute()
    
    # convert to numpy array
    data = pipeline.arrays[0]
    data = np.stack((data['X'], data['Y'], data['Z']), axis = 1)
    
    # return numpy array
    return data
