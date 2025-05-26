# -*- coding: utf-8 -*-
"""
Created on Mon Mai 26 10:25:32 2025
Script to load a single las file, prepare the data, and save it as a CSV file for further processing.
The csv file contains the file name, a blank species_id column, and the tree height.
All single trees based on their TreeID are saved as separate las files to the target directory.

@author: Julian Frey
"""

import os
import pandas as pd
import laspy as las
import numpy as np


def prepare_las_file(input_las, output_csv, output_dir, skip_0=True, instance_column = 'TreeID'):
    """
    Prepare a single las file for further processing.

    Parameters:
    input_las (str): Path to the input las file.
    output_csv (str): Path to the output CSV file.
    output_dir (str): Directory where the processed las files will be saved.
    skip_0 (bool): If True, skip trees with TreeID 0.
    instance_column (str): The column name in the las file that contains the tree IDs (default is 'TreeID').

    Returns:
    None
    """
    # ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # chek if the input file exists
    if not os.path.exists(input_las):
        raise FileNotFoundError(f"The input file {input_las} does not exist.")

    # read the las file
    las_data = las.read(input_las)

    # grep the TreeIDs from the las file
    ids = np.unique(las_data[instance_column])

    # if skip_0 is True, remove TreeID 0 from the list of ids
    if skip_0 is True:
        ids = ids[ids != 0]

    # create a DataFrame with the required columns filename,species_id,tree_H
    df = pd.DataFrame({
        'filename': [np.nan],  # blank filename
        'species_id': [np.nan],  # blank species_id
        'tree_H': [np.nan]   # tree height based on Z_range
    })

    # save each tree as a separate las file based on TreeID
    for tree_id in ids:
        tree_points = las_data[las_data[instance_column] == tree_id]
        if len(tree_points) > 0:
            # calculate tree height as the range of Z values
            tree_height = np.max(tree_points.z) - np.min(tree_points.z)
            output_file = os.path.join(output_dir, f'tree_{int(tree_id)}.las')
            tree_points.write(output_file)
            # append the output file and tree height to the DataFrame
            df = df._append({
                'filename': output_file,
                'species_id': -999,  # still blank
                'tree_H': tree_height
            }, ignore_index=True)

    # save the DataFrame to a CSV file
    df = df[df['filename'].notna()]
    df.to_csv(output_csv, index=False)
    return df


# #%% test the function
# if __name__ == "__main__":
#     input_las = r"T:\Ecosense\2024-10-15 ecosense.RiSCAN\EXPORTS\Export Point Clouds\segmented_circles\citercle_1_segmented.las"  # replace with your input las file path
#     output_csv = r".\test_labels_es.csv"  # replace with your desired output csv file path
#     output_dir = r".\test_las"  # replace with your desired output directory for las files
#
#     prepare_las_file(input_las, output_csv, output_dir)