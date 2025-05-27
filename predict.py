# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 09:21:58 2023

@author: Julian
"""

# import packages
import os
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
import laspy
import argparse
from datetime import datetime

# import own scripts
import parallel_densenet as net

#%% set parameters

# parse command line arguments
parser = argparse.ArgumentParser(description="Tree species prediction")
parser.add_argument('--prediction_data', type=str, default=r"/input/circle_3_segmented.las",
                    help='Path to LAS file or CSV for prediction')
parser.add_argument('--path_las', type=str, default="",
                    help='Path to LAS files (used if prediction_data is a CSV)')
parser.add_argument('--model_path', type=str, default="./model_ft_202412171652_3",
                    help='Path to model weights')
parser.add_argument('--tree_id_col', type=str, default='TreeID',
                    help='Column name for tree IDs in LAS/CSV')
parser.add_argument('--n_aug', type=str, default=10,
                    help='Number of augmentations per tree (default: 10)')

args = parser.parse_args()

prediction_data = args.prediction_data
path_las = args.path_las
model_path = args.model_path
tree_id_col = args.tree_id_col
n_aug = args.n_aug # number of augmentations per tree

if os.path.splitext(prediction_data)[1].lower() in ['.las', '.laz']:
    prediction_data = laspy.read(prediction_data)

# set variables
# prediction_data = laspy.read(r"T:\Ecosense\2024-10-15 ecosense.RiSCAN\EXPORTS\Export Point Clouds\segmented_circles\circle_1_segmented.las") # r".\test_labels_es.csv"
# path_las        = r"" # only needed if prediction_data is a csv file, otherwise set to empty string
# model_path      = r".\model_ft_202412171652_3" # path to the model weights if file does not exist it will be downloaded from https://freidata.uni-freiburg.de/records/xw42t-6mt03/files/model_202305171452_60?download=1
outfile         = "/output/predictions_" + datetime.today().strftime('%Y-%m-%d_%H-%M-%S') + "_.csv" # path to the output file
outfile_probs   = "/output/predictions_probs" + datetime.today().strftime('%Y-%m-%d_%H-%M-%S') + "_.csv" # path to the output file with probabilities.
# tree_id_col     = 'TreeID' # column name for the tree id in the las file (only used if prediction_data is a las file).
path_csv_train  = 'default_vals' # r".\train_labels.csv"

# lookup file for species names (do not change)
path_csv_lookup = "./lookup.csv"

# set parameters (adapt if you run into memory issuses)
n_class = 33    # number of classes
n_view  = 7     # number of views
n_batch = 2**1  # batch size
n_train = 2**13 # training dataset size
res = 256       # image ressolution
n_sides = n_view - 3      # number of sideviews


# check if model exists otherwise load the best model from https://freidata.uni-freiburg.de/records/xw42t-6mt03/files/model_202305171452_60?download=1
if not os.path.exists(model_path):
    print("Model" + model_path + " not found, downloading basemodel from freidata.uni-freiburg.de...")
    # download the file
    import requests
    response = requests.get("https://freidata.uni-freiburg.de/records/xw42t-6mt03/files/model_202305171452_60?download=1")
    model_path = "/model_202305171452_60"
    with open(model_path, 'wb') as f:
        f.write(response.content)


# get mean & sd of height from training data if given otherwise take the one from the original paper
if os.path.exists(path_csv_train):
    train_metadata = pd.read_csv(path_csv_train)
    train_height_mean = np.mean(train_metadata["tree_H"])
    train_height_sd = np.std(train_metadata["tree_H"])
else:
    train_height_mean = 15.2046
    train_height_sd = 9.5494

#%% set environment variables

# add more threads
os.environ["OMP_NUM_THREADS"] = "10" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "10" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "10" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "10" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "10" # export NUMEXPR_NUM_THREADS=6


#%% make predictions

# load best model
# model = net.ParallelDenseNet(n_classes = n_class, n_views = n_view)
model = net.SimpleView(n_classes = n_class, n_views = n_view)
model.load_state_dict(torch.load(model_path))

# get the device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu")

# give to device
model.to(device)

# turn on evaluation mode
model.eval()

# setting up image augmentation
img_trans = transforms.Compose([
    transforms.RandomVerticalFlip(0.5)])

# prepare data for testing
test_dataset = net.TrainDataset_AllChannels(prediction_data, path_las,  img_trans = img_trans, pc_rotate = True, height_noise = 0.01, test = True, res = res, n_sides = n_sides, height_mean = train_height_mean, height_sd = train_height_sd, tree_id_col = tree_id_col)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = int(n_batch/2), shuffle = False, pin_memory = True)

# create dictionary for the accumulated probabilities for each data point
all_paths = test_dataset.trees_frame.iloc[:,0]
data_probs = {path: [] for path in all_paths}

# iterate over the whole dataset 50 times
for epoch in range(n_aug):
    print("epoch: %d" % (epoch + 1))
    
    # iterate over validation dataloader in batches
    for i, t_data in enumerate(test_dataloader, 0):
        
        # load the batch
        t_inputs, t_heights, t_paths = t_data
        t_inputs, t_heights = t_inputs.to(device), t_heights.to(device)
        
        # get predictions
        t_preds = model(t_inputs, t_heights)
        t_probs = torch.nn.functional.softmax(t_preds, dim = 1)
        t_probs = t_probs.cpu().detach().numpy()
        
        # accumulate probabilities for each data point
        for i, path in enumerate(t_paths):
            if not any(data_probs[path]):
                data_probs[path] = t_probs[i,:]
            else:
                data_probs[path] += t_probs[i,:]

# get class id with maximum accumulated probabilities
max_prob_class = {}
for key, array in data_probs.items():
    max_prob_class[key] = np.argmax(array)

# create dataframe
df = pd.DataFrame({
    "filename": max_prob_class.keys(),
    "species_id": max_prob_class.values()})

# add species name
lookup = pd.read_csv(path_csv_lookup)
joined = pd.merge(df, lookup, on = 'species_id')
# joined = joined.drop("species_id", axis = 1)

# create a df with the probabilities
data_probs_df = pd.DataFrame.from_dict(data_probs, orient='index').reset_index()
col_labels = lookup['species']
data_probs_df.columns = pd.concat([pd.Series(["File"]), col_labels])

# save data frame
joined.to_csv(outfile, index = False)
data_probs_df.to_csv(outfile_probs, index = False)
