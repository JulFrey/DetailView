# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 12:26:11 2023

@author: Julian
"""

# import packages
import os
import torch
import datetime
import torchmetrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms

# import own scripts
import parallel_densenet as net

# set parameters
n_class = 33    # number of classes
n_view  = 7     # number of views
n_batch = 2**1  # batch size
n_train = 2**13 # training dataset size
res = 512       # image ressolution
n_sides = n_view - 3      # number of sideviews

# set paths
path_csv_lookup = r"C:\TLS\down\lookup.csv"
path_csv_train  = r"C:\TLS\down\train_labels.csv"
path_csv_vali   = r"C:\TLS\down\vali_labels.csv"
path_csv_test   = r"C:\TLS\down\test_labels.csv"
path_las        = r"C:\TLS\down"

# get mean & sd of height from training data
train_metadata = pd.read_csv(path_csv_train)
train_height_mean = np.mean(train_metadata["tree_H"])
train_height_sd = np.std(train_metadata["tree_H"])

# add more threads
os.environ["OMP_NUM_THREADS"] = "40" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "40" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "40" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "40" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "40" # export NUMEXPR_NUM_THREADS=6


#%% prepare simple view

# setting up image augmentation
img_trans = transforms.Compose([
    transforms.RandomVerticalFlip(0.5)])

# prepare data
dataset = net.TrainDataset_AllChannels(path_csv_train, path_las, img_trans = img_trans, height_noise = 0.01, res = res, n_sides = n_sides, height_mean = train_height_mean, height_sd = train_height_sd)

# define a sampler
sampler = torch.utils.data.sampler.WeightedRandomSampler(dataset.weights(), n_train, replacement = True)

# create data loader
dataloader = torch.utils.data.DataLoader(dataset, batch_size = n_batch, sampler = sampler, pin_memory = True)

# load the model
# model = net.ParallelDenseNet(n_classes = n_class, n_views = n_view)
model = net.SimpleView(n_classes = n_class, n_views = n_view)

# get the device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu")

# give to device
model.to(device)

# define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss() #(label_smoothing = 0.2)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', patience = 3, verbose = True, factor = 0.5)

# # get test prediction
# inputs, heights, labels = next(iter(dataloader))
# inputs, heights, labels = inputs.to(device), heights.to(device), labels.to(device)
# preds = model(inputs, heights)

#%% training loop

# prepare validation data for checking
vali_dataset = net.TrainDataset_AllChannels(path_csv_vali, path_las, pc_rotate = False, height_noise = 0, res = res, n_sides = n_sides, height_mean = train_height_mean, height_sd = train_height_sd)
vali_dataloader = torch.utils.data.DataLoader(vali_dataset, batch_size = n_batch, shuffle = False, pin_memory = True)

# prepare training
num_epochs = 100
best_v_loss = 1000
last_improvement = 0
timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M')

# save loss
ls_loss = []
ls_v_loss = []

# loop through epochs
for epoch in range(num_epochs):
    running_loss = 0.0
    running_epoch_loss = 0.0
    
    # loop through whole dataset?
    for i, data in enumerate(dataloader, 0): 
        
        # load data
        inputs, heights, labels = data
        inputs, heights, labels = inputs.to(device), heights.to(device), labels.to(device)
        
        # forward + backward + optimize
        outputs = model(inputs, heights)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # update epoch loss
        running_epoch_loss += loss.item()
        
        # print loss every 100 batches
        running_loss += loss.item()
        if i % 400 == 399:
            
            # optimize
            optimizer.step()
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # print current loss
            print('[epoch: %d, batch: %d, dataset: %.2f%%] loss: %.4f' %
                  (epoch + 1, i + 1, i / len(dataloader) * 100, running_loss / 400))
            running_loss = 0.0
        
        # clear memory
        inputs = heights = labels = 0
        del inputs, heights, labels
        torch.cuda.empty_cache()
        
    # clear memory
    del running_loss
    torch.cuda.empty_cache()
        
    # validation loss
    running_v_loss = 0
    accuracy = torchmetrics.Accuracy(task = "multiclass", num_classes = int(n_class)).to(device)
    model.eval()
    with torch.no_grad():
        for j, v_data in enumerate(vali_dataloader, 0):
            v_inputs, v_heights, v_labels = v_data
            v_inputs, v_heights, v_labels = v_inputs.to(device), v_heights.to(device), v_labels.to(device)
            v_outputs = model(v_inputs, v_heights)
            v_loss = criterion(v_outputs, v_labels)
            running_v_loss += v_loss.item()
            accuracy.update(v_outputs, v_labels)
            del v_inputs, v_heights, v_labels
            torch.cuda.empty_cache()
        avg_v_loss = running_v_loss / len(vali_dataloader)
        final_accuracy = accuracy.compute()

    model.train()
    print('[epoch: %d, validation] loss: %.4f, accuracy: %.4f' %
          (epoch + 1, avg_v_loss, final_accuracy))
    
    # append lists tracking loss
    ls_loss.append(running_epoch_loss / len(dataloader))
    ls_v_loss.append(avg_v_loss)    
    
    # adjust learning rate
    scheduler.step(avg_v_loss)
    
    # save best model
    if avg_v_loss < best_v_loss:
        best_v_loss = avg_v_loss
        model_path = "model_{}_{}".format(timestamp, epoch + 1)
        torch.save(model.state_dict(), model_path)
        last_improvement = 0
    else:
        last_improvement += 1
    
    # check how long last improvement was ago
    if last_improvement > 6:
        break

# finish training
torch.cuda.empty_cache()
print('\nFinished training\n')

# plot loss
plt.plot(range(1, len(ls_loss) + 1), ls_loss, color = "cornflowerblue", label = "Training loss")
plt.plot(range(1, len(ls_v_loss) + 1), ls_v_loss, color = "salmon", label = "Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Create a DataFrame from the dictionary
df = pd.DataFrame({'Loss': ls_loss, 'Validation Loss': ls_v_loss})

# Save the DataFrame to a CSV file
df.to_csv('loss_' + timestamp + '.csv', index=False)

#%% validating cnn

# load best model
# model = net.ParallelDenseNet(n_classes = n_class, n_views = n_view)
model = net.SimpleView(n_classes = n_class, n_views = n_view)
# To-Do load best model dynamically 
model.load_state_dict(torch.load("model_202305171452_60"))

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

# prepare data for validation
vali_dataset = net.TrainDataset_AllChannels(path_csv_vali, path_las, pc_rotate = False, height_noise = 0, res = res, n_sides = n_sides, height_mean = train_height_mean, height_sd = train_height_sd)
vali_dataloader = torch.utils.data.DataLoader(vali_dataset, batch_size = int(n_batch / 2), shuffle = False, pin_memory = True)

#%%

# create metrics
accuracy = torchmetrics.Accuracy(task = "multiclass", num_classes = int(n_class)).to(device)
f1 = torchmetrics.F1Score(task = "multiclass", num_classes = int(n_class)).to(device)

# iterate over validation dataloader in batches
for i, v_data in enumerate(vali_dataloader, 0):
    v_inputs, v_heights, v_labels = v_data
    v_inputs, v_heights, v_labels = v_inputs.to(device), v_heights.to(device), v_labels.to(device)
    
    # get predictions
    v_preds = model(v_inputs, v_heights)

    # calculate metrics for the batch
    accuracy.update(v_preds, v_labels)
    f1.update(v_preds, v_labels)

# get the final metrics
final_accuracy = accuracy.compute()
final_f1 = f1.compute()

# print final metrics
print('accuracy: %.3f' % final_accuracy)
print('f1: %.3f' % final_f1)

