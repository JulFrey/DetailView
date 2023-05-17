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
import augmentation as au
import sideview as sv
import read_las as rl
import parallel_densenet as net

# set parameters
n_class = 33    # number of classes
n_view  = 7     # number of views
n_batch = 2**3  # batch size
n_train = 2**13 # training dataset size

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

#%% setup new dataset class

# create dataset class to load the data from csv and las files
class TrainDataset_AllChannels():
    
    """Tree species dataset."""
    
    # initialization
    def __init__(self, csv_file, root_dir, img_trans = None, pc_rotate = True,
                 height_noise = 0.01, height_mean = train_height_mean,
                 height_sd = train_height_sd, test = False):
        
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations with the collumns
                0: filenme, 1: label_id, 2: tree height.
            root_dir (string): Directory with all the las files.
            img_trans (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        # set attributes
        self.trees_frame  = pd.read_csv(csv_file)
        self.root_dir     = root_dir
        self.img_trans    = img_trans
        self.pc_rotate    = pc_rotate
        self.height_noise = height_noise
        self.height_mean  = height_mean
        self.height_sd    = height_sd
        self.test         = test
    
    # length
    def __len__(self):
        return len(self.trees_frame)
    
    # indexing
    def __getitem__(self, idx):
        
        # convert indices to list
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # get full las path
        las_name = os.path.join(
            self.root_dir,
            *self.trees_frame.iloc[idx, 0].split('/'))
        
        # get side views
        if self.pc_rotate:
            image = sv.points_to_images(au.augment(las_name), res_im = 128)
        else:
            image = sv.points_to_images(rl.read_las(las_name), res_im = 128)
        image = torch.from_numpy(image)
        
        # augment images (all channels at once)
        if self.img_trans:
            image = self.img_trans(image)
        
        # add dimension
        image = image.unsqueeze(1)
        
        # get height
        height = torch.tensor(self.trees_frame.iloc[idx, 2], dtype = torch.float32)
        
        # augment height
        if self.height_noise > 0:
            height += np.random.normal(0, self.height_noise)
        
        # scale height using training mean & sd
        height = (height - self.height_mean) / self.height_sd
        
        # return images with filenames
        if self.test:
            las_path = self.trees_frame.iloc[idx, 0]
            return image, height, las_path
        
        # return images with labels
        label = torch.tensor(self.trees_frame.iloc[idx, 1], dtype = torch.int64)
        return image, height, label
    
    # training weights
    def weights(self):
        return torch.tensor(self.trees_frame["weight"].values)
        
#%% test dataset & dataloader

# # setting up image augmentation
# img_trans = transforms.Compose([
#     transforms.RandomVerticalFlip(0.5),
#     transforms.RandomAffine(
#         degrees = 0, translate = (0.1, 0.1), scale = (0.9, 1.1))])

# # create dataset object
# dataset = TrainDataset_AllChannels(path_csv_train, path_las) # without
# dataset = TrainDataset_AllChannels(path_csv_train, path_las, img_trans = img_trans) # with

# # # show image
# # plt.imshow(dataset[0][0][0,0,:,:], interpolation = 'nearest')
# # plt.show()

# # define a sampler
# sampler = torch.utils.data.sampler.WeightedRandomSampler(dataset.weights(), n_train, replacement = True)

# # create data loader
# batch_size = 2**0
# dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, sampler = sampler, pin_memory = True)

# # # test output of iterator
# # image, height, label = next(iter(dataloader))
# # print(image.shape); print(height.shape); print(label.shape)

# # # show image
# # plt.imshow(image[0,2,0,:,:], interpolation = 'nearest')
# # plt.show()

#%% prepare simple view

# setting up image augmentation
img_trans = transforms.Compose([
    transforms.RandomVerticalFlip(0.5)])

# prepare data
dataset = TrainDataset_AllChannels(path_csv_train, path_las, img_trans = img_trans, height_noise = 0.01)

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
vali_dataset = TrainDataset_AllChannels(path_csv_vali, path_las, pc_rotate = False, height_noise = 0)
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
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = model(inputs, heights)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # update epoch loss
        running_epoch_loss += loss.item()
        
        # print loss every 100 batches
        running_loss += loss.item()
        if i % 100 == 99:
            print('[epoch: %d, batch: %d, dataset: %.2f%%] loss: %.4f' %
                  (epoch + 1, i + 1, i / len(dataloader) * 100, running_loss / 100))
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
    for j, v_data in enumerate(vali_dataloader, 0):
        v_inputs, v_heights, v_labels = v_data
        v_inputs, v_heights, v_labels = v_inputs.to(device), v_heights.to(device), v_labels.to(device)
        v_outputs = model(v_inputs, v_heights)
        v_loss = criterion(v_outputs, v_labels)
        running_v_loss += v_loss.item()
        accuracy.update(v_outputs, v_labels)
        v_inputs = v_heights = v_labels = 0
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

#%% validate cnn

# load best model
# model = net.ParallelDenseNet(n_classes = n_class, n_views = n_view)
model = net.SimpleView(n_classes = n_class, n_views = n_view)
model.load_state_dict(torch.load("model_"))

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
vali_dataset = TrainDataset_AllChannels(path_csv_vali, path_las, pc_rotate = False, height_noise = 0)
vali_dataloader = torch.utils.data.DataLoader(vali_dataset, batch_size = n_batch, shuffle = False, pin_memory = True)

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

#%% make predictions

# load best model
# model = net.ParallelDenseNet(n_classes = n_class, n_views = n_view)
model = net.SimpleView(n_classes = n_class, n_views = n_view)
model.load_state_dict(torch.load("model_"))

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
test_dataset = TrainDataset_AllChannels(path_csv_test, path_las,  img_trans = img_trans, pc_rotate = True, height_noise = 0.01, test = True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False, pin_memory = True)

# create dictionary for the accumulated probabilities for each data point
data_probs = {i: [] for i in range(len(test_dataset))}

# iterate over the whole dataset 50 times
for epoch in range(50):
    
    # iterate over validation dataloader in batches
    for i, t_data in enumerate(test_dataloader, 0):
        
        # load the batch
        t_inputs, t_heights, t_paths = t_data
        t_inputs, t_heights = t_inputs.to(device), t_heights.to(device)
        
        # get predictions
        t_preds = model(t_inputs, t_heights)
        t_probs = torch.nn.functional.softmax(t_preds, dim = 1)
        
        # accumulate probabilities for each data point
        if len(data_probs[i]) == 0:
            data_probs[i] = t_probs
        else:
            data_probs[i] += t_probs

# get class id with maximum accumulated probabilities
max_prob_class = {i: probs.index(max(probs)) for i, probs in data_probs.items()}

# create dataframe
df = pd.DataFrame({
    "filename": test_dataset.trees_frame.iloc[:,0],
    "species_id": max_prob_class.values()})

# load lookup table
lookup = pd.read_csv(path_csv_lookup)

# join tables
joined = pd.merge(df, lookup, on = 'species_id')

# save data frame
joined.to_csv("test_predictions.csv", index = False)
