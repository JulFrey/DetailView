# DetailView
Tree species classification for [3D4EcoTec](https://github.com/stefp/Tr3D_species)
by Zoe Schindler and Julian Frey (<a href ="https://www.iww.uni-freiburg.de/">Chair of Forest Growth and Dendroecology, University of Freiburg</a>)

The project is based on pytorch. Point clouds of single trees are augmented and projected to grids 
(4 side views, 1 top view, 1 bottom view, 1 detail view 1 - 1.5 m height). Classification is done 
with three densenet201 instances, one for the side views, one for the top and bottom views and the 
last one for the detail view. These classifications are merged and the tree height is also added. 
After that, two more linear layers with a relu layer in between are used as classifiers. 

The final predictions are made by adding the probabilities of 50 augmentations of each test tree 
and selecting the most probable class.

## Code Overview

### Main Code

1. <a href ="https://github.com/JulFrey/DetailView/blob/main/downsampling.py">downsample.py</a>: downsample data & remove point clouds with < 100 points
2. <a href = "https://github.com/JulFrey/DetailView/blob/main/balancing.py">balancing.py</a>: balancing the data set
3. <a href = "https://github.com/JulFrey/DetailView/blob/main/parallel_densenet.py">parallel_densenet.py</a>: setting up neural network
4. <a href = "https://github.com/JulFrey/DetailView/blob/main/training.py">training.py</a>: training, validating, testing

### Utilities

* <a href = "https://github.com/JulFrey/DetailView/blob/main/augmentation.py">augmentation.py</a>: augment point clouds
* <a href = "https://github.com/JulFrey/DetailView/blob/main/read_las.py">read_las.py.py</a>: read las files to numpy arrays
* <a href = "https://github.com/JulFrey/DetailView/blob/main/sideview.py">sideview.py</a>: create sideviews of point clouds

