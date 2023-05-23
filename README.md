# DetailView
Tree species classification for [3D4EcoTec](https://github.com/stefp/Tr3D_species)
by Zoe Schindler and Julian Frey (Chair of Forest Growth and Dendroecology, University of Freiburg)

The project is based on pytorch. Point clouds of single trees are augmented and projected to grids 
(4 side views, 1 top view, 1 bottom view, 1 detail view 1 - 1.5 m height). Classification is done 
with three densenet201 instances, one for the side views, one for the top and bottom views and the 
last one for the detail view. These classifications are merged and the tree height is also added. 
After that, two more linear layers with a relu layer in between are used as classifiers. 

The final predictions are made by adding the probabilities of 50 augmentations of each test tree 
and selecting the most probable class.

## Roadmap
1. downsample data & remove point clouds with < 100 points --> output: las files ✓
2. balancing the data set --> output: weights ✓
3. read point clouds, augment point clouds, create sideviews --> output: tensors ✓
4. setting up neural network --> output: pytorch model structure ✓
5. setting up data pipeline --> output: pytorch data pipeline ✓
6. training & validating neural network --> output: trained pytorch model ✓
