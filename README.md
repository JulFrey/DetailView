# DetailView
Tree species classification for 3D4EcoTec

## Roadmap
1. downsample data & remove point clouds with < 100 points --> output: las files
    - downsample to 1 point/cm³
2. balancing the data set --> output: indices
    - based on sensor tree species, tree height, sensor type
3. read point clouds, augment point clouds, create sideviews --> output: tensors
    - rotation, translation, scaling, sampling
4. setting up neural network --> output: pytorch model structure
    - combining two DenseNet-201 networks, one for the whole tree, one for details
5. setting up data pipeline --> output: pytorch data pipeline
    - combining point cloud augmentation, sideview creation, image augmentation
6. training & validating neural network --> output: trained pytorch model
    - getting confusion matrix & accuracy
