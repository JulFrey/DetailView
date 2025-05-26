# DetailView
Tree species classification for [3D4EcoTec](https://github.com/stefp/Tr3D_species)
by [Zoe Schindler](https://freidok.uni-freiburg.de/pers/307357) and [Julian Frey](https://freidok.uni-freiburg.de/pers/24353) (<a href ="https://www.iww.uni-freiburg.de/">Chair of Forest Growth and Dendroecology, University of Freiburg</a>)



[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14204431.svg)](https://doi.org/10.5281/zenodo.14204431)



The project is based on pytorch. Point clouds of single trees are augmented and projected to grids 
(4 side views, 1 top view, 1 bottom view, 1 detail view 1 - 1.5 m height).  
The dataset is available on [<a href = "https://zenodo.org/records/13255198?utm_source=substack&utm_medium=email"> Zenodo</a>](https://zenodo.org/records/13255198). 

The training weights are available on [<a href = "https://doi.org/10.60493/xw42t-6mt03"> FreiData</a>](https://doi.org/10.60493/xw42t-6mt03).

Classification is done with three densenet201 instances, one for the side views, one for the top and bottom views and the 
last one for the detail view. These classifications are merged and the tree height is also added. 
After that, two more linear layers with a relu layer in between are used as classifiers. 

The final predictions are made by adding the probabilities of 50 augmentations of each test tree 
and selecting the most probable class.

Any scientific publication using the data should cite the following paper:

Puliti, S., Lines, E., Müllerová, J., Frey, J., Schindler, Z., Straker, A., Allen, M.J., Winiwarter, L., Rehush, N., Hristova, H., Murray, B., Calders, K., Terryn, L., Coops, N., Höfle, B., Krůček, M., Krokm, G., Král, K., Luck, L., Levick, S.R., Missarov, A., Mokroš, M., Owen, H., Stereńczak, K., Pitkänen, T.P., Puletti, N., Saarinen, N., Hopkinson, C., Torresan, C., Tomelleri, E., Weiser, H., Junttila, S., and Astrup, R. (2024) Benchmarking tree species classification from proximally-sensed laser scanning data: introducing the FOR-species20K dataset. ArXiv; [available here](https://www.arxiv.org/abs/2408.06507)

The development of the model was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – Project FR 4404/1-1, “Making the direct link between light regime and forest biodiversity – a 3D spatially explicit modelling approach based on TLS, CNNs and ray tracing”.

## Code Overview

### Main Code

1. <a href ="https://github.com/JulFrey/DetailView/blob/main/downsampling.py">downsample.py</a>: downsample data & remove point clouds with < 100 points
2. <a href = "https://github.com/JulFrey/DetailView/blob/main/balancing.py">balancing.py</a>: balancing the data set
3. <a href = "https://github.com/JulFrey/DetailView/blob/main/parallel_densenet.py">parallel_densenet.py</a>: setting up neural network & data loader
4. <a href = "https://github.com/JulFrey/DetailView/blob/main/training.py">training.py</a>: training & validating
5. <a href = "https://github.com/JulFrey/DetailView/blob/main/predict.py">predict.py</a>: predicting to test data
   The model weights trained in the FOR-species20K paper can be downloaded [here](https://freidata.uni-freiburg.de/records/xw42t-6mt03).

### Utilities

* <a href = "https://github.com/JulFrey/DetailView/blob/main/augmentation.py">augmentation.py</a>: augment point clouds
* <a href = "https://github.com/JulFrey/DetailView/blob/main/read_las.py">read_las.py</a>: read las files to numpy arrays
* <a href = "https://github.com/JulFrey/DetailView/blob/main/sideview.py">sideview.py</a>: create sideviews of point clouds

### Results

* <a href = "https://github.com/JulFrey/DetailView/blob/main/loss_202305171452.csv">loss_202305171452.csv</a>: history of the loss & validation loss
* <a href = "https://github.com/JulFrey/DetailView/blob/main/test_predictions.csv">test_predictions.csv</a>: predicted labels for the test data

### How to use the model to perform predictions

Setup the python environment with the required packages.
```bash 
conda create -n detailview python=3.12
conda activate detailview
pip3 install numpy pandas scikit-learn laspy matplotlib requests
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
You will need a single las/laz file with a coloumn with single tree labels (e.g. TreeID, can be specified with the argument tree_id_col) or one file for every tree and a csv file with the columns filename, species_id, tree_H. filename gives the ralative path to the path_las argument, species_id can be blank or -999, tree_H is the height of the tree in meters.


