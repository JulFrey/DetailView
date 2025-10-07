# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:43:42 2023

@author: Julian
"""

# import packages
import torch
import torchvision
import torch.nn as nn
import pandas as pd
import os
import numpy as np
import laspy

# import own scripts
import augmentation as au
import sideview as sv
import sideview_torch as svt
import read_las as rl

# https://github.com/isaaccorley/simpleview-pytorch/blob/main/simpleview_pytorch/simpleview.py
class SimpleView(nn.Module):
    def __init__(self, n_classes: int, n_views: int):
        super().__init__()
        
        # load model for sides views
        sides = torchvision.models.densenet201(weights = "DenseNet201_Weights.DEFAULT")
        
        # change first layer to greyscale
        sides.features[0].in_channels = 1
        sides.features[0].weight = torch.nn.Parameter(sides.features[0].weight.sum(dim = 1, keepdim = True))
        
        # remove effect of classifier
        z_dim = sides.classifier.in_features
        sides.classifier = nn.Identity()
        
        # load model for tops views
        tops = torchvision.models.densenet201(weights = "DenseNet201_Weights.DEFAULT")
        
        # change first layer to greyscale
        tops.features[0].in_channels = 1
        tops.features[0].weight = torch.nn.Parameter(tops.features[0].weight.sum(dim = 1, keepdim = True))
        
        # remove effect of classifier
        tops.classifier = nn.Identity()
        
        # load model for details
        details = torchvision.models.densenet201(weights = "DenseNet201_Weights.DEFAULT")
        
        # change first layer to greyscale
        details.features[0].in_channels = 1
        details.features[0].weight = torch.nn.Parameter(details.features[0].weight.sum(dim = 1, keepdim = True))
        
        # remove effect of classifier
        details.classifier = nn.Identity()
        
        # add new classifier & float pathway
        self.sides_pathway = sides
        self.tops_pathway = tops
        self.details_pathway = details
        self.height_pathway = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, z_dim),
            nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(in_features = z_dim * (n_views + 1), out_features = 256),
            nn.ReLU(),
            nn.Linear(in_features = 256, out_features = n_classes))

    def forward(self, inputs: torch.Tensor, heights: torch.Tensor) -> torch.Tensor:
        
        # prepare data
        b, v, c, h, w = inputs.shape
        sides = inputs[:,1:-2,:,:,:].reshape(b * (v - 3), c, h, w)
        tops = inputs[:,[0,-2],:,:,:].reshape(b * 2, c, h, w)
        details = inputs[:,-1,:,:,:].reshape(b * 1, c, h, w)
        del inputs
        
        # process sides views
        sides = self.sides_pathway(sides)
        sides = sides.reshape(b, (v - 3), -1).reshape(b, -1)
        
        # process tops views
        tops = self.tops_pathway(tops)
        tops = tops.reshape(b, 2, -1).reshape(b, -1)
        
        # process details
        details = self.details_pathway(details)
        details = details.reshape(b, 1, -1).reshape(b, -1)
        
        # process height
        heights = self.height_pathway(heights.view(-1, 1))
        
        # get label
        label = self.classifier(torch.cat((sides, tops, details, heights), dim = 1))
        return label

#%% create dataset class to load the data from csv and las files
class TrainDataset_AllChannels():
    """Tree species dataset."""

    def __init__(self, csv_or_las, root_dir=None, img_trans=None, pc_rotate=True,
                 height_noise=0.01, height_mean=None, height_sd=None, test=False,
                 res=512, n_sides=4, tree_id_col="TreeID", projection_backend="numpy", max_points_per_tree=1000000):

        self.img_trans = img_trans
        self.pc_rotate = pc_rotate
        self.height_noise = height_noise
        self.height_mean = height_mean
        self.height_sd = height_sd
        self.test = test
        self.res = res
        self.n_sides = n_sides
        self.tree_id_col = tree_id_col
        self.projection_backend = projection_backend
        self.max_points_per_tree = max_points_per_tree  # None or int, to limit memory use in laspy mode

        if isinstance(csv_or_las, str):
            # CSV path
            self.trees_frame = pd.read_csv(csv_or_las)
            self.root_dir = root_dir
        elif isinstance(csv_or_las, laspy.LasData):
            # laspy object
            las_data = csv_or_las
            dims = list(las_data.point_format.dimension_names)
            if tree_id_col not in dims:
                raise ValueError(
                    f"LAS dimension '{tree_id_col}' not found. Available dimensions: {dims}"
                )

            # ids = np.unique(las_data[tree_id_col])
            # Filter to valid points and compute per-tree stats, requiring >= 50 points
            min_points = 50
            tree_ids_arr = np.asarray(las_data[tree_id_col])
            z_arr = np.asarray(las_data.z)
            valid_mask = tree_ids_arr != 0
            df_tmp = pd.DataFrame({
                tree_id_col: tree_ids_arr[valid_mask].astype(np.int64, copy=False),
                "z": z_arr[valid_mask]
            })
            stats = df_tmp.groupby(tree_id_col).agg(
                z_min=("z", "min"),
                z_max=("z", "max"),
                count=("z", "size"),
            )
            stats = stats[stats["count"] >= min_points]
            if stats.empty:
                raise ValueError(f"No trees with >= {min_points} points were found in LAS.")
            heights = (stats["z_max"] - stats["z_min"]).astype(float)
            data = [[f"tree_{int(tid)}", -999, float(h), int(tid)]
                    for tid, h in zip(stats.index.values, heights.values)]
            self.trees_frame = pd.DataFrame(
                data, columns=["filename", "species_id", "tree_H", self.tree_id_col]
            )
            tree_ids = np.asarray(las_data[tree_id_col], dtype=np.int64)
            x = np.asarray(las_data.x, dtype=np.float32)
            y = np.asarray(las_data.y, dtype=np.float32)
            z = np.asarray(las_data.z, dtype=np.float32)

            # Sorted by tree_id
            order = np.argsort(tree_ids, kind="stable")
            self._tree_ids_np = tree_ids[order]
            self._x_np = x[order]
            self._y_np = y[order]
            self._z_np = z[order]

            # Precompute spans per tree_id for O(1) lookup (avoid searchsorted in __getitem__)
            uniq, starts, counts = np.unique(self._tree_ids_np, return_index=True, return_counts=True)
            self._tid2span = {int(u): (int(s), int(s + c)) for u, s, c in zip(uniq.tolist(), starts.tolist(), counts.tolist())}

            self.las_data = True
            self.root_dir = None
        else:
            raise TypeError("csv_or_las must be a CSV file path or laspy.LasData object.")

    @staticmethod
    def _random_thin(points: np.ndarray, max_points: int) -> np.ndarray:
        n = points.shape[0]
        if max_points is None or n <= max_points:
            return points
        idx = np.random.choice(n, size=max_points, replace=False)
        return points[idx]

    def _augment_torch(self, pts: torch.Tensor,
                       rotate_h_max: float = 22.5,
                       rotate_v_max: float = 180.0,
                       sampling_max: float = 0.1) -> torch.Tensor:
        # Optional random point drop (up to sampling_max fraction)
        if sampling_max > 0.0 and pts.shape[0] > 1:
            keep = max(1, int(pts.shape[0] * (1.0 - sampling_max)))
            idx = torch.randperm(pts.shape[0], device=pts.device)[:keep]
            pts = pts.index_select(0, idx)

        # Random Euler rotations (degrees -> radians)
        deg2rad = torch.pi / 180.0
        rx = (torch.rand((), device=pts.device) * 2 * rotate_h_max - rotate_h_max) * deg2rad
        ry = (torch.rand((), device=pts.device) * 2 * rotate_h_max - rotate_h_max) * deg2rad
        rz = (torch.rand((), device=pts.device) * 2 * rotate_v_max - rotate_v_max) * deg2rad

        cx, sx = torch.cos(rx), torch.sin(rx)
        cy, sy = torch.cos(ry), torch.sin(ry)
        cz, sz = torch.cos(rz), torch.sin(rz)

        # Build proper 3x3 rotation matrices on the same device/dtype
        Rx = torch.eye(3, dtype=pts.dtype, device=pts.device)
        Rx[1, 1] = cx;
        Rx[1, 2] = -sx
        Rx[2, 1] = sx;
        Rx[2, 2] = cx

        Ry = torch.eye(3, dtype=pts.dtype, device=pts.device)
        Ry[0, 0] = cy;
        Ry[0, 2] = sy
        Ry[2, 0] = -sy;
        Ry[2, 2] = cy

        Rz = torch.eye(3, dtype=pts.dtype, device=pts.device)
        Rz[0, 0] = cz;
        Rz[0, 1] = -sz
        Rz[1, 0] = sz;
        Rz[1, 1] = cz

        R = Rz @ Ry @ Rx
        pts = pts @ R.T

        # Shift z so min z == 0 (match NumPy augmentation)
        zmin = pts[:, 2].amin()
        pts[:, 2] -= zmin
        return pts

    def __len__(self):
        return len(self.trees_frame)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if hasattr(self, "las_data"):
            # In-memory laspy object
            tree_id = int(self.trees_frame.iloc[idx][self.tree_id_col])
            span = self._tid2span.get(tree_id)
            if span is None:
                raise ValueError(f"No points found for tree_id={tree_id}")
            lo, hi = span

            # Slice once, thin if requested
            tree = np.column_stack((
                self._x_np[lo:hi],
                self._y_np[lo:hi],
                self._z_np[lo:hi]
            ))  # float32, shape [K,3]

            if self.max_points_per_tree:
                tree = self._random_thin(tree, self.max_points_per_tree)

            if self.pc_rotate:
                if self.projection_backend == "numpy":
                    image = sv.points_to_images(au.augment(tree),
                                                res_im=self.res, num_side=self.n_sides)
                    image = torch.from_numpy(image)
                elif self.projection_backend == "torch": # torch
                    points = torch.from_numpy(tree).float()
                    points = self._augment_torch(points)
                    image = svt.points_to_images(points, res_im=self.res, num_side=self.n_sides)
                    #image = image.numpy()
                else : raise ValueError(f"Unknown projection_backend: {self.projection_backend}")
            else:
                if self.projection_backend == "numpy":
                    image = sv.points_to_images(tree, res_im=self.res, num_side=self.n_sides)
                    image = torch.from_numpy(image)
                elif self.projection_backend == "torch": # torch
                    points = torch.from_numpy(tree).float()
                    image = svt.points_to_images(points, res_im=self.res, num_side=self.n_sides)
                    #image = image.numpy()
                else : raise ValueError(f"Unknown projection_backend: {self.projection_backend}")

            if self.img_trans:
                image = self.img_trans(image)
            image = image.unsqueeze(1)
            height = torch.tensor(self.trees_frame.iloc[idx, 2], dtype=torch.float32)
            if self.height_noise > 0:
                height += np.random.normal(0, self.height_noise)
            height = (height - self.height_mean) / self.height_sd
            if self.test:
                return image, height, f"tree_{int(tree_id)}"
            label = torch.tensor(self.trees_frame.iloc[idx, 1], dtype=torch.int64)
            return image, height, label
        else:
            # CSV mode (original)
            las_name = os.path.join(
                self.root_dir,
                *self.trees_frame.iloc[idx, 0].split('/'))
            if self.pc_rotate:
                if self.projection_backend == "numpy":
                    image = sv.points_to_images(au.augment(las_name), res_im=self.res, num_side=self.n_sides)
                elif self.projection_backend == "torch": # torch
                    points = au.augment(las_name)
                    points = torch.from_numpy(points).float()
                    image = svt.points_to_images(points, res_im=self.res, num_side=self.n_sides)
                    #image = image.numpy()
                else : raise ValueError(f"Unknown projection_backend: {self.projection_backend}")
            else:
                if self.projection_backend == "numpy":
                    image = sv.points_to_images(rl.read_las(las_name), res_im=self.res, num_side=self.n_sides)
                elif self.projection_backend == "torch": # torch
                    points = rl.read_las(las_name)
                    points = torch.from_numpy(points).float()
                    image = svt.points_to_images(points, res_im=self.res, num_side=self.n_sides)
                    #image = image.numpy()
                else : raise ValueError(f"Unknown projection_backend: {self.projection_backend}")
            if self.projection_backend == "numpy":
                image = torch.from_numpy(image)
            if self.img_trans:
                image = self.img_trans(image)
            image = image.unsqueeze(1)
            height = torch.tensor(self.trees_frame.iloc[idx, 2], dtype=torch.float32)
            if self.height_noise > 0:
                height += np.random.normal(0, self.height_noise)
            height = (height - self.height_mean) / self.height_sd
            if self.test:
                las_path = self.trees_frame.iloc[idx, 0]
                return image, height, las_path
            label = torch.tensor(self.trees_frame.iloc[idx, 1], dtype=torch.int64)
            return image, height, label

    def weights(self):
        return torch.tensor(self.trees_frame["weight"].values)