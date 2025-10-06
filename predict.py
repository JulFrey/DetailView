from datetime import datetime

def run_predict(
    prediction_data,
    path_las="",
    model_path="./model_ft_202412171652_3",
    tree_id_col="TreeID",
    n_aug=10,
    output_dir="/output",
    path_csv_train="default_vals",
    path_csv_lookup="./lookup.csv",
    projection_backend="numpy",
    output_type="csv"
):
    import os
    import torch
    import numpy as np
    import pandas as pd
    from torchvision import transforms
    import laspy
    from datetime import datetime
    import parallel_densenet as net
    #import datetime

    if os.path.splitext(prediction_data)[1].lower() in ['.las', '.laz']:
        prediction_data = laspy.read(prediction_data)
        # check if coordinates exceed float32 mm precision; shift by updating header offsets (avoids OverflowError)
        if projection_backend == "torch":
            min_vals = [prediction_data.x.min(), prediction_data.y.min(), prediction_data.z.min()]
            if any(abs(val) > 1e5 for val in min_vals):
                hdr = prediction_data.header
                hdr.offsets = (
                    hdr.offsets[0] - min_vals[0],
                    hdr.offsets[1] - min_vals[1],
                    hdr.offsets[2] - min_vals[2],
                )
                prediction_data = laspy.LasData(hdr, prediction_data.points)

        # exclude prediction_data were TreeID == 0
        # ids = np.unique(prediction_data[tree_id_col])
        # ids = ids[ids != 0]  # skip 0 if needed
        # mask = np.isin(prediction_data[tree_id_col], ids)
        # prediction_data = prediction_data[mask]

    outfile = f"{output_dir}/predictions_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}_.csv"
    outfile_probs = f"{output_dir}/predictions_probs{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}_.csv"

    n_class = 33
    n_view = 7
    n_batch = 2**2
    res = 256
    n_sides = n_view - 3
    n_workers = 0

    if not os.path.exists(model_path):
        import requests
        response = requests.get("https://freidata.uni-freiburg.de/records/xw42t-6mt03/files/model_202305171452_60?download=1")
        model_path = "/model_202305171452_60"
        with open(model_path, 'wb') as f:
            f.write(response.content)

    if os.path.exists(path_csv_train):
        train_metadata = pd.read_csv(path_csv_train)
        train_height_mean = np.mean(train_metadata["tree_H"])
        train_height_sd = np.std(train_metadata["tree_H"])
    else:
        train_height_mean = 15.2046
        train_height_sd = 9.5494

    os.environ["OMP_NUM_THREADS"] = "12"
    os.environ["OPENBLAS_NUM_THREADS"] = "12"
    os.environ["MKL_NUM_THREADS"] = "12"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "12"
    os.environ["NUMEXPR_NUM_THREADS"] = "12"

    model = net.SimpleView(n_classes=n_class, n_views=n_view)
    model.load_state_dict(torch.load(model_path))
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu")
    model.to(device)
    model.eval()

    img_trans = transforms.Compose([
        transforms.RandomVerticalFlip(0.5)])

    test_dataset = net.TrainDataset_AllChannels(
        prediction_data, path_las, img_trans=img_trans, pc_rotate=True,
        height_noise=0.01, test=True, res=res, n_sides=n_sides,
        height_mean=train_height_mean, height_sd=train_height_sd,
        tree_id_col=tree_id_col, projection_backend=projection_backend)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=int(n_batch), shuffle=False, pin_memory=True, num_workers=n_workers)

    all_paths = test_dataset.trees_frame.iloc[:, 0]
    data_probs = {path: [] for path in all_paths}

    for epoch in range(int(n_aug)):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] epoch: {epoch + 1}")
        if epoch == 0:
            prev_argmax = {path: None for path in all_paths}

        for b, t_data in enumerate(test_dataloader, 0):
            t_inputs, t_heights, t_paths = t_data
            t_inputs, t_heights = t_inputs.to(device), t_heights.to(device)
            t_preds = model(t_inputs, t_heights)
            t_probs = torch.nn.functional.softmax(t_preds, dim=1).cpu().detach().numpy()
            for j, path in enumerate(t_paths):
                if not any(data_probs[path]):
                    data_probs[path] = t_probs[j, :]
                else:
                    data_probs[path] += t_probs[j, :]

        changes = 0
        curr_argmax = {}
        for path, probs in data_probs.items():
            cls = None if not any(probs) else int(np.argmax(probs))
            curr_argmax[path] = cls
            if prev_argmax.get(path) is not None and cls is not None and cls != prev_argmax[path]:
                changes += 1
        print(f"aggregation changes vs. previous epoch: {changes}")
        prev_argmax = curr_argmax

    max_prob_class = {key: np.argmax(array) for key, array in data_probs.items()}
    max_prob = {key: (array.max() / array.sum()) if np.any(array) else 0.0 for key, array in data_probs.items()}
    df = pd.DataFrame({
        "filename": list(max_prob_class.keys()),
        "species_id": list(max_prob_class.values()),
        "species_prob": list(max_prob.values())
    })

    lookup = pd.read_csv(path_csv_lookup)
    joined = pd.merge(df, lookup, on='species_id')

    data_probs_df = pd.DataFrame.from_dict(data_probs, orient='index').reset_index()
    col_labels = lookup['species']
    data_probs_df.columns = pd.concat([pd.Series(["File"]), col_labels])
    if output_type in ["csv", "both"]:
        joined.to_csv(outfile, index=False)
        data_probs_df.to_csv(outfile_probs, index=False)
    if output_type in ["las", "both"]:
        if not isinstance(prediction_data, laspy.LasData):
            print("Error: prediction_data must be provided as a single LAS/LAZ file to output LAS with predictions.")
        else:
            prediction_data.add_extra_dim(
                laspy.ExtraBytesParams(name="species_id", type=np.uint8))
            prediction_data.add_extra_dim(
                laspy.ExtraBytesParams(name="species_prob", type=np.float32))
            species_ids = np.zeros(prediction_data.X.shape[0], dtype=np.uint8)
            species_probs = np.zeros(prediction_data.X.shape[0], dtype=np.float32)
            # Vectorized assignment by mapping point TreeIDs to predictions
            # parse tree_id from filenames
            joined["tree_id"] = joined["filename"].str.extract(r"(\d+)$").astype(int)

            # build species map
            _species_map = joined.set_index("tree_id")["species_id"].astype(np.uint8)

            # build max-prob map
            _prob_map = joined.set_index("tree_id")["species_prob"].astype(np.float32)

            # map for all points in one shot
            _point_tids = pd.Series(np.asarray(prediction_data[tree_id_col]), copy=False)
            species_ids = _point_tids.map(_species_map).fillna(255).astype(np.uint8).to_numpy()
            species_probs = _point_tids.map(_prob_map).fillna(0.0).astype(np.float32).to_numpy()
            prediction_data.species_id = species_ids
            prediction_data.species_prob = species_probs

            # revert offsetting if applied before
            if projection_backend == "torch" and any(abs(val) > 1e5 for val in min_vals):
                hdr = prediction_data.header
                hdr.offsets = (
                    hdr.offsets[0] + min_vals[0],
                    hdr.offsets[1] + min_vals[1],
                    hdr.offsets[2] + min_vals[2],
                )
                prediction_data = laspy.LasData(hdr, prediction_data.points)

            outlas_path = f"{output_dir}/predictions_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}_.laz"
            prediction_data.write(outlas_path)
            print(f"Wrote: {outlas_path}")
    return outfile, outfile_probs, joined, data_probs_df

# For CLI usage
if __name__ == "__main__":
    # record starting time
    t0 = datetime.now()
    print(f"Start time: {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    import argparse
    parser = argparse.ArgumentParser(description="Tree species prediction")
    parser.add_argument('--prediction_data', type=str, default=r"/input/circle_3_segmented.las")
    parser.add_argument('--path_las', type=str, default="")
    parser.add_argument('--model_path', type=str, default="./model_ft_202412171652_3")
    parser.add_argument('--tree_id_col', type=str, default='TreeID')
    parser.add_argument('--n_aug', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default="/output")
    parser.add_argument('--projection_backend', type=str, default="numpy", choices=["torch", "numpy"])
    parser.add_argument('--output_type', type=str, default="csv", choices=["csv", "las", "both"])
    args = parser.parse_args()
    run_predict(
        args.prediction_data,
        path_las=args.path_las,
        model_path=args.model_path,
        tree_id_col=args.tree_id_col,
        n_aug=args.n_aug,
        output_dir=args.output_dir,
        projection_backend=args.projection_backend,
        output_type=args.output_type
    )
    t1 = datetime.now()
    print(f"End time: {t1.strftime('%Y-%m-%d %H:%M:%S')}")
    # print elapsed time
    print(f"Elapsed time: {t1 - t0}")