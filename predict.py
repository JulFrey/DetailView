def run_predict(
    prediction_data,
    path_las="",
    model_path="./model_ft_202412171652_3",
    tree_id_col="TreeID",
    n_aug=10,
    output_dir="/output",
    path_csv_train="default_vals",
    path_csv_lookup="./lookup.csv"
):
    import os
    import torch
    import numpy as np
    import pandas as pd
    from torchvision import transforms
    import laspy
    from datetime import datetime
    import parallel_densenet as net

    if os.path.splitext(prediction_data)[1].lower() in ['.las', '.laz']:
        prediction_data = laspy.read(prediction_data)
        # exclude prediction_data were TreeID == 0
        # ids = np.unique(prediction_data[tree_id_col])
        # ids = ids[ids != 0]  # skip 0 if needed
        # mask = np.isin(prediction_data[tree_id_col], ids)
        # prediction_data = prediction_data[mask]

    outfile = f"{output_dir}/predictions_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}_.csv"
    outfile_probs = f"{output_dir}/predictions_probs{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}_.csv"

    n_class = 33
    n_view = 7
    n_batch = 2**8
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
        tree_id_col=tree_id_col)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=int(n_batch), shuffle=False, pin_memory=True, num_workers=n_workers)

    all_paths = test_dataset.trees_frame.iloc[:, 0]
    data_probs = {path: [] for path in all_paths}

    for epoch in range(int(n_aug)):
        print("epoch: %d" % (epoch + 1))
        for i, t_data in enumerate(test_dataloader, 0):
            t_inputs, t_heights, t_paths = t_data
            t_inputs, t_heights = t_inputs.to(device), t_heights.to(device)
            t_preds = model(t_inputs, t_heights)
            t_probs = torch.nn.functional.softmax(t_preds, dim=1)
            t_probs = t_probs.cpu().detach().numpy()
            for i, path in enumerate(t_paths):
                if not any(data_probs[path]):
                    data_probs[path] = t_probs[i, :]
                else:
                    data_probs[path] += t_probs[i, :]

    max_prob_class = {key: np.argmax(array) for key, array in data_probs.items()}
    df = pd.DataFrame({
        "filename": max_prob_class.keys(),
        "species_id": max_prob_class.values()})

    lookup = pd.read_csv(path_csv_lookup)
    joined = pd.merge(df, lookup, on='species_id')

    data_probs_df = pd.DataFrame.from_dict(data_probs, orient='index').reset_index()
    col_labels = lookup['species']
    data_probs_df.columns = pd.concat([pd.Series(["File"]), col_labels])

    joined.to_csv(outfile, index=False)
    data_probs_df.to_csv(outfile_probs, index=False)
    return outfile, outfile_probs, joined, data_probs_df

# For CLI usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tree species prediction")
    parser.add_argument('--prediction_data', type=str, default=r"/input/circle_3_segmented.las")
    parser.add_argument('--path_las', type=str, default="")
    parser.add_argument('--model_path', type=str, default="./model_ft_202412171652_3")
    parser.add_argument('--tree_id_col', type=str, default='TreeID')
    parser.add_argument('--n_aug', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default="/output")
    args = parser.parse_args()
    run_predict(
        args.prediction_data,
        path_las=args.path_las,
        model_path=args.model_path,
        tree_id_col=args.tree_id_col,
        n_aug=args.n_aug,
        output_dir=args.output_dir
    )