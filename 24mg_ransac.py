import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from spyral.core.run_stacks import form_run_string
import h5py as h5
import plotly.express as px
from spyral.core.point_cloud import PointCloud

from skimage.measure import LineModelND, ransac
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import tqdm 


run_to_examine = 30

workspace_path = Path("/Volumes/researchEXT/24Mg/mg24_spyral/")
pointcloud_path = workspace_path / "Pointcloud"
point_file_path = pointcloud_path / f"{form_run_string(run_to_examine)}.h5"
point_file = h5.File(point_file_path, 'r')
group = list(point_file.keys())[0]
group_data = point_file[group]    
attributes = dict(group_data.attrs)
min_event = attributes["min_event"]
max_event = attributes["max_event"]


output_path = f"/Volumes/researchEXT/24Mg/ransac_runs/00{run_to_examine}.h5"
file_out = h5.File(output_path, 'w')
group_label = file_out.create_group("ransac")
group_label.attrs["min_event"] = min_event
group_label.attrs["max_event"] = max_event

# counter = 0

for event in tqdm.tqdm(group_data, desc="Processing events"):
    # if counter > 1:
    #     break
    # counter += 1
    evt_group = group_label.create_group(f"{event}")
    nclus = 0
    X = group_data[event][:,:3]
    min_points = len(X) * 0.15
    # print(f"Event: {event}, Total points: {len(X)}, Minimum points for RANSAC: {min_points}")
    for idx in range(6):
        try:
            model_robust, inliers = ransac(
            X, LineModelND, min_samples=2, residual_threshold=12, max_trials=1000)

            outliers = inliers == False
            # print(f"Number of inliers: {len(inliers)}")
            if len(inliers) < min_points:
                X = X[outliers]
                continue
            
            cluster_group = evt_group.create_group(f"cluster_{idx}")
            nclus += 1
            key = "cloud"
            cluster_group.create_dataset(key, data=X[inliers])
            X = X[outliers]
    
        except ValueError as e:
            continue

    evt_group.attrs["n_clusters"] = nclus