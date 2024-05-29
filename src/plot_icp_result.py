import numpy as np
import open3d as o3d
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from pathlib import Path


def list2pc(li):
    arr = np.array(li)
    arr = arr[:, :3]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(arr))
    return pcd


def get_copy_without_z(pc):
    arr = np.asarray(pc.points)
    arr[:, 2] = 0
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(arr))


def get_tf_yaw_only(tf):
    # get tf with roll and pitch set to 0
    eulers = R.from_matrix(tf[:3, :3]).as_euler("xyz")
    eulers[:2] = 0
    tf_yaw_only = tf.copy()
    tf_yaw_only[:3, :3] = R.from_euler("xyz", eulers).as_matrix()
    return tf_yaw_only


def get_cam_transform_arr(seq_path):
    gt_cam_tf_arr = np.loadtxt(seq_path / "poses.txt")
    gt_cam_tf_arr = gt_cam_tf_arr.reshape(-1, 3, 4)
    last_row = np.zeros((gt_cam_tf_arr.shape[0], 1, 4))
    last_row[:, :, -1] = 1
    gt_cam_tf_arr = np.concatenate([gt_cam_tf_arr, last_row], axis=1)
    return gt_cam_tf_arr


def get_Tr_TrInv(seq_path):
    with open(seq_path / "calib.txt", "r") as f:
        lines = f.readlines()
        Tr = np.fromstring(lines[4].split(": ")[1], dtype=float, sep=" ").reshape(3, 4)
        Tr = np.row_stack([Tr, [0, 0, 0, 1]])
        Tr_inv = np.linalg.inv(Tr)
    return Tr, Tr_inv


def get_ground_truth_tf(src_idx, que_idx, gt_cam_tf_arr, Tr, Tr_inv):
    return Tr_inv @ np.linalg.inv(gt_cam_tf_arr[src_idx]) @ gt_cam_tf_arr[que_idx] @ Tr


def get_pose_error(tf_gt, tf, is_2d=False) -> tuple[np.ndarray, float]:
    def get_yaw_from_tf(tf):
        return np.arctan2(tf[1, 0], tf[0, 0])

    if is_2d:
        trans_error = np.abs(tf_gt[:2, 3] - tf[:2, 3])
        rot_error = abs(get_yaw_from_tf(tf_gt) - get_yaw_from_tf(tf))
        rot_error = rot_error if rot_error < np.pi else 2 * np.pi - rot_error
    else:
        trans_error = np.abs(tf_gt[:3, 3] - tf[:3, 3])
        rot = R.from_matrix(tf[:3, :3] @ tf_gt[:3, :3].T)
        rot_error = rot.magnitude()
    return trans_error, rot_error * 180 / np.pi


def get_non_outlier_indices(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    non_outlier_indices = np.where((data >= lower_bound) & (data <= upper_bound))
    return non_outlier_indices[0]


# %%
if __name__ == "__main__":
    # evaluate the pose estimation result

    DATASET_PATH = Path(__file__).parent / ".." / "data/dataset"
    SEQ_LIST = [0, 2, 5, 6, 7, 8]

    # Load data
    seq_trans_data = dict()
    seq_error_data = {
        "yaw_error": [],
        "yaw_error_raw": [],
        "inlier_ratio": [],
        "inlier_num": [],
    }
    use_3d = False
    for seq in SEQ_LIST:
        seq_path = DATASET_PATH / f"sequences/{seq:02d}"
        with open(Path(__file__).parent / f"../data/icp_data/{seq:02d}.pkl", "rb") as f:
            pair_data_list = pickle.load(f)
        gt_cam_tf_arr = get_cam_transform_arr(seq_path)
        pose_error_list = []
        angle_error_list = []
        for pair_data in tqdm(
            pair_data_list, desc=f"Calculating pose error after ICP for seq {seq:02d}"
        ):
            src_idx = pair_data["src_frame"]
            que_idx = pair_data["que_frame"]
            tf_icp = pair_data["icp_transform"]
            Tr, Tr_inv = get_Tr_TrInv(seq_path)
            tf_gt = get_ground_truth_tf(src_idx, que_idx, gt_cam_tf_arr, Tr, Tr_inv)
            pose_error, angle_error = get_pose_error(tf_gt, tf_icp, True)
            pose_error_list.append(pose_error)
            angle_error_list.append(angle_error)
        pose_error_arr = np.array(pose_error_list)
        angle_error_arr = np.array(angle_error_list)
        # only calculate those whose pose is not outlier
        inliers_idxs = get_non_outlier_indices(np.linalg.norm(pose_error_arr, axis=1))
        seq_trans_data[f"{seq:02d}"] = pose_error_arr
        seq_error_data["yaw_error"].append(np.mean(angle_error_arr[inliers_idxs]))
        seq_error_data["yaw_error_raw"].append(np.mean(angle_error_arr))
        seq_error_data["inlier_ratio"].append(
            inliers_idxs.shape[0] / angle_error_arr.shape[0]
        )
        seq_error_data["inlier_num"].append(int(inliers_idxs.shape[0]))

    # Show angle error
    print("Result")
    error_df = pd.DataFrame(seq_error_data, index=SEQ_LIST)
    angle_err_all = (error_df["yaw_error"] * error_df["inlier_num"]).sum()
    angle_mean_err_all = angle_err_all / error_df["inlier_num"].sum()
    print(error_df)
    print("mean yaw error over all sequences:", angle_mean_err_all)
    print(
        "mean yaw error with outliers over all sequences:",
        error_df["yaw_error_raw"].mean(),
    )

    # Split the errors into separate lists for plotting.
    x_errors = {
        seq: [error[0] for error in errors] for seq, errors in seq_trans_data.items()
    }
    y_errors = {
        seq: [error[1] for error in errors] for seq, errors in seq_trans_data.items()
    }

    # Define the number of sequences and create a positions list for the box plots.
    n_sequences = len(seq_trans_data)
    positions = np.arange(n_sequences)
    fig, ax = plt.subplots(figsize=(12, 4))
    width = 0.13  # the width of the boxes
    width_box = 0.19
    sep = 0.15
    for i, seq in enumerate(seq_trans_data.keys()):
        pos_ssc_x = [pos - (2.5 + sep) * width for pos in positions]
        pos_x = [pos - (1 + sep) * width for pos in positions]
        pos_ssc_y = [pos + (1 + sep) * width for pos in positions]
        pos_y = [pos + (2.5 + sep) * width for pos in positions]

        bp_x = ax.boxplot(
            x_errors[seq],
            positions=[pos_x[i]],
            widths=width_box,
            patch_artist=True,
            boxprops=dict(facecolor="tab:orange"),
            medianprops=dict(color="red"),
            showfliers=False,
        )
        bp_y = ax.boxplot(
            y_errors[seq],
            positions=[pos_y[i]],
            widths=width_box,
            patch_artist=True,
            boxprops=dict(facecolor="blue"),
            medianprops=dict(color="red"),
            showfliers=False,
        )

    # Adjust axis and show
    ax.set_xticks(positions)
    ax.set_xticklabels(seq_trans_data.keys())
    for pos in positions[:-1]:
        ax.axvline(x=pos + 0.5, color="grey")
    # 调整label大小
    ax.set_ylabel("Translation Error (m)", fontsize=14)
    ax.set_xlabel("Sequence", fontsize=16)
    box_legend = [
        bp_x["boxes"][0],
        bp_y["boxes"][0],
    ]
    name_legend = [
        "Ours x",
        "Ours y",
    ]
    plt.legend(box_legend, name_legend, loc="upper right", fontsize=12)
    plt.grid()
    # plt.xlim(-0.5, 5.5)
    plt.savefig("boxplot.pdf", bbox_inches="tight")
    plt.show()
