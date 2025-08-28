import argparse
import copy
import json
import os
import subprocess
import tempfile
from glob import glob
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from natsort import natsorted
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation as R

from kitti_odometry import KittiEvalOdom

# ===== eval_plots.py Functions =====


def rmse(np_array):
    mse = np.mean(np.square(np_array))
    rmse = np.sqrt(mse)
    return rmse


def find_closest_gt_time(pred_time, gt_timestamps):
    return np.argmin(np.abs(gt_timestamps - pred_time))


def load_ground_truth(gt_path):
    """Load ground truth coordinates from either a CSV file or JSON directory."""
    gt_path = Path(gt_path)

    # Check if it's a directory (JSON files) or file (CSV)
    if gt_path.is_dir():
        # Load from JSON directory
        json_files = sorted(gt_path.glob("*.json"))
        timestamps, x, y, z, yaw = [], [], [], [], []
        frames = []

        for f in json_files:
            data = json.loads(f.read_text())
            if all(key in data for key in ["x", "y", "rel_altitude"]):
                timestamps.append(data["timestamp"])
                x.append(data["x"])
                y.append(data["y"])
                z.append(data["rel_altitude"])
                # yaw is optional
                if "yaw" in data:
                    yaw.append(data["yaw"])
                else:
                    yaw.append(0.0)  # default to 0 if no yaw
                frames.append(int(f.stem.replace("frame", "")))
            else:
                print(f"Warning: missing required fields in {f.name}")

        return (
            np.array(timestamps),
            np.array(x),
            np.array(y),
            np.array(z),
            np.array(yaw),
            np.array(frames),
        )

    else:
        # Load from CSV file (backward compatibility)
        try:
            df = pd.read_csv(gt_path, delimiter=" ")
        except:
            df = pd.read_csv(gt_path, delimiter=",")
        if {"x", "y", "z", "yaw"}.issubset(df.columns):
            return (
                df["timestamp"].values,
                df["x"].values,
                df["y"].values,
                df["z"].values,
                df["yaw"].values,
                np.arange(len(df)),
            )
        else:
            cols = df.columns
            print(f"Warning: missing x/y/z/yaw columns; using {cols[1:5]}")
            return (
                df[cols[0]].values,
                df[cols[1]].values,
                df[cols[2]].values,
                df[cols[3]].values,
                df[cols[4]].values,
                np.arange(len(df)),
            )


def load_predictions(pred_dir):
    """Load prediction coordinates from JSON files in a directory."""
    pred_dir = Path(pred_dir)
    json_files = sorted(pred_dir.glob("*.json"))
    timestamps, x, y, z, yaw, frames = [], [], [], [], [], []
    for f in json_files:
        data = json.loads(f.read_text())
        if "x" in data and "y" in data:  # Only require x and y, z is optional
            timestamps.append(data["timestamp"])
            x.append(data["x"])
            y.append(data["y"])
            z.append(data.get("z", 0.0))  # Default to 0 if z is not provided
            yaw.append(data.get("p", 0.0))
            frames.append(int(f.stem.replace("pred", "")))
    return (
        np.array(timestamps),
        np.array(x),
        np.array(y),
        np.array(z),
        np.array(yaw),
        np.array(frames),
    )


def pose_matrix(x, y, z, yaw, pitch=0.0, roll=0.0):
    """Create 4x4 transformation matrix from translation and Euler angles."""
    rot = R.from_euler("zyx", [yaw, pitch, roll], degrees=False)
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def kitti_write_poses_to_txt(poses, file_path):
    """
    Writes a list of 4x4 transformation matrices to a KITTI odometry pose txt file.

    Args:
        pose_matrices (list of np.ndarray): List where each element is a 4x4 numpy array
                                            representing a pose matrix of a frame.
        file_path (str): The filepath where the pose file will be saved.

    The output file will contain one line per frame, each with 12 pose values
    corresponding to the first 3 rows and 4 columns of the 4x4 matrix.
    """
    with open(file_path, "w") as f:
        for i, T in enumerate(poses):
            # Extract first 3 rows and 4 columns, flatten to 1D array
            pose_line_values = T[:3, :].flatten()
            # Convert to space-separated string
            pose_line_str = " ".join(
                [str(i), *list(map(str, pose_line_values))]
            )
            f.write(pose_line_str + "\n")


def kitti_read_errors_txt(file_path) -> pd.DataFrame:
    """Reads the errors from a KITTI odometry error txt file."""
    df = pd.read_csv(
        file_path,
        sep=" ",
        names=[
            "first_frame",
            "rotation_error",
            "translation_error",
            "subsequence_length",
            "speed",
        ],
    )
    df["speed"] = (
        df["speed"] / 2.5
    )  # Because it has been calculated considering 10 fps by default...
    return df


def get_interpolated_frame_to_error(frame_to_error: dict, frame_count: int):
    """Calculates the continuous frame to error."""
    x = np.array(sorted(frame_to_error.keys()))
    y = np.array([frame_to_error[k] for k in x])
    new_keys = np.arange(0, frame_count, 1)
    new_values = np.interp(new_keys, x, y)

    return new_values


def apply_global_rotation(pred_poses, rotation_matrix):
    """Applies a fixed global rotation to all predicted poses."""
    rotated = []
    for T in pred_poses:
        R_old = T[:3, :3]
        t_old = T[:3, 3]
        R_new = rotation_matrix @ R_old
        t_new = rotation_matrix @ t_old
        T_new = np.eye(4)
        T_new[:3, :3] = R_new
        T_new[:3, 3] = t_new
        rotated.append(T_new)
    return rotated


def apply_global_scale(pred_poses, scale_factor):
    """Applies a fixed global scale to all predicted poses."""
    scaled = []
    for T in pred_poses:
        T_new = np.eye(4)
        T_new[:3, :3] = T[:3, :3]  # Keep rotation unchanged
        T_new[:3, 3] = scale_factor * T[:3, 3]  # Scale translation
        scaled.append(T_new)
    return scaled


def kitti_trajectory_times(gt_timestamps):
    """Compute time for each pose w.r.t frame-0
    Args:
        poses (dict): {idx: 4x4 array}
    Returns:
        dist (float list): distance of each pose w.r.t frame-0
    """
    times = [0]
    for i in range(len(gt_timestamps) - 1):
        P1 = gt_timestamps[i]
        P2 = gt_timestamps[i + 1]
        times.append(times[i] + P2 - P1)
    return np.array(times)


def kitti_translation_error(pose_error):
    """Compute translation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        trans_error (float): translation error
    """
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    trans_error = np.sqrt(dx**2 + dy**2 + dz**2)
    return trans_error


def kitti_rotation_error(pose_error):
    """Compute rotation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        rot_error (float): rotation error
    """
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5 * (a + b + c - 1.0)
    rot_error = np.arccos(max(min(d, 1.0), -1.0))
    return rot_error / (np.pi / 180)


def kitti_last_frame_from_segment_length(values, first_frame, segment_length):
    """Find frame (index) that away from the first_frame with
    the required time.
    Args:
        values (times/distances) (float list): values of each pose w.r.t frame-0
        first_frame (int): start-frame index
        segment_length (float): required time
    Returns:
        i (int) / -1: end-frame index. if not found return -1
    """
    for i in range(first_frame, len(values), 1):
        if values[i] > (values[first_frame] + segment_length):
            return i
    return -1


def calculate_errors_with_subsequence_length_in_seconds(
    gt_poses,
    pred_poses,
    gt_poses_timestamps,
    segment_length_s=60,
    error_key=None,
):
    times = kitti_trajectory_times(gt_poses_timestamps)
    t_err = {}
    drift_per_min = {}
    for first_frame in range(0, len(gt_poses), 1):
        last_frame = kitti_last_frame_from_segment_length(
            times, first_frame, segment_length_s
        )

        # Continue if sequence not long enough
        if last_frame == -1:
            continue

        # compute rotational and translational errors
        pose_delta_gt = np.dot(
            np.linalg.inv(gt_poses[first_frame]), gt_poses[last_frame]
        )
        pose_delta_result = np.dot(
            np.linalg.inv(pred_poses[first_frame]), pred_poses[last_frame]
        )
        pose_error = np.dot(np.linalg.inv(pose_delta_result), pose_delta_gt)

        t_error = kitti_translation_error(pose_error)
        drift_per_min[first_frame] = t_error / 60  # m/min

        t_err[first_frame] = t_error
    translation_error__rmse = rmse(list(t_err.values()))
    translation_error__avg = np.mean(list(t_err.values()))
    drift_per_min__rmse = rmse(list(drift_per_min.values()))
    drift_per_min__avg = np.mean(list(drift_per_min.values()))
    drift_per_min__per_frame = get_interpolated_frame_to_error(
        drift_per_min, len(gt_poses)
    )

    t_err__per_frame = get_interpolated_frame_to_error(t_err, len(gt_poses))

    result = {
        "translation_error__rmse": translation_error__rmse,
        "translation_error__avg": translation_error__avg,
        "translation_error__per_frame": t_err__per_frame,
        "drift_per_min__rmse": drift_per_min__rmse,
        "drift_per_min__avg": drift_per_min__avg,
        "drift_per_min__per_frame": drift_per_min__per_frame,
    }
    return result.get(error_key, result)


def rotate_prediction_and_calculate_pos_error(
    pred_poses,
    gt_poses,
    rotation_degrees: float,
    pos_error_dimensions: int = 2,
):
    """Calculates error after rotating the prediction by given degrees."""
    theta = np.pi * rotation_degrees / 180  # Convert degrees to radians
    R_z = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    pred_poses__rotated = apply_global_rotation(pred_poses, R_z)

    gt_xyz = np.array([p[:3, 3] for p in gt_poses])
    pred_xyz = np.array([p[:3, 3] for p in pred_poses__rotated])

    pos_errors = np.sqrt(
        np.sum(
            (
                gt_xyz[:, :pos_error_dimensions]
                - pred_xyz[:, :pos_error_dimensions]
            )
            ** 2,
            axis=1,
        )
    )

    return rmse(pos_errors)


def scale_prediction_and_calculate_pos_error(
    pred_poses,
    gt_poses,
    scale_factor: float,
    pos_error_dimensions: int = 2,
):
    """Calculates error after scaling the prediction by given factor."""
    pred_poses__scaled = apply_global_scale(pred_poses, scale_factor)

    gt_xyz = np.array([p[:3, 3] for p in gt_poses])
    pred_xyz = np.array([p[:3, 3] for p in pred_poses__scaled])

    pos_errors = np.sqrt(
        np.sum(
            (
                gt_xyz[:, :pos_error_dimensions]
                - pred_xyz[:, :pos_error_dimensions]
            )
            ** 2,
            axis=1,
        )
    )

    return rmse(pos_errors)


def find_optimal_rotation_angle(
    pred_poses, gt_poses, pos_error_dimensions: int = 2
):
    """Finds the optimal rotation angle (degree) for the given pred & gt."""

    def objective(degree):
        return rotate_prediction_and_calculate_pos_error(
            pred_poses, gt_poses, degree, pos_error_dimensions
        )

    res = minimize_scalar(
        objective,
        bounds=(-180.0, 180.0),
        method="bounded",
        options={"xatol": 1e-5},
    )

    # Round to 2 decimal places
    best_angle = round(res.x, 2)
    return best_angle


def find_optimal_scale_factor(
    pred_poses, gt_poses, pos_error_dimensions: int = 2
):
    """Finds the optimal scale factor for the given pred & gt."""

    def objective(scale_factor):
        return scale_prediction_and_calculate_pos_error(
            pred_poses, gt_poses, scale_factor, pos_error_dimensions
        )

    res = minimize_scalar(
        objective,
        bounds=(0.0001, 100.0),
        method="bounded",
        options={"xatol": 1e-5},
    )

    # Round to 6 decimal places
    best_scale = round(res.x, 6)
    return best_scale


def plot_trajectories(
    pred_x,
    pred_y,
    pos_errors,
    translation_errors,
    rotation_errors,
    timestamps,
    gt_x=None,
    gt_y=None,
    gt_yaw=None,
    traj_2d_plot_path=None,
    title=None,
):
    """Plot prediction (and optional ground truth) trajectories in 2D."""
    # Set fixed normalization for color gradient (0 to 100m)
    norm = Normalize(vmin=0, vmax=np.max(pos_errors))

    # Create figure with 3 subplots vertically stacked
    _, (ax1, ax2, ax3, ax4) = plt.subplots(
        4,
        1,
        figsize=(12, 20),
        sharex=False,
        gridspec_kw={"height_ratios": [3, 1, 1, 1]},
    )

    # First subplot - Trajectory plot
    if gt_x is not None:
        # center & rotate GT (only if yaw is available and non-zero)
        cx = gt_x[0] if isinstance(gt_x, np.ndarray) else gt_x.iloc[0]
        cy = gt_y[0] if isinstance(gt_y, np.ndarray) else gt_y.iloc[0]
        pts = np.vstack((gt_x - cx, gt_y - cy, np.zeros_like(gt_x))).T

        if gt_yaw is not None:
            yaw = (
                gt_yaw[0] if isinstance(gt_yaw, np.ndarray) else gt_yaw.iloc[0]
            )
            if yaw != 0:  # only rotate if yaw is non-zero
                rot = R.from_euler("z", yaw, degrees=True)
                gt_r = rot.apply(pts)
                gx, gy = gt_r[:, 0], gt_r[:, 1]
            else:
                gx, gy = gt_x - cx, gt_y - cy
        else:
            gx, gy = gt_x - cx, gt_y - cy

        ax1.plot(
            -gx,
            gy,
            color="darkgray",
            linestyle="-",
            linewidth=2,
            label="Ground Truth",
        )
        ax1.scatter(
            -gx[0],
            gy[0],
            color="lime",
            marker="o",
            s=100,
            label="GT Start",
            zorder=3,
        )
        ax1.scatter(
            -gx[-1],
            gy[-1],
            color="red",
            marker="o",
            s=100,
            label="GT End",
            zorder=3,
        )

    # predictions
    for i in range(len(pred_x) - 1):
        ax1.plot(
            -pred_x[i : i + 2],
            pred_y[i : i + 2],
            color=plt.cm.jet(norm(pos_errors[min(i, len(pos_errors) - 1)])),
            linewidth=2,
        )

    pred_end_x = (
        pred_x[-1] if isinstance(pred_x, np.ndarray) else pred_x.iloc[-1]
    )
    pred_end_y = (
        pred_y[-1] if isinstance(pred_y, np.ndarray) else pred_y.iloc[-1]
    )
    pred_start_x = (
        pred_x[0] if isinstance(pred_x, np.ndarray) else pred_x.iloc[0]
    )
    pred_start_y = (
        pred_y[0] if isinstance(pred_y, np.ndarray) else pred_y.iloc[0]
    )

    ax1.scatter(
        -pred_start_x,
        pred_start_y,
        color="lime",
        marker="x",
        s=100,
        label="Predicted Start",
        zorder=3,
    )
    ax1.scatter(
        -pred_end_x,
        pred_end_y,
        color="red",
        marker="x",
        s=100,
        label="Predicted End",
        zorder=3,
    )

    # Add colorbar to the first subplot
    sm = ScalarMappable(cmap=plt.cm.jet, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1)  # Specify ax1 for the colorbar
    cbar.set_label("Error (meters)")

    ax1.set_title(
        (f"{title} - " if title else "")
        + "Trajectory Comparison with Error Visualization"
    )
    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Y Position (m)")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.axis("equal")
    ax1.legend()

    # Second subplot - Positional Error over time plot
    flight_time_minutes = (timestamps[-1] - timestamps[0]) / 60
    x_data = np.arange(
        0, flight_time_minutes, flight_time_minutes / len(pos_errors)
    )[: len(pos_errors)]
    ax2.plot(x_data, pos_errors, color="red", label="Positional Error")
    ax2.set_xlabel("Time (minutes)")
    ax2.set_ylabel("Error (meters)")
    ax2.legend()
    ax2.set_title("Positional Error per Step")
    ax2.grid(True)

    # Third subplot - Translation Error over time plot
    for subsequence_length, translation_error in translation_errors.items():
        x_data_translation = np.arange(
            0, flight_time_minutes, flight_time_minutes / len(translation_error)
        )[: len(translation_error)]
        ax3.plot(
            x_data_translation,
            translation_error,
            label=f"Subsequence Length: {subsequence_length}m",
        )
    ax3.set_xlabel("Time (minutes)")
    ax3.set_ylabel("Error (m/m)")
    ax3.legend()
    ax3.set_title("Translation Error (m/m)")
    ax3.grid(True)

    # Fourth subplot - Rotation Error over time plot
    for subsequence_length, rotation_error in rotation_errors.items():
        x_data_rotation = np.arange(
            0, flight_time_minutes, flight_time_minutes / len(rotation_error)
        )[: len(rotation_error)]
        ax4.plot(
            x_data_rotation,
            rotation_error,
            label=f"Subsequence Length: {subsequence_length}m",
        )
    ax4.set_xlabel("Time (minutes)")
    ax4.set_ylabel("Error (deg/m)")
    ax4.legend()
    ax4.set_title("Rotation Error (deg/m)")
    ax4.grid(True)

    plt.tight_layout()
    if traj_2d_plot_path:
        plt.savefig(
            traj_2d_plot_path,
            dpi=300,
            bbox_inches="tight",
        )
    else:
        plt.show()
    plt.close()


# ===== Utility Functions =====


def read_gt_csv(gt_csv_path):
    """Legacy function to read ground truth from CSV (kept for backward compatibility)."""
    try:
        df = pd.read_csv(gt_csv_path)
        assert "timestamp" in df.columns
    except:
        df = pd.read_csv(gt_csv_path, delimiter=" ")
    return df


def read_folder_of_jsons(folder_path: str):
    frames = natsorted(
        [f for f in os.listdir(folder_path) if f.endswith(".json")]
    )
    frames_data = []
    for fname in frames:
        with open(os.path.join(folder_path, fname), "r") as f:
            data = json.load(f)
        frames_data.append((fname, data))
    return frames_data


def align_gt_with_pred_timestamps(
    gt_path: str,
    pred_folder_path: str,
    new_gt_path: str,
):
    """Aligns the gt csv with the prediction timestamps and writes the updated gt csv."""
    gt_df = pd.read_csv(gt_path, delimiter=" ")
    gt_timestamps = np.sort(gt_df["timestamp"])

    pred = load_predictions(pred_folder_path)
    pred_timestamps = pred[0]
    pred_timestamps = np.sort(pred_timestamps)
    while True:
        if pred_timestamps[0] == 0:
            pred_timestamps = np.delete(pred_timestamps, 0, axis=None)
        else:
            break

    gt_df["timestamp"] = gt_df["timestamp"] + (
        pred_timestamps[0] - gt_timestamps[0]
    )
    gt_df.to_csv(new_gt_path, sep=" ", index=False)


def show_pngs_side_by_side(
    image_paths,
    titles,
    main_title,
    output_path,
):
    fig, axes = plt.subplots(1, len(image_paths), figsize=(15, 5))

    # Read and show each image
    for i, (image_path, title) in enumerate(zip(image_paths, titles)):
        axes[i].imshow(mpimg.imread(image_path))
        axes[i].set_title(title, fontsize=8)
        axes[i].axis("off")

    plt.tight_layout()
    plt.suptitle(main_title, fontsize=12)
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def flight_analysis(
    gt_path: str,
    pred_folder: str,
    output_path: str,
    pos_error_dimensions: int = 3,
    rotation_degrees: float = None,
    scale_factor: float = None,
    dont_use_gt_z_for_pred_z: bool = False,
    subsequence_lengths: list[int] = [100],  # in meters
    save_files: bool = False,
):
    os.makedirs(output_path, exist_ok=True)
    metrics = {}

    # Load data...
    gt_data = load_ground_truth(gt_path)
    predicted_data = load_predictions(pred_folder)
    print(
        f"📊 Loaded {len(gt_data[0])} ground truth frames & {len(predicted_data[0])} predicted frames..."
    )

    gt_timestamps = gt_data[0]
    predicted_timestamps = predicted_data[0]

    # Create synced poses...
    gt_poses, pred_poses = [], []
    gt_poses_timestamps = []
    for i, pred_time in enumerate(predicted_timestamps):
        idx = find_closest_gt_time(pred_time, gt_timestamps)
        gt_poses.append(
            pose_matrix(
                gt_data[1][idx],
                gt_data[2][idx],
                gt_data[3][idx],
                gt_data[4][idx],
            )
        )
        pred_poses.append(
            pose_matrix(
                predicted_data[1][i],
                predicted_data[2][i],
                (
                    gt_data[3][idx]
                    if not dont_use_gt_z_for_pred_z
                    else predicted_data[3][i]
                ),
                predicted_data[4][i],
            )
        )
        gt_poses_timestamps.append(gt_data[0][idx])
    print(f"🔗 Created {len(gt_poses)} synced poses...")

    # Rotate the predictions...
    if rotation_degrees is None:
        rotation_degrees = find_optimal_rotation_angle(
            pred_poses, gt_poses, pos_error_dimensions
        )
        print(f"🧩 Found optimal rotation angle: {rotation_degrees} degrees...")
    theta = np.pi * rotation_degrees / 180  # Convert degrees to radians
    R_z = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    pred_poses__rotated = apply_global_rotation(pred_poses, R_z)
    print(f"🌀 Rotated predictions by {rotation_degrees} degrees...")

    # Scale the predictions...
    if scale_factor is None:
        scale_factor = find_optimal_scale_factor(
            pred_poses__rotated, gt_poses, pos_error_dimensions
        )
        print(f"🧩 Found optimal scale factor: {scale_factor}x...")
    pred_poses__rotated_scaled = apply_global_scale(
        pred_poses__rotated, scale_factor
    )
    print(f"⛶ Scaled predictions by {scale_factor}x...")

    # Calculate KITTI errors...
    with tempfile.TemporaryDirectory() as tmp_dir:
        gt_dir = os.path.join(tmp_dir, "gt")
        pred_dir = os.path.join(tmp_dir, "pred")
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)

        kitti_write_poses_to_txt(gt_poses, os.path.join(gt_dir, "01.txt"))
        kitti_write_poses_to_txt(
            pred_poses__rotated_scaled, os.path.join(pred_dir, "01.txt")
        )

        eval_tool = KittiEvalOdom()
        eval_tool.eval(
            gt_dir,
            pred_dir,
            alignment="scale",
            seqs=["01"],
        )

        errors_df = kitti_read_errors_txt(
            os.path.join(pred_dir, "errors", "01.txt")
        )
        translation_error_per_m = errors_df["translation_error"].mean()  # m/m
        rotation_error_per_m = (
            errors_df["rotation_error"].mean() / np.pi * 180
        )  # deg/m

        # Calculate the errors...
        translation_errors = {}
        rotation_errors = {}
        per_frame_translation_errors = {}
        per_frame_rotation_errors = {}
        for subsequence_length in subsequence_lengths:
            df_filtered = errors_df[
                errors_df["subsequence_length"] == subsequence_length
            ]
            translation_errors[subsequence_length] = np.array(
                df_filtered["translation_error"]
            )
            rotation_errors[subsequence_length] = (
                np.array(df_filtered["rotation_error"]) / np.pi * 180
            )  # Convert to degrees

            per_frame_translation_errors[subsequence_length] = (
                get_interpolated_frame_to_error(
                    df_filtered.set_index("first_frame")[
                        "translation_error"
                    ].to_dict(),
                    len(gt_poses),
                )
            )
            per_frame_rotation_errors[subsequence_length] = (
                get_interpolated_frame_to_error(
                    df_filtered.set_index("first_frame")[
                        "rotation_error"
                    ].to_dict(),
                    len(gt_poses),
                )
            )

    # Extract the x, y, z, yaw from the poses...
    gt_xyz = np.array([p[:3, 3] for p in gt_poses])
    gx, gy, gz = gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2]
    gyaw = np.array([p[3, 3] for p in gt_poses])

    pred_xyz = np.array([p[:3, 3] for p in pred_poses__rotated_scaled])
    px, py, pz = pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2]

    # Calculate the errors...
    pos_errors = np.linalg.norm(
        gt_xyz[:, :pos_error_dimensions] - pred_xyz[:, :pos_error_dimensions],
        axis=1,
    )
    print(f"📏 Calculated absolute positional errors...")

    # Calculate KITTI translation errors using 60s subsequence length...
    errors_with_60s_subsequence_length = (
        calculate_errors_with_subsequence_length_in_seconds(
            gt_poses, pred_poses__rotated_scaled, gt_poses_timestamps, 60
        )
    )
    print(f"📏 Calculated translation errors using 60s subsequence length...")

    # Plot the trajectories...
    plot_title = os.path.basename(pred_folder)
    traj_2d_plot_path = os.path.join(
        output_path,
        f"{plot_title}.png",
    )
    plot_trajectories(
        px,
        py,
        pos_errors,
        translation_errors,
        rotation_errors,
        gt_timestamps,
        gx,
        gy,
        gyaw,
        traj_2d_plot_path=traj_2d_plot_path if save_files else None,
        title=f"{plot_title}",
    )
    if save_files:
        print(f"📈 Saved 2D plot to {traj_2d_plot_path}...")

    # Save frame-wise errors in a CSV file...
    if save_files:
        frame_wise_errors = []
        for i, error in enumerate(gt_poses_timestamps):
            frame_wise_error = {
                "index": i,
                "timestamp": gt_poses_timestamps[i],
                "pos_error": error,
                "distance_travelled_m": (
                    np.linalg.norm(gt_xyz[i] - gt_xyz[i - 1]) if i > 0 else 0
                ),
                "time_travelled_s": (
                    gt_poses_timestamps[i] - gt_poses_timestamps[i - 1]
                    if i > 0
                    else 0
                ),
            }
            for subsequence_length in subsequence_lengths:
                frame_wise_error[
                    f"translation_error__{subsequence_length}m"
                ] = per_frame_translation_errors[subsequence_length][i]
                frame_wise_error[f"rotation_error__{subsequence_length}m"] = (
                    per_frame_rotation_errors[subsequence_length][i]
                )
            frame_wise_errors.append(frame_wise_error)
        df = pd.DataFrame(frame_wise_errors)
        frame_wise_errors_csv_path = os.path.join(
            output_path,
            f"frame-wise-errors--{plot_title}.csv",
        )
        df.to_csv(frame_wise_errors_csv_path, index=False)
        print(f"📄 Saved frame-wise errors to {frame_wise_errors_csv_path}...")

    # Calculate drift per minute and per meter...
    drift_per_min = round(
        errors_with_60s_subsequence_length["drift_per_min__avg"], 4
    )  # using 60s subsequence length
    drift_per_m = round(
        translation_error_per_m, 4
    )  # using 100m subsequence length

    print(f"\t📏 Drift in m/min: {drift_per_min} m/min")
    print(f"\t📏 Drift in m/m: {drift_per_m} m/m")

    metrics["drift_per_min"] = drift_per_min
    metrics["drift_per_m"] = drift_per_m

    for subsequence_length in subsequence_lengths:
        translation_error_rmse = rmse(translation_errors[subsequence_length])
        rotation_error_rmse = rmse(rotation_errors[subsequence_length])

        print(
            f"🔍 Translation error in m/m for {subsequence_length}m: {translation_error_rmse} m/m"
        )
        print(
            f"🔍 Rotation error in deg/m for {subsequence_length}m: {rotation_error_rmse} deg/m"
        )

        metrics[f"translation_error_per_m__{subsequence_length}m"] = (
            translation_error_rmse
        )
        metrics[f"rotation_error_per_m__{subsequence_length}m"] = (
            rotation_error_rmse
        )
    metrics["translation_error_%"] = translation_error_per_m * 100
    metrics["rotation_error_deg"] = rotation_error_per_m
    # NOTE: Uncomment this if you want to use 60s subsequence length for translation error and drift per minute...
    # metrics["translation_error__avg__with_60s_subsequence_length"] = errors_with_60s_subsequence_length["translation_error__avg"]
    # metrics["drift_per_min__avg__with_60s_subsequence_length"] = errors_with_60s_subsequence_length["drift_per_min__avg"]

    # Save the metrics...
    if save_files:
        metrics_json_path = os.path.join(
            output_path,
            f"metrics--{plot_title}.json",
        )
        with open(metrics_json_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"📊 Saved metrics JSON to {metrics_json_path}...")
    else:
        print(f"📊 Metrics: {json.dumps(metrics, indent=4)}")


def frames_to_video(frame_paths, output_path, fps=4):
    """
    Convert a sequence of image frames into an MP4 video using ffmpeg.

    Args:
        frame_paths (list of str): Ordered list of image file paths.
        output_path (str): Desired output MP4 path.
        fps (int or float): Frame rate of the resulting video.
    """
    if not frame_paths:
        raise ValueError("No frames provided.")

    # Create a temporary directory and symlink frames with a sequential pattern
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a temporary file listing selected frames
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            for frame_path in frame_paths:
                # Use absolute paths and escape any special characters
                abs_path = os.path.abspath(frame_path)
                escaped_path = abs_path.replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")
            frames_list_path = f.name

        # Build the ffmpeg command
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-f",
            "concat",
            "-safe",
            "0",
            "-r",
            str(fps),  # Set output framerate
            "-i",
            frames_list_path,
            "-c:v",
            "libx264",  # Use H.264 codec
            "-pix_fmt",
            "yuv420p",  # Ensure compatibility
            "-preset",
            "medium",  # Encoding preset
            output_path,
        ]

        subprocess.run(cmd, check=True)
