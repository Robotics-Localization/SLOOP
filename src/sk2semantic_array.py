import yaml
import numpy as np
import open3d as o3d
from os import listdir
from matplotlib import colormaps
import os


class SKPreprocess:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config_ = yaml.load(f, Loader=yaml.FullLoader)
        # 用于保存当前索引的成员变量
        self.curr_seq_idx_ = 0
        self.update_seq_path()
        self.curr_idx_ = 0
        label_list = listdir(self.curr_seq_path_ + "/labels")
        self.curr_idx_max_ = len(label_list) - 1

    def set_visualize_inst_pointcloud(self, flag):
        self.config_["use_vis_inst_cloud"] = flag

    def update_seq_path(self):
        self.curr_seq_path_ = "/".join(
            [
                self.config_["dataset_path"],        #dataset_path
                "sequences",
                f"{self.config_['seq_list'][self.curr_seq_idx_]:02d}",
            ]
        )
        

    def get_idx_and_seq_path(self, seq: int, frame_idx: int) -> (str, int):
        self.update_seq_path()
        
        curr_seq_path = self.curr_seq_path_
        # print(curr_seq_path)
        if frame_idx is not None:
            curr_idx = frame_idx
        else:
            curr_idx = self.curr_idx_
            self.curr_idx_ += self.config_["idx_step"]
            if self.curr_idx_ > self.curr_idx_max_:
                self.curr_idx_ = 0
                self.curr_seq_idx_ += 1
                if self.curr_seq_idx_ >= len(self.config_["seq_list"]):
                    print("All sequences have been processed.")
                    return None
                # self.update_seq_path()
        return curr_seq_path, curr_idx

    def load_data(
        self, curr_seq_path: str, curr_idx: int
    ) -> (np.ndarray, np.ndarray, np.ndarray):
        
        frame_points = np.fromfile(
            "/".join([curr_seq_path, "velodyne", f"{curr_idx:06d}.bin"]),
            dtype=np.float32,
        ).reshape(-1, 4)[:, :3]
        frame_labels = np.fromfile(
            "/".join([curr_seq_path, "labels", f"{curr_idx:06d}.label"]),
            dtype=np.uint32,
        ).squeeze()
        frame_sems = frame_labels & 0xFFFF
        frame_insts = frame_labels >> 16
        return frame_points, frame_sems, frame_insts

    def arr2downsample_pcd(self, arr: np.ndarray) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(arr)
        pcd = pcd.voxel_down_sample(voxel_size=self.config_["voxel_size"])
        return pcd

    def add_sem_inst(
        self, points: np.ndarray, sem: np.ndarray, inst: np.ndarray
    ) -> np.ndarray:
        sem_array = np.full((points.shape[0], 1), sem)
        inst_array = np.full((points.shape[0], 1), inst)
        return np.hstack([points, sem_array, inst_array])

    def cluster_with_sem_param(
        self, pcd: o3d.geometry.PointCloud, sem_used: int
    ) -> list:
        sem_used_param = self.config_["sem_used2cluster_param"][sem_used]
        cluster_idxs = np.array(
            pcd.cluster_dbscan(
                eps=sem_used_param["dbscan_eps"],
                min_points=sem_used_param["dbscan_min_pts"],
                print_progress=False,
            ),
            int,
        )
        pcd_clusters = []
        for cluster_idx in np.unique(cluster_idxs):
            if cluster_idx == -1:  # -1是没有被聚到任何一类中的点
                continue
            inlier_idx = np.argwhere(cluster_idxs == cluster_idx).squeeze(1)
            if (
                self.config_["use_cluster_min_pts"]
                and inlier_idx.shape[0] < sem_used_param["cluster_min_pts"]
            ):
                continue
            pcd_clusters.append(pcd.select_by_index(inlier_idx))
        return pcd_clusters

    def vis_semantic_array(
        self,
        semantic_array: np.ndarray,
        centers: list = None,
        semantics: list = None,
        normal: np.ndarray = None,
    ) -> None:
        semantic_pcd = o3d.geometry.PointCloud()
        semantic_pcd.points = o3d.utility.Vector3dVector(semantic_array[:, :3])
        colors_code = semantic_array[:, 3] + semantic_array[:, 4] * 3
        colors_code /= np.max(colors_code)
        colors = colormaps.get_cmap("tab20")(colors_code)
        semantic_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        vis_list = [semantic_pcd]
        if normal is not None:
            vis_list.append(self.get_o3d_vec(semantic_pcd.get_center(), normal * 3))
        if centers is not None and semantics is not None:
            for cent, sem in zip(centers, semantics):
                sphere = o3d.geometry.TriangleMesh.create_sphere(
                    radius=0.5
                )  # 创建一个球体，半径为0.05
                sphere.translate(cent)
                sphere.paint_uniform_color(self.config_["sem_used2color"][sem])
                vis_list.append(sphere)
        # o3d.visualization.draw_geometries(vis_list)

    def get_o3d_vec(self, start, vec, color=[1, 0, 0]):
        end = start + vec
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector(np.vstack((start, end)))
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.colors = o3d.utility.Vector3dVector([color])  # 设置线段颜色为红色
        return line

    def get_clustered_semantic_arary(
        self, seq: int = None, frame_idx: int = None
    ) -> (np.ndarray, list, list, np.ndarray):
        curr_seq_path, curr_idx = self.get_idx_and_seq_path(seq, frame_idx)
        frame_points, frame_sems, frame_insts = self.load_data(curr_seq_path, curr_idx)
        # 提取地面法向量
        pcd_raw = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(frame_points))
        # o3d.visualization.draw_geometries([pcd_raw])
        ground_model, _ = pcd_raw.segment_plane(
            distance_threshold=self.config_["ground_dist_thresh"],
            ransac_n=self.config_["ground_ransac_n"],
            num_iterations=self.config_["ground_num_iter"],
            probability=self.config_["ground_probability"],
        )
        ground_model = np.array(ground_model)
        # # 如果朝下，则反一下
        # if ground_model[2] < 0:
        #     ground_model = - ground_model

        # 按照sk中的语义顺序遍历语义点云，根据是否有实例信息进行聚类
        sem_sk_set = set(np.unique(frame_sems))
        clustered_semantic_arr = []
        inst_center_list = []
        inst_semantic_list = []
        cnt_insts_sk, cnt_insts_cluster = 0, 0
        for sem_sk in sem_sk_set:  # 遍历当前帧中的sk的语义
            if sem_sk not in self.config_["map_sk2used"].keys():
                continue
            sem_used = self.config_["map_sk2used"][sem_sk]
            # 提取对应语义的点云与实例
            sem_idxs = np.argwhere(frame_sems == sem_sk).squeeze(1)
            points_in_curr_sem = frame_points[sem_idxs]
            insts_in_curr_sem = frame_insts[sem_idxs]
            # 将语义与实例附到xyz后，同时提取实例中心
            insts_list = list(np.unique(insts_in_curr_sem))
            insts_list.sort()
            if insts_list[0] > 0:  # 如果有实例信息，则实例从非零数（不是从1开始，因为数据集中还做了重识别）开始标号，直接使用
                for idx, inst in enumerate(insts_list):
                    inst_idxs = np.argwhere(insts_in_curr_sem == inst).squeeze(1)
                    points_in_curr_inst = points_in_curr_sem[inst_idxs]
                    if self.config_["use_voxel_in_sk_insts"]:
                        pcd = self.arr2downsample_pcd(points_in_curr_inst)
                        points_in_curr_inst = np.array(pcd.points)
                    clustered_semantic_arr.append(
                        self.add_sem_inst(points_in_curr_inst, sem_used, idx + 1)
                    )
                    inst_center_list.append(np.mean(points_in_curr_inst, axis=0))
                    inst_semantic_list.append(sem_used)
                    cnt_insts_sk += 1
            else:  # 如果无实例信息，则实例全都标的事0，这时根据yaml中的参数按语义聚类
                pcd = self.arr2downsample_pcd(points_in_curr_sem)
                pcd_clusters = self.cluster_with_sem_param(pcd, sem_used)
                for cluster_idx, pcd_cluster in enumerate(pcd_clusters):
                    clustered_semantic_arr.append(
                        self.add_sem_inst(
                            np.array(pcd_cluster.points), sem_used, cluster_idx + 1
                        )
                    )
                    inst_center_list.append(pcd_cluster.get_center())
                    inst_semantic_list.append(sem_used)
                    cnt_insts_cluster += 1
                # 可视化聚类点云bbox及聚类前的语义点云方便调参
                if (
                    self.config_["use_vis_cluster"]
                    and sem_used in self.config_["vis_cluster_sem_used_list"]
                ):
                    vis_list = [pcd]
                    for cluster in pcd_clusters:
                        color = np.random.rand(3) * 0.5
                        cluster.paint_uniform_color(color)
                        vis_list.append(cluster.get_oriented_bounding_box())
                        vis_list[-1].color = color
                    pcd.paint_uniform_color(np.array([0.5, 0.5, 0.5]))
                    vis_list.extend(pcd_clusters)
                    o3d.visualization.draw_geometries(
                        vis_list,
                        str(sem_used) + "-" + self.config_["sem2str_used"][sem_used],
                    )
        clustered_semantic_arr = np.vstack(clustered_semantic_arr)
        # 打印该帧的语义、实例信息到终端
        if self.config_["use_print_screen"]:
            print(f"================== idx: {curr_idx} ===================")
            sem_strs_sk = [self.config_["sem2str_sk"][sk] for sk in sem_sk_set]
            sem_used_list = []
            for used in sem_sk_set:
                if used in self.config_["map_sk2used"].keys():
                    sem_used_list.append(self.config_["map_sk2used"][used])
            sem_strs_used = [
                self.config_["sem2str_used"][used] for used in sem_used_list
            ]
            print("kitti:", sem_strs_sk)
            print("used:", sem_strs_used)
            print(
                "instances num:",
                f" - sk: {cnt_insts_sk}",
                f" - cluster: {cnt_insts_cluster}",
                f" - total: {cnt_insts_sk + cnt_insts_cluster}",
                # sep="\n",
            )
        # 显示语义实例点云
        if self.config_["use_vis_inst_cloud"]:
            self.vis_semantic_array(
                clustered_semantic_arr,
                inst_center_list,
                inst_semantic_list,
                ground_model[:3],
            )
        inst_semantic_value_list = [self.config_["sem_used2color"][index] for index in inst_semantic_list]
        return (
            clustered_semantic_arr,
            np.array(inst_center_list),
            np.array(inst_semantic_value_list),         #inst_semantic_list
            ground_model[:3]
        )


if __name__ == "__main__":
    curr_path = os.path.dirname(os.path.abspath(__file__))
    sk_preprocess = SKPreprocess(f"{curr_path}/../config/sk_preprocess.yaml")
    for i in range(10):
        _ = sk_preprocess.get_clustered_semantic_arary()
    print("Done.")
