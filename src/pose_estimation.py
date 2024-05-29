# -*- coding:UTF-8 -*-

# %%
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


# %% pose estimation
def transform_two_coordinate_system(
    z_axis_src,
    y_axis_src,
    x_axis_src,
    origin_src,
    origin_2_src,
    z_axis_query,
    y_axis_query,
    x_axis_query,
    origin_query,
    origin_2_query,
):
    tf_que_sph = np.column_stack(
        [x_axis_query, y_axis_query, z_axis_query, (origin_query + origin_2_query) / 2]
    )
    tf_que_sph = np.row_stack([tf_que_sph, [0, 0, 0, 1]])
    tf_src_sph = np.column_stack(
        [x_axis_src, y_axis_src, z_axis_src, (origin_src + origin_2_src) / 2]
    )
    tf_src_sph = np.row_stack([tf_src_sph, [0, 0, 0, 1]])

    # transform from query frame to src frame
    return tf_src_sph @ np.linalg.inv(tf_que_sph)


def icp(pc1, pc2, init_tf=np.eye(4), dist=3, iter=1000):
    icp_result = o3d.pipelines.registration.registration_icp(
        pc1,
        pc2,
        dist,
        init_tf,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iter),
    )
    return icp_result.transformation


def get_copy_without_z(pc):
    arr = np.asarray(pc.points)
    arr[:, 2] = 0
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(arr))


def list2pc(li):
    arr = np.array(li)
    arr = arr[:, :3]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(arr))
    return pcd


def get_tf_yaw_only(tf):
    # get tf with roll and pitch set to 0
    eulers = R.from_matrix(tf[:3, :3]).as_euler("xyz")
    eulers[:2] = 0
    tf_yaw_only = tf.copy()
    tf_yaw_only[:3, :3] = R.from_euler("xyz", eulers).as_matrix()
    return tf_yaw_only


def vertex_icp_2d(src_vertexs, que_vertexs, init_tf):
    src_2d_vertexs = get_copy_without_z(list2pc(src_vertexs))
    que_2d_vertexs = get_copy_without_z(list2pc(que_vertexs))
    init_tf_yaw_only = get_tf_yaw_only(init_tf)
    icp_tf = icp(que_2d_vertexs, src_2d_vertexs, init_tf_yaw_only)
    return icp_tf

