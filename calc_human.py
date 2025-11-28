import open3d as o3d
import numpy as np
import argparse


def crop_roi(cloud, min_bound, max_bound):
    aabb = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=min_bound,
        max_bound=max_bound
    )
    return cloud.crop(aabb)


def pose_detection_open3d(cloud_A, cloud_B, dist_thresh=0.1):
    kdtree_A = o3d.geometry.KDTreeFlann(cloud_A)

    pts_B = np.asarray(cloud_B.points)
    colors_B = None
    if cloud_B.has_colors():
        colors_B = np.asarray(cloud_B.colors)

    keep_indices = []

    for i, p in enumerate(pts_B):
        k, idx, dist2 = kdtree_A.search_knn_vector_3d(p, 1)
        if k == 0:
            keep_indices.append(i)
        else:
            d = np.sqrt(dist2[0])
            if d > dist_thresh:
                keep_indices.append(i)

    keep_indices = np.array(keep_indices, dtype=int)

    cloud_out = o3d.geometry.PointCloud()
    if len(keep_indices) > 0:
        cloud_out.points = o3d.utility.Vector3dVector(pts_B[keep_indices])
        if colors_B is not None:
            cloud_out.colors = o3d.utility.Vector3dVector(colors_B[keep_indices])
        centerPos = pts_B[keep_indices].mean(axis=0)
    else:
        centerPos = np.array([np.nan, np.nan, np.nan], dtype=np.float32)

    return cloud_out, centerPos


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bg_pcd_path",
        type=str,
        default="./data/pcd_out_1.pcd",
        help="배경"
    )
    parser.add_argument(
        "--human_pcd_path",
        type=str,
        default="./data/pcd_out_110.pcd",
        help="배경+사람"
    )
    parser.add_argument(
        "--out_pcd_path",
        type=str,
        default="./data/human_only.pcd",
        help="결과"
    )
    parser.add_argument(
        "--dist_thresh",
        type=float,
        default=0.15
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    bg_pcd_path = args.bg_pcd_path          # cloud_A: 배경만 있는 pcd
    human_pcd_path = args.human_pcd_path    # cloud_B: 배경 + 사람 pcd
    out_pcd_path = args.out_pcd_path        # 결과 파일

    # pcd 읽기
    cloud_A = o3d.io.read_point_cloud(bg_pcd_path)
    cloud_B = o3d.io.read_point_cloud(human_pcd_path)

    print(f"cloud_A points: {len(cloud_A.points)}")
    print(f"cloud_B points: {len(cloud_B.points)}")

    # 거리 threshold
    dist_thresh = args.dist_thresh

    cloud_out, centerPos = pose_detection_open3d(cloud_A, cloud_B, dist_thresh)

    # 라이다 - 인간 거리 반영 x앞뒤 y위아래 z좌우
    min_bound = np.array([0.0, -1100.0, -500.0])
    max_bound = np.array([3000.0,  500.0,  500.0])

    cloud_out = crop_roi(cloud_out, min_bound, max_bound)

    print(f"detected points: {len(cloud_out.points)}")
    print(f"centerPos: {centerPos}")

    # 결과 저장
    o3d.io.write_point_cloud(out_pcd_path, cloud_out)
    print(f"saved: {out_pcd_path}")

    center_pcd = o3d.geometry.PointCloud()
    center_pcd.points = o3d.utility.Vector3dVector(centerPos.reshape(1, 3))
    center_pcd.paint_uniform_color([1.0, 0.0, 0.0])

    # o3d.visualization.draw_geometries([cloud_out, center_pcd])
