import open3d as o3d
import numpy as np


def estimate_pelvis_center(
    human_cloud: o3d.geometry.PointCloud, pelvis_ratio: float, band: float
):
    """
    human_cloud : 사람만 남긴 pcd (background 제거 완료된 상태)
    pelvis_ratio : 발바닥에서 pelvis까지 비율 (0~1). 기본 0.55
    band : pelvis 높이 주변으로 잡을 z 두께 (m). 기본 0.10 → ±0.05m

    return:
      center : (3,) numpy array, pelvis 추정 좌표 [x, y, z]
      slice_cloud : pelvis 주변 포인트만 모은 pcd (디버깅/시각화용)
    """
    pts = np.asarray(human_cloud.points)
    if pts.shape[0] == 0:
        raise ValueError("human_cloud에 포인트가 없음")

    # z축 기준 키 추정
    z_min = pts[:, 2].min()
    z_max = pts[:, 2].max()
    height = z_max - z_min
    print("height:", height)

    if height <= 0:
        raise ValueError("height 비정상")

    # pelvis 예상 높이
    pelvis_z = z_min + pelvis_ratio * height
    print("estimated pelvis_z:", pelvis_z)

    # pelvis 주변 band 슬라이스 선택
    half_band = band / 2.0
    mask = (pts[:, 2] >= pelvis_z - half_band) & (pts[:, 2] <= pelvis_z + half_band)
    slice_pts = pts[mask]
    # print("slice_pts: ", slice_pts)

    if slice_pts.shape[0] == 0:
        print("pelvis band에 포인트가 없어서 fallback 사용")
        mid_z = (z_min + z_max) / 2.0
        mask2 = (pts[:, 2] >= mid_z - band) & (pts[:, 2] <= mid_z + band)
        slice_pts = pts[mask2]

        if slice_pts.shape[0] == 0:
            raise ValueError("pelvis 근처 포인트가 너무 적음")

    # 슬라이스 포인트 평균 -> pelvis center 근사
    center = slice_pts.mean(axis=0)

    # 시각화용
    slice_cloud = o3d.geometry.PointCloud()
    slice_cloud.points = o3d.utility.Vector3dVector(slice_pts)

    return center, slice_cloud


if __name__ == "__main__":
    human_pcd_path = "./data/human_only.pcd"
    human_cloud = o3d.io.read_point_cloud(human_pcd_path)

    center, slice_cloud = estimate_pelvis_center(
        human_cloud, pelvis_ratio=0.55, band=0.1
    )  # pelvis: 94/171cm
    print("pelvis center:", center)

    pelvis_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
    pelvis_sphere.translate(center)
    pelvis_sphere.paint_uniform_color([1.0, 0.0, 1.0])

    human_cloud.paint_uniform_color([0.7, 0.7, 0.7])

    o3d.visualization.draw_geometries([human_cloud, slice_cloud, pelvis_sphere])
