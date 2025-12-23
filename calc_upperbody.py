import os
import argparse
import open3d as o3d
import numpy as np


def estimate_pelvis_center(
    human_cloud: o3d.geometry.PointCloud, pelvis_ratio: float, band: float
):
    """
    human_cloud : 사람만 남긴 pcd (background 제거 완료된 상태)
    pelvis_ratio : 발바닥에서 pelvis까지 비율 (0~1). 기본 0.55
    band : pelvis 높이 주변으로 잡을 z 두께 (m). 기본 0.10 → ±0.05m
    """
    pts = np.asarray(human_cloud.points)
    if pts.shape[0] == 0:
        raise ValueError("human_cloud에 포인트가 없음")

    # 1) z축 기준 키 추정
    y_min = pts[:, 1].min()
    y_max = pts[:, 1].max()
    # print(y_max, y_min)
    height = y_max - y_min
    # print("height:", height)

    if height <= 0:
        raise ValueError("height 비정상")

    # 2) pelvis 예상 높이
    pelvis_y = y_min + pelvis_ratio * height
    # print("estimated pelvis_z:", pelvis_y)

    # 3) pelvis 주변 band 슬라이스 선택
    half_band = band / 2.0
    mask = (pts[:, 1] >= pelvis_y - half_band) & (pts[:, 1] <= pelvis_y + half_band)
    slice_pts = pts[mask]

    if slice_pts.shape[0] == 0:
        print("pelvis band에 포인트가 없어서 fallback 사용")
        mid_z = (y_min + y_max) / 2.0
        mask2 = (pts[:, 1] >= mid_z - band) & (pts[:, 1] <= mid_z + band)
        slice_pts = pts[mask2]

        if slice_pts.shape[0] == 0:
            raise ValueError("pelvis 근처 포인트가 너무 적음")

    # 4) 슬라이스 포인트 평균 → pelvis center 근사
    center = slice_pts.mean(axis=0)

    # 시각화용 pcd
    slice_cloud = o3d.geometry.PointCloud()
    slice_cloud.points = o3d.utility.Vector3dVector(slice_pts)

    # print("center:", center)
    # print("min:", pts.min(axis=0), "max:", pts.max(axis=0))

    return center, slice_cloud


def estimate_shoulders_simple(
    human_cloud: o3d.geometry.PointCloud,
    pelvis_center: np.ndarray,
    shoulder_ratio: float = 0.82,
    shoulder_width_ratio: float = 0.26,
):
    """
    shoulder_ratio       : 발바닥 기준 어깨 높이 비율 (보통 0.8~0.85)
    shoulder_width_ratio : 키 대비 어깨폭 비율 (보통 0.24~0.28)

    return:
      L_shoulder, R_shoulder : 각각 (3,) numpy array
    """
    pts = np.asarray(human_cloud.points)
    if pts.shape[0] == 0:
        raise ValueError("human_cloud에 포인트가 없음")

    z_min = pts[:, 1].min()
    z_max = pts[:, 1].max()
    height = z_max - z_min
    if height <= 0:
        raise ValueError("height 비정상")

    # 1) 어깨 높이
    shoulder_z = z_min + shoulder_ratio * height
    # print("estimated shoulder_z:", shoulder_z)

    # 2) 어깨 중심 (x, y는 pelvis랑 동일하게 가정, z만 어깨 높이로)
    shoulder_center = np.array([pelvis_center[0], pelvis_center[2], shoulder_z])

    # 3) 어깨 폭 (키의 일정 비율)
    shoulder_width = shoulder_width_ratio * height
    half_width = shoulder_width / 2.0

    # 4) 좌우 방향: y축을 좌우로 가정
    #   왼쪽(+y), 오른쪽(-y)로 배치
    L_shoulder = shoulder_center + np.array([0.0, +half_width, 0.0])
    R_shoulder = shoulder_center + np.array([0.0, -half_width, 0.0])

    return L_shoulder, R_shoulder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--human_pcd_path",
        type=str,
        default="./data/human_only.pcd",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 사람만 계산한 pcd
    human_pcd_path = args.human_pcd_path
    human_cloud = o3d.io.read_point_cloud(human_pcd_path)

    center, slice_cloud = estimate_pelvis_center(
        human_cloud, pelvis_ratio=0.55, band=60
    )  # pelvis: 94/171cm

    # 저장
    save_dir = os.path.dirname(human_pcd_path)
    if save_dir == "":
        save_dir = "."

    filename = os.path.basename(human_pcd_path).split(".")[0]

    # 저장 파일 이름: <원본이름>_pelvis_center.csv
    save_path = os.path.join(save_dir, f"pelvis-{filename}.csv")

    # mm → cm
    center_cm = center / 10.0
    print("pelvis center (cm):", center_cm)

    # CSV로 저장 (콤마 구분)
    np.savetxt(
        save_path,
        center_cm.reshape(1, 3),
        fmt="%.3f",
        delimiter=",",
        header="x_cm,y_cm,z_cm",
        comments="",
    )

    print(f"pelvis CSV로 저장됨: {save_path}")

    # pelvis 구
    pelvis_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=15)
    pelvis_sphere.translate(center)
    pelvis_sphere.paint_uniform_color([1.0, 0.0, 1.0])  # 핑크

    # 라인 지정
    human_cloud.paint_uniform_color([0.7, 0.7, 0.7])

    o3d.visualization.draw_geometries(
        [
            human_cloud,
            # slice_cloud,
            pelvis_sphere,
            # L_sphere,
            # R_sphere,
        ]
    )
