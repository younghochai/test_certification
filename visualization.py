import argparse
from typing import Optional, Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # 3D axes 활성화용

from endSite import BVH  # 프로젝트의 BVH 파서 사용

# 오른손 근처 시각화에 사용할 반지름 (cm)
HAND_SPHERE_RADIUS = 1.5

plt.rcParams.update({
    "font.size": 14,        # 기본 폰트 크기
    "axes.labelsize": 16,   # X, Y, Z 라벨
    "axes.titlesize": 18,   # 타이틀
    "legend.fontsize": 14,  # 범례 글자
})


def load_bvh(path: str) -> BVH:
    """BVH 파일을 읽어서 BVH 객체로 반환."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return BVH(text)


def extract_skeleton(bvh: BVH, frame_idx: int):
    """
    주어진 BVH에서 특정 frame_idx의 스켈레톤 정보를 계산.

    반환:
        joint_positions: {joint_name: np.array([x, y, z])}
        end_positions:   {parent_joint_name: np.array([x, y, z])}  # End Site 좌표
        bones:           [(parent_name, child_name, is_end_site), ...]
                         child_name는 End Site인 경우 None
    """
    if frame_idx < 0 or frame_idx >= bvh.frames:
        raise IndexError(
            f"frame_idx {frame_idx}가 범위를 벗어났습니다 (0 ~ {bvh.frames - 1})"
        )

    frame_vals = bvh.motion[frame_idx]

    joint_positions: Dict[str, np.ndarray] = {}
    end_positions: Dict[str, np.ndarray] = {}

    # 1) 각 조인트의 월드 좌표와 End Site 좌표 계산
    for name, joint in bvh.joints_by_name.items():
        # 조인트 원점 좌표
        M = bvh.world_matrix(joint, frame_vals)
        origin = M @ np.array([0.0, 0.0, 0.0, 1.0])
        joint_positions[name] = origin[:3]

        # End Site가 있으면 그 좌표도 계산
        if joint.end_site_offset is not None:
            end_h = M @ np.append(joint.end_site_offset, 1.0)
            end_positions[name] = end_h[:3]

    # 2) 뼈대(선분) 정보 생성:
    #    (부모 조인트 -> 자식 조인트), (조인트 -> End Site)
    bones: List[Tuple[str, Optional[str], bool]] = []
    for name, joint in bvh.joints_by_name.items():
        # 자식 조인트들과 연결
        for child in joint.children:
            bones.append((name, child.name, False))

        # End Site와 연결
        if joint.end_site_offset is not None:
            # child_name을 None으로 두고, parent_name의 End Site라고 표시
            bones.append((name, None, True))

    return joint_positions, end_positions, bones


def set_axes_equal(ax):
    """
    3D Axes에서 x, y, z 축 스케일을 동일하게 맞추기 위한 헬퍼 함수.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    x_middle = float(np.mean(x_limits))
    y_range = y_limits[1] - y_limits[0]
    y_middle = float(np.mean(y_limits))
    z_range = z_limits[1] - z_limits[0]
    z_middle = float(np.mean(z_limits))

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def is_right_arm_chain(name: str) -> bool:
    """
    오른팔/손 체인인지 판별.
    (손끝/손목 항상 보여주기 위함)
    """
    lower = name.lower()
    # 대표적인 이름들
    if "right" in lower and ("wrist" in lower or "hand" in lower):
        return True
    # 약어 형태 r_elbow, r_hand 등도 대비
    if lower.startswith("r_") and ("wrist" in lower or "hand" in lower):
        return True
    return False


def plot_skeleton(
    ax,
    joint_positions: Dict[str, np.ndarray],
    end_positions: Dict[str, np.ndarray],
    bones: List[Tuple[str, Optional[str], bool]],
    color: str,
    label: Optional[str] = None,
    focus_center: Optional[np.ndarray] = None,
    use_handsphere: bool = False,
    visible_points_acc: Optional[List[np.ndarray]] = None,
):
    """
    하나의 스켈레톤(프레임)을 3D Axes에 그리는 함수.

    use_handsphere=True이면:
      - focus_center(오른손 끝) 주변 HAND_SPHERE_RADIUS 이내의 점만 시각화.
      - 단, 오른팔 체인(손끝/손목/elbow)은 거리와 상관없이 항상 시각화.
    """

    def is_visible_point(name: str, pos: np.ndarray) -> bool:
        if not use_handsphere:
            return True
        if focus_center is None:
            # 중심이 없으면 필터링 불가 → 일단 전부 그림
            return True
        # 오른팔 체인(Hand/Wrist/Elbow)은 항상 보이게
        if is_right_arm_chain(name):
            return True
        # 그 외 관절은 오른손 끝에서 일정 반지름 이내인 경우만 보이게
        dist = float(np.linalg.norm(pos - focus_center))
        return dist <= HAND_SPHERE_RADIUS

    any_point_plotted = False

    # 1) 조인트 점 찍기
    for name, p in joint_positions.items():
        if is_visible_point(name, p):
            ax.scatter(p[0], p[1], p[2], color=color, s=8)
            any_point_plotted = True
            if visible_points_acc is not None:
                visible_points_acc.append(p)

    # 2) End Site 점 찍기 (키는 부모 조인트 이름)
    for parent_name, p in end_positions.items():
        if is_visible_point(parent_name, p):
            ax.scatter(p[0], p[1], p[2], color=color, s=15)
            any_point_plotted = True
            if visible_points_acc is not None:
                visible_points_acc.append(p)

    # 3) 뼈대(선분) 그리기
    for parent_name, child_name, is_end in bones:
        if parent_name not in joint_positions:
            continue

        p1 = joint_positions[parent_name]

        if is_end:
            # parent -> End Site
            if parent_name not in end_positions:
                continue
            p2 = end_positions[parent_name]
            name2 = parent_name  # End Site도 같은 조인트 이름 기준으로 visibility 판정
        else:
            if child_name is None or child_name not in joint_positions:
                continue
            p2 = joint_positions[child_name]
            name2 = child_name

        vis1 = is_visible_point(parent_name, p1)
        vis2 = is_visible_point(name2, p2)

        # 두 끝점 모두 보이는 경우에만 선을 그림
        if not (vis1 and vis2):
            continue

        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color=color,
            linewidth=1.0,
        )
        any_point_plotted = True

    # legend용 dummy 점
    if any_point_plotted and label is not None:
        ax.scatter([], [], [], color=color, label=label)


def get_right_hand_center(end_positions: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """
    End Site 딕셔너리에서 오른손 끝(오른손 End Site)의 좌표를 찾아 반환.
    우선순위: RightWrist > RightHand > (그 외 heuristic)
    """
    for key in ("RightWrist", "RightHand"):
        if key in end_positions:
            return end_positions[key]

    # 이름이 다를 수 있으니 heuristic으로 한 번 더 시도
    for name, pos in end_positions.items():
        lower = name.lower()
        if "right" in lower and ("hand" in lower or "wrist" in lower):
            return pos

    return None


def draw_hand_sphere(
    ax,
    center: np.ndarray,
    radius: float,
    color: str = "k",
):
    """
    center를 중심으로 반지름 radius인 구(sphere)를 하나 그림.
    """
    u = np.linspace(0.0, 2.0 * np.pi, 64)
    v = np.linspace(0.0, np.pi, 32)
    uu, vv = np.meshgrid(u, v)

    xs = center[0] + radius * np.sin(vv) * np.cos(uu)
    ys = center[1] + radius * np.sin(vv) * np.sin(uu)
    zs = center[2] + radius * np.cos(vv)

    ax.plot_surface(xs, ys, zs, color=color, alpha=0.25, linewidth=0.2)


# ---------------------- 새로 추가된 헬퍼 함수들 ---------------------- #
def collect_right_arm_keypoints(
    joint_positions: Dict[str, np.ndarray],
    end_positions: Dict[str, np.ndarray],
) -> List[np.ndarray]:
    """
    오른손끝(End Site), 오른손목, 오른팔꿈치에 해당하는 점들을 모은다.
    - 이름은 heuristic 으로 판별 (RightHand / RightWrist / RightElbow 등)
    """
    pts: List[np.ndarray] = []

    # 1) 손끝(End Site)
    hand_center = get_right_hand_center(end_positions)
    if hand_center is not None:
        pts.append(hand_center)

    # 2) 손목/팔꿈치 조인트
    for name, p in joint_positions.items():
        lower = name.lower()
        if "right" not in lower:
            continue
        if ("wrist" in lower) or ("hand" in lower):
            pts.append(p)

    return pts


def draw_2d_circle_on_3d(
    ax,
    center_xy: np.ndarray,
    radius: float,
    z_level: float,
    color: str = "magenta",
    linewidth: float = 2.5,
    linestyle: str = "-",
):
    """
    3D Axes 위에 XY 평면 상의 원을 하나 그린다.
    - 카메라를 (elev=90, azim=...) 로 두면 화면에서는 '동그라미'로 보인다.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    xs = center_xy[0] + radius * np.cos(theta)
    ys = center_xy[1] + radius * np.sin(theta)
    zs = np.full_like(xs, z_level)

    ax.plot(xs, ys, zs, color=color, linewidth=linewidth, linestyle=linestyle)


# ------------------------------------------------------------------- #


def visualize_four_poses(
    input_bvh_path: str,
    patched_bvh_path: str,
    n1: int,
    n2: int,
    save_path: Optional[str] = None,
    use_midsphere: bool = False,
) -> None:
    """
    총 4개의 스켈레톤을 시각화:

      1) input.bvh  2번째 프레임 (빨강)  → 시작 프레임
      2) patched.bvh n1 프레임 (노랑)   → 중간 프레임 1
      3) patched.bvh n2 프레임 (초록)   → 중간 프레임 2
      4) patched.bvh 마지막 프레임 (파랑) → 마지막 프레임

    ※ end raw (input.bvh의 마지막 프레임)는 시각화하지 않음.

    use_midsphere=True 인 경우:
      - 왼쪽: 전신 스켈레톤 (4개)만 표시 + 오른손끝/손목/팔꿈치 영역을 하나의 동그라미로 강조
      - 오른쪽: 손끝/손목/elbow 중심 확대 시각화 +
        빨/노/초/파 4개 오른손 끝의 산술 평균 위치에 반지름 1.5cm 구 표시
    """
    # BVH 로드
    bvh_input = load_bvh(input_bvh_path)
    bvh_patched = load_bvh(patched_bvh_path)

    if bvh_patched.frames < 2:
        raise ValueError("patched BVH에는 최소 2프레임 이상이 필요합니다.")

    # 시작 프레임: 기준 프레임 = 2번째 프레임 (index 1)
    start_idx = 1
    # 마지막 프레임: patched BVH 마지막
    last_idx = bvh_patched.frames - 1

    # --frames n1 n2 (1-based) → 0-based 인덱스로 변환
    idx1 = n1 - 1
    idx2 = n2 - 1

    for idx, name in [(idx1, "n1"), (idx2, "n2")]:
        if idx < 0 or idx >= bvh_patched.frames:
            raise ValueError(
                f"{name} 프레임 {idx + 1}가 범위를 벗어났습니다. "
                f"(허용: 1 ~ {bvh_patched.frames})"
            )

    # 각 포즈의 스켈레톤 추출
    j_start, e_start, bones_start = extract_skeleton(bvh_input, start_idx)
    j_mid1, e_mid1, bones_mid1 = extract_skeleton(bvh_patched, idx1)
    j_mid2, e_mid2, bones_mid2 = extract_skeleton(bvh_patched, idx2)
    j_last, e_last, bones_last = extract_skeleton(bvh_patched, last_idx)

    # 오른손 끝 End Site (각 포즈마다)
    center_start = get_right_hand_center(e_start)
    center_mid1 = get_right_hand_center(e_mid1)
    center_mid2 = get_right_hand_center(e_mid2)
    center_last = get_right_hand_center(e_last)

    pose_infos = [
        ("Start", "red",    j_start, e_start, bones_start, center_start),
        ("Return 1", "black", j_mid1,  e_mid1,  bones_mid1,  center_mid1),
        ("Return 2", "green",  j_mid2,  e_mid2,  bones_mid2,  center_mid2),
        ("Return 3", "blue",   j_last,  e_last,  bones_last,  center_last),
    ]

    # 네 손끝의 평균 중심 (None 아닌 것만 사용)
    centers = [c for c in (center_start, center_mid1, center_mid2, center_last) if c is not None]
    sphere_center: Optional[np.ndarray] = None
    if centers:
        sphere_center = np.mean(np.stack(centers, axis=0), axis=0)

    # ---- 시각화 ----
    if not use_midsphere:
        # 전신 스켈레톤만 하나의 3D 뷰에 그림 (구 없음)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        all_points: List[np.ndarray] = []
        right_arm_points_all: List[np.ndarray] = []

        for label, color, joints, ends, bones, _center in pose_infos:
            plot_skeleton(
                ax,
                joint_positions=joints,
                end_positions=ends,
                bones=bones,
                color=color,
                label=label,
                focus_center=None,
                use_handsphere=False,
                visible_points_acc=all_points,
            )
            # 오른팔 주요 포인트 수집 (손끝/손목/elbow)
            right_arm_points_all.extend(
                collect_right_arm_keypoints(joints, ends)
            )

        # 축 범위 설정
        if all_points:
            pts = np.vstack(all_points)
            min_xyz = pts.min(axis=0)
            max_xyz = pts.max(axis=0)

            margin = HAND_SPHERE_RADIUS * 2.0

            ax.set_xlim(min_xyz[0] - margin, max_xyz[0] + margin)
            ax.set_ylim(min_xyz[1] - margin, max_xyz[1] + margin)
            ax.set_zlim(min_xyz[2] - margin, max_xyz[2] + margin)
            set_axes_equal(ax)

        # ----- 오른손끝/손목/팔꿈치 영역 동그라미로 강조 (전체 뷰에서도 표시) -----
        if right_arm_points_all:
            pts_ra = np.vstack(right_arm_points_all)
            # XY 평면 상의 중심 및 반지름 계산
            center_xy = pts_ra[:, :2].mean(axis=0)
            dists = np.linalg.norm(pts_ra[:, :2] - center_xy[None, :], axis=1)
            radius = float(dists.max()) * 1.1  # 약간 margin
            z_level = float(pts_ra[:, 2].mean())
            draw_2d_circle_on_3d(
                ax,
                center_xy=center_xy,
                radius=radius,
                z_level=z_level,
                color="magenta",
                linewidth=2.5,
                linestyle="-",
            )

        # 축 라벨 및 타이틀
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Skeleton visualization (start, return 1 2 3)")

        # 보기 각도(카메라): 위에서 내려다보는 뷰
        ax.view_init(elev=90, azim=-90)

        # 범례
        ax.legend(loc="upper right")
        plt.tight_layout()

    else:
        # 왼쪽: 전신 / 오른쪽: 손끝 확대 + 평균 중심 구
        fig = plt.figure(figsize=(14, 6))
        ax_full = fig.add_subplot(1, 2, 1, projection="3d")
        ax_focus = fig.add_subplot(1, 2, 2, projection="3d")

        # ----- 왼쪽: 전신 스켈레톤 4개 (구 없음) -----
        visible_full: List[np.ndarray] = []
        right_arm_points_full: List[np.ndarray] = []

        for label, color, joints, ends, bones, _center in pose_infos:
            plot_skeleton(
                ax_full,
                joint_positions=joints,
                end_positions=ends,
                bones=bones,
                color=color,
                label=label,
                focus_center=None,
                use_handsphere=False,
                visible_points_acc=visible_full,
            )
            # 오른팔 주요 포인트(손끝/손목/elbow) 수집
            right_arm_points_full.extend(
                collect_right_arm_keypoints(joints, ends)
            )

        if visible_full:
            pts = np.vstack(visible_full)
            min_xyz = pts.min(axis=0)
            max_xyz = pts.max(axis=0)

            margin = HAND_SPHERE_RADIUS * 2.0

            ax_full.set_xlim(min_xyz[0] - margin, max_xyz[0] + margin)
            ax_full.set_ylim(min_xyz[1] - margin, max_xyz[1] + margin)
            ax_full.set_zlim(min_xyz[2] - margin, max_xyz[2] + margin)
            set_axes_equal(ax_full)

        # ----- 왼쪽: 오른손끝/손목/팔꿈치 영역 동그라미로 강조 -----
        if right_arm_points_full:
            pts_ra = np.vstack(right_arm_points_full)
            # XY 평면 상 중심/반지름 계산
            center_xy = pts_ra[:, :2].mean(axis=0)
            dists = np.linalg.norm(pts_ra[:, :2] - center_xy[None, :], axis=1)
            radius = float(dists.max()) * 1.1  # 약간 margin
            z_level = float(pts_ra[:, 2].mean())
            draw_2d_circle_on_3d(
                ax_full,
                center_xy=center_xy,
                radius=radius,
                z_level=z_level,
                color="magenta",
                linewidth=2.5,
                linestyle="-",
            )

        ax_full.set_xlabel("X")
        ax_full.set_ylabel("Y")
        ax_full.set_zlabel("Z")
        ax_full.set_title("Full body (start, return 1 2 3)")
        ax_full.view_init(elev=90, azim=-90)
        ax_full.legend(loc="upper right")

        # ----- 오른쪽: 손끝/손목/elbow 주변 확대 + 평균 구 -----
        visible_focus: List[np.ndarray] = []
        for label, color, joints, ends, bones, center in pose_infos:
            plot_skeleton(
                ax_focus,
                joint_positions=joints,
                end_positions=ends,
                bones=bones,
                color=color,
                label=label,
                focus_center=center,
                use_handsphere=True,
                visible_points_acc=visible_focus,
            )

        # 평균 손끝 위치에 구 그리기 (오른쪽 뷰에만) + 구 지름(3cm) 텍스트
        if sphere_center is not None:
            draw_hand_sphere(ax_focus, sphere_center, radius=HAND_SPHERE_RADIUS, color="k")

            # 구 지름(2 * radius) = 3cm 텍스트 표시
            diameter = HAND_SPHERE_RADIUS * 2.0  # 1.5cm * 2 = 3cm
            # 구 중심에서 약간 옆/위로 치우친 위치에 텍스트 표시
            offset = np.array([
                HAND_SPHERE_RADIUS * 1.3,
                0.0,
                HAND_SPHERE_RADIUS * 1.3,
            ])
            pos = sphere_center + offset
            ax_focus.text(
                pos[0],
                pos[1],
                pos[2],
                f"{diameter:.1f} cm",
                color="red",
                fontsize=16,
                ha="center",
                va="center",
            )

        # 오른손 끝 사이 거리 텍스트 표시
        # 기준: Start(빨강) 손끝
        if center_start is not None:
            def _draw_distance_text(p_from: np.ndarray,
                                    p_to: Optional[np.ndarray],
                                    color_text: str):
                if p_to is None:
                    return
                dist = float(np.linalg.norm(p_to - p_from))
                mid = (p_from + p_to) / 2.0
                mid = mid.copy()

                # --- 색깔(라인 종류)에 따라 서로 다른 offset 적용 ---
                if color_text == "black":
                    offset = np.array([
                        HAND_SPHERE_RADIUS * 0.0,   # x
                        HAND_SPHERE_RADIUS * 0.8,   # y
                        HAND_SPHERE_RADIUS * 0.3,   # z
                    ])
                elif color_text == "green":
                    offset = np.array([
                        HAND_SPHERE_RADIUS * -0.8,  # x
                        HAND_SPHERE_RADIUS * -0.2,  # y
                        HAND_SPHERE_RADIUS * 0.3,   # z
                    ])
                elif color_text == "blue":
                    offset = np.array([
                        HAND_SPHERE_RADIUS * 0.8,   # x
                        HAND_SPHERE_RADIUS * -0.2,  # y
                        HAND_SPHERE_RADIUS * 0.3,   # z
                    ])
                else:
                    # 기본값 (혹시 다른 색 들어올 때)
                    offset = np.array([
                        0.0,
                        0.0,
                        HAND_SPHERE_RADIUS * 0.3,
                    ])

                pos = mid + offset

                ax_focus.text(
                    pos[0],
                    pos[1],
                    pos[2],
                    f"{dist:.2f} cm",
                    color=color_text,
                    fontsize=16,
                    ha="center",
                    va="bottom",
                )

            # 빨간 손끝 ~ 노랑 손끝 = 노랑 글씨
            _draw_distance_text(center_start, center_mid1, "black")

            # 빨간 손끝 ~ 초록 손끝 = 초록 글씨
            _draw_distance_text(center_start, center_mid2, "green")

            # 빨간 손끝 ~ 파랑 손끝 = 파랑 글씨
            _draw_distance_text(center_start, center_last, "blue")

        if visible_focus:
            pts = np.vstack(visible_focus)
            min_xyz = pts.min(axis=0)
            max_xyz = pts.max(axis=0)

            margin = HAND_SPHERE_RADIUS * 2.0

            ax_focus.set_xlim(min_xyz[0] - margin, max_xyz[0] + margin)
            ax_focus.set_ylim(min_xyz[1] - margin, max_xyz[1] + margin)
            ax_focus.set_zlim(min_xyz[2] - margin, max_xyz[2] + margin)
            set_axes_equal(ax_focus)

        ax_focus.set_xlabel("X")
        ax_focus.set_ylabel("Y")
        ax_focus.set_zlabel("Z")
        ax_focus.set_title("Right hand zoom (start, return 1 2 3)")
        ax_focus.view_init(elev=90, azim=-90)
        ax_focus.legend(loc="upper right")

        plt.tight_layout()

    # 저장 & 표시
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"[정보] 그림 저장 완료: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="input.bvh / input_patched.bvh 4개 프레임 스켈레톤 비교 시각화"
    )
    parser.add_argument(
        "--input_bvh",
        type=str,
        default="./data/BVH/input.bvh",
        help="원본 BVH 경로 (기본: ./data/BVH/input.bvh)",
    )
    parser.add_argument(
        "--patched_bvh",
        type=str,
        default="./data/BVH/input_patched_all.bvh",
        help="패치된 BVH 경로 (기본: ./data/BVH/input_patched_all.bvh)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs=2,
        metavar=("N1", "N2"),
        required=True,
        help="중간 프레임 번호 (1-based, 예: --frames 30 60)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="결과 이미지를 저장할 경로 (예: ./skeleton_compare.png). 지정 안 하면 화면만 표시.",
    )
    parser.add_argument(
        "--midsphere",
        action="store_true",
        help=(
            "전신 + 손끝(손끝/손목/elbow) 확대 시각화 모드 활성화. "
            "오른손 끝 4개(빨/노/초/파)의 산술 평균 위치에 반지름 1.5cm 구를 표시."
        ),
    )

    args = parser.parse_args()
    n1, n2 = args.frames

    visualize_four_poses(
        input_bvh_path=args.input_bvh,
        patched_bvh_path=args.patched_bvh,
        n1=n1,
        n2=n2,
        save_path=args.save,
        use_midsphere=args.midsphere,
    )


if __name__ == "__main__":
    main()
