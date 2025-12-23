# visualization_all.py
import argparse
from typing import Optional, Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # 3D axes 활성화용

from endSite import BVH  # 프로젝트의 BVH 파서 사용

# 오른손 근처 시각화에 사용할 반지름 (cm)
HAND_SPHERE_RADIUS = 1.5


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
    'RightElbow', 'RightWrist', 'RightHand' 등 오른팔/손 체인인지 판별.
    (손끝/손목/elbow는 항상 보여주기 위함)
    """
    lower = name.lower()
    # 대표적인 이름들
    if "right" in lower and ("elbow" in lower or "wrist" in lower or "hand" in lower):
        return True
    # 약어 형태 r_elbow, r_hand 등도 대비
    if lower.startswith("r_") and (
        "elbow" in lower or "wrist" in lower or "hand" in lower
    ):
        return True
    return False


def plot_skeleton(
    ax,
    joint_positions: Dict[str, np.ndarray],
    end_positions: Dict[str, np.ndarray],
    bones: List[Tuple[str, Optional[str], bool]],
    color: str,
    linestyle: str = "-",
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
            linestyle=linestyle,
        )
        any_point_plotted = True

    # legend용 dummy 점 (linestyle은 legend에서 표현 안 해도 되므로 scatter 유지)
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


def visualize_all_poses(
    input_bvh_path: str,
    patched_bvh_path: str,
    n1: int,
    n2: int,
    save_path: Optional[str] = None,
    use_midsphere: bool = False,
) -> None:
    """
    총 7개의 스켈레톤을 시각화:

      - start        : input.bvh  2번째 프레임 (빨간 실선)
      - mid 1 edited : patched.bvh n1 프레임 (노란 실선)
      - mid 1 raw    : input.bvh  n1 프레임  (노란 점선)
      - mid 2 edited : patched.bvh n2 프레임 (초록 실선)
      - mid 2 raw    : input.bvh  n2 프레임  (초록 점선)
      - end edited   : patched.bvh 마지막 프레임 (파란 실선)
      - end raw      : input.bvh  마지막 프레임 (파란 점선)

    use_midsphere=True 인 경우:
      - 왼쪽: 전신 스켈레톤 (7개) 표시 (구 X)
      - 오른쪽: 손끝/손목/elbow 중심 확대 시각화 (7개) +
        빨/노/초/파 4개( start + mid1/2 edited + end edited ) 오른손 끝의
        산술 평균 위치에 반지름 1.5cm 구 표시
    """
    # BVH 로드
    bvh_input = load_bvh(input_bvh_path)
    bvh_patched = load_bvh(patched_bvh_path)

    if bvh_patched.frames < 2:
        raise ValueError("patched BVH에는 최소 2프레임 이상이 필요합니다.")

    # 시작 프레임: 기준 프레임 = 2번째 프레임 (index 1)
    start_idx = 1

    # 마지막 프레임 (edited / raw)
    last_idx_edited = bvh_patched.frames - 1
    last_idx_raw = bvh_input.frames - 1

    # --frames n1 n2 (1-based) → 0-based 인덱스로 변환
    idx1 = n1 - 1
    idx2 = n2 - 1

    max_frames = min(bvh_input.frames, bvh_patched.frames)

    for idx, name in [(idx1, "n1"), (idx2, "n2")]:
        if idx < 0 or idx >= max_frames:
            raise ValueError(
                f"{name} 프레임 {idx + 1}가 범위를 벗어났습니다. "
                f"(허용: 1 ~ {max_frames})"
            )

    # 각 포즈의 스켈레톤 추출
    # start (input, frame 2)
    j_start, e_start, bones_start = extract_skeleton(bvh_input, start_idx)

    # mid 1 / mid 2 edited (patched)
    j_mid1_edit, e_mid1_edit, bones_mid1_edit = extract_skeleton(bvh_patched, idx1)
    j_mid2_edit, e_mid2_edit, bones_mid2_edit = extract_skeleton(bvh_patched, idx2)

    # mid 1 / mid 2 raw (input)
    j_mid1_raw, e_mid1_raw, bones_mid1_raw = extract_skeleton(bvh_input, idx1)
    j_mid2_raw, e_mid2_raw, bones_mid2_raw = extract_skeleton(bvh_input, idx2)

    # end edited / end raw
    j_end_edit, e_end_edit, bones_end_edit = extract_skeleton(
        bvh_patched, last_idx_edited
    )
    j_end_raw, e_end_raw, bones_end_raw = extract_skeleton(bvh_input, last_idx_raw)

    # 오른손 끝 End Site (각 포즈마다)
    center_start = get_right_hand_center(e_start)
    center_mid1_edit = get_right_hand_center(e_mid1_edit)
    center_mid1_raw = get_right_hand_center(e_mid1_raw)
    center_mid2_edit = get_right_hand_center(e_mid2_edit)
    center_mid2_raw = get_right_hand_center(e_mid2_raw)
    center_end_edit = get_right_hand_center(e_end_edit)
    center_end_raw = get_right_hand_center(e_end_raw)

    # pose_infos: (label, color, linestyle, joints, ends, bones, center)
    pose_infos = [
        ("Start", "red", "-", j_start, e_start, bones_start, center_start),
        (
            "Mid 1 (edited)",
            "yellow",
            "-",
            j_mid1_edit,
            e_mid1_edit,
            bones_mid1_edit,
            center_mid1_edit,
        ),
        (
            "Mid 1 (raw)",
            "yellow",
            "--",
            j_mid1_raw,
            e_mid1_raw,
            bones_mid1_raw,
            center_mid1_raw,
        ),
        (
            "Mid 2 (edited)",
            "green",
            "-",
            j_mid2_edit,
            e_mid2_edit,
            bones_mid2_edit,
            center_mid2_edit,
        ),
        (
            "Mid 2 (raw)",
            "green",
            "--",
            j_mid2_raw,
            e_mid2_raw,
            bones_mid2_raw,
            center_mid2_raw,
        ),
        (
            "End (edited)",
            "blue",
            "-",
            j_end_edit,
            e_end_edit,
            bones_end_edit,
            center_end_edit,
        ),
        (
            "End (raw)",
            "blue",
            "--",
            j_end_raw,
            e_end_raw,
            bones_end_raw,
            center_end_raw,
        ),
    ]

    # 구(손끝 평균) 중심은 Start + mid1/2 edited + end edited 의 4개만 사용 (기존 규칙 유지)
    centers_for_sphere = [
        c
        for c in (center_start, center_mid1_edit, center_mid2_edit, center_end_edit)
        if c is not None
    ]
    sphere_center: Optional[np.ndarray] = None
    if centers_for_sphere:
        sphere_center = np.mean(np.stack(centers_for_sphere, axis=0), axis=0)

    # ---- 시각화 ----
    if not use_midsphere:
        # 전신 스켈레톤만 하나의 3D 뷰에 그림 (구 없음)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        all_points: List[np.ndarray] = []

        for label, color, linestyle, joints, ends, bones, _center in pose_infos:
            plot_skeleton(
                ax,
                joint_positions=joints,
                end_positions=ends,
                bones=bones,
                color=color,
                linestyle=linestyle,
                label=label,
                focus_center=None,
                use_handsphere=False,
                visible_points_acc=all_points,
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

        # 축 라벨 및 타이틀
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Skeleton visualization (start, mid1/2, end / edited & raw)")

        # 보기 각도(카메라): 위에서 내려다보는 뷰
        ax.view_init(elev=90, azim=-90)
        ax.legend(loc="upper right")

        plt.tight_layout()

    else:
        # 왼쪽: 전신, 오른쪽: 손끝/손목/elbow 주변 확대 + 평균 구
        fig = plt.figure(figsize=(14, 6))
        ax_full = fig.add_subplot(1, 2, 1, projection="3d")
        ax_focus = fig.add_subplot(1, 2, 2, projection="3d")

        # ----- 왼쪽: 전신 스켈레톤 7개 (구 없음) -----
        visible_full: List[np.ndarray] = []
        for label, color, linestyle, joints, ends, bones, _center in pose_infos:
            plot_skeleton(
                ax_full,
                joint_positions=joints,
                end_positions=ends,
                bones=bones,
                color=color,
                linestyle=linestyle,
                label=label,
                focus_center=None,
                use_handsphere=False,
                visible_points_acc=visible_full,
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

        ax_full.set_xlabel("X")
        ax_full.set_ylabel("Y")
        ax_full.set_zlabel("Z")
        ax_full.set_title("Full body (start, mid1/2, end / edited & raw)")
        ax_full.view_init(elev=90, azim=-90)
        ax_full.legend(loc="upper right")

        # ----- 오른쪽: 손끝/손목/elbow 주변 확대 + 평균 구 -----
        visible_focus: List[np.ndarray] = []
        for label, color, linestyle, joints, ends, bones, center in pose_infos:
            plot_skeleton(
                ax_focus,
                joint_positions=joints,
                end_positions=ends,
                bones=bones,
                color=color,
                linestyle=linestyle,
                label=label,
                focus_center=center,
                use_handsphere=True,
                visible_points_acc=visible_focus,
            )

        # 평균 손끝 위치에 구 그리기 (오른쪽 뷰에만)
        if sphere_center is not None:
            draw_hand_sphere(
                ax_focus, sphere_center, radius=HAND_SPHERE_RADIUS, color="k"
            )

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
        ax_focus.set_title("Right hand zoom (start, mid1/2, end / edited & raw)")
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
        description="input.bvh / input_patched_all.bvh에서 "
        "start/mid1/mid2/end (raw+edited) 7개 포즈 스켈레톤 비교 시각화"
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
        help="결과 이미지를 저장할 경로 (예: ./skeleton_compare_all.png). 지정 안 하면 화면만 표시.",
    )
    parser.add_argument(
        "--midsphere",
        action="store_true",
        help=(
            "전신 + 손끝(손끝/손목/elbow) 확대 시각화 모드 활성화. "
            "오른손 끝 4개(빨/노/초/파; start + mid1/2 edited + end edited)의 "
            "산술 평균 위치에 반지름 1.5cm 구를 표시."
        ),
    )

    args = parser.parse_args()
    n1, n2 = args.frames

    visualize_all_poses(
        input_bvh_path=args.input_bvh,
        patched_bvh_path=args.patched_bvh,
        n1=n1,
        n2=n2,
        save_path=args.save,
        use_midsphere=args.midsphere,
    )


if __name__ == "__main__":
    main()
