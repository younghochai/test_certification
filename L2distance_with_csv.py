import sys
import os
import math
import numpy as np


# =========================
# BVH → CSV 변환용 코드
#   - pelvis, L/R shoulder, L/R hand End Site 월드 좌표를
#     <input_basename>.csv 로 저장
#   - 기존 endSite.py 의 핵심 로직을 축약/통합
# =========================

def Rx(deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array(
        [[1, 0, 0, 0],
         [0, c, -s, 0],
         [0, s,  c, 0],
         [0, 0, 0, 1]],
        dtype=float
    )


def Ry(deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array(
        [[ c, 0, s, 0],
         [ 0, 1, 0, 0],
         [-s, 0, c, 0],
         [ 0, 0, 0, 1]],
        dtype=float
    )


def Rz(deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array(
        [[c, -s, 0, 0],
         [s,  c, 0, 0],
         [0,  0, 1, 0],
         [0,  0, 0, 1]],
        dtype=float
    )


def T(tx, ty, tz):
    M = np.eye(4, dtype=float)
    M[:3, 3] = [tx, ty, tz]
    return M


class Joint:
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent
        self.children = []
        self.offset = np.zeros(3, dtype=float)
        self.channels = []        # ["Xposition", "Yposition", ...]
        self.chan_start = -1
        self.end_site_offset = None

    def __repr__(self):
        return f"Joint({self.name}, ch={len(self.channels)})"


class BVH:
    def __init__(self, text):
        self.text = text
        self.root = None
        self.joints_by_name = {}
        self.frames = 0
        self.frame_time = 0.0
        self.motion = []          # List[frame][channel]
        self.total_channels = 0
        self._parse()

    def _parse(self):
        lines = [ln.strip() for ln in self.text.splitlines() if ln.strip() != ""]
        if not lines or not lines[0].upper().startswith("HIERARCHY"):
            raise ValueError("BVH 포맷이 아니거나 HIERARCHY 헤더가 없습니다.")
        i = 1
        channel_cursor = 0

        def parse_joint_block(j, idx):
            nonlocal channel_cursor
            if lines[idx] != "{":
                raise ValueError(f"{j.name} 블록에서 '{{' 를 기대했으나: {lines[idx]}")
            idx += 1
            while idx < len(lines):
                ln = lines[idx]
                if ln.startswith("OFFSET"):
                    nums = list(map(float, ln.split()[1:4]))
                    j.offset = np.array(nums, dtype=float)
                    idx += 1
                elif ln.startswith("CHANNELS"):
                    parts = ln.split()
                    n = int(parts[1])
                    j.channels = parts[2:2 + n]
                    j.chan_start = channel_cursor
                    channel_cursor += n
                    idx += 1
                elif ln.startswith("JOINT"):
                    name = ln.split()[1]
                    child = Joint(name, j)
                    j.children.append(child)
                    self.joints_by_name[name] = child
                    idx += 1
                    idx = parse_joint_block(child, idx)
                elif ln.startswith("End Site"):
                    # End Site 블록
                    idx += 1
                    if lines[idx] != "{":
                        raise ValueError("End Site 뒤에 '{' 필요")
                    idx += 1
                    if not lines[idx].startswith("OFFSET"):
                        raise ValueError("End Site 블록에 OFFSET 필요")
                    nums = list(map(float, lines[idx].split()[1:4]))
                    j.end_site_offset = np.array(nums, dtype=float)
                    idx += 1
                    if lines[idx] != "}":
                        raise ValueError("End Site 블록 종료 '}' 필요")
                    idx += 1
                elif ln == "}":
                    idx += 1
                    return idx
                else:
                    idx += 1
            return idx

        # ROOT
        if not lines[i].startswith("ROOT"):
            raise ValueError("HIERARCHY 다음에 ROOT 가 나와야 합니다.")
        root_name = lines[i].split()[1]
        self.root = Joint(root_name, None)
        self.joints_by_name[root_name] = self.root
        i += 1
        i = parse_joint_block(self.root, i)

        # MOTION 섹션
        if not lines[i].startswith("MOTION"):
            raise ValueError("MOTION 섹션을 찾을 수 없습니다.")
        i += 1
        if not lines[i].startswith("Frames:"):
            raise ValueError("Frames: 라인을 찾을 수 없습니다.")
        self.frames = int(lines[i].split()[1])
        i += 1
        if not lines[i].startswith("Frame Time:"):
            raise ValueError("Frame Time: 라인을 찾을 수 없습니다.")
        self.frame_time = float(lines[i].split()[2])
        i += 1

        self.total_channels = channel_cursor

        # 모션 데이터
        for f in range(self.frames):
            if i + f >= len(lines):
                raise ValueError(f"프레임 {f} 데이터를 읽는 중 파일이 끝났습니다.")
            vals = list(map(float, lines[i + f].split()))
            if len(vals) != self.total_channels:
                raise ValueError(
                    f"Frame {f} 채널 수 불일치: {len(vals)}개, 기대값 {self.total_channels}"
                )
            self.motion.append(vals)

    def _local_tr(self, j, frame_vals):
        tx = ty = tz = 0.0
        R = np.eye(4, dtype=float)
        if j.channels:
            base = j.chan_start
            for k, ch in enumerate(j.channels):
                v = frame_vals[base + k]
                if ch == "Xposition":
                    tx = v
                elif ch == "Yposition":
                    ty = v
                elif ch == "Zposition":
                    tz = v
                elif ch == "Xrotation":
                    R = Rx(v) @ R
                elif ch == "Yrotation":
                    R = Ry(v) @ R
                elif ch == "Zrotation":
                    R = Rz(v) @ R
                else:
                    raise ValueError(f"알 수 없는 채널: {ch}")
        return T(tx, ty, tz), R

    def world_matrix(self, j, frame_vals):
        if j.parent is None:
            Tloc, Rloc = self._local_tr(j, frame_vals)
            return Tloc @ Rloc
        else:
            M_parent = self.world_matrix(j.parent, frame_vals)
            Tloc, Rloc = self._local_tr(j, frame_vals)
            # parent @ T(OFFSET) @ R(ch) @ T(pos)
            return M_parent @ T(*j.offset.tolist()) @ Rloc @ Tloc

    def _joint_world_positions(self, j):
        pts = np.zeros((self.frames, 3), dtype=float)
        origin = np.array([0.0, 0.0, 0.0, 1.0])
        for f in range(self.frames):
            M = self.world_matrix(j, self.motion[f])
            p = M @ origin
            pts[f, :] = p[:3]
        return pts

    def right_hand_end_positions(self):
        rw = self.joints_by_name.get("RightWrist")
        if rw is None or rw.end_site_offset is None:
            raise KeyError("RightWrist 또는 End Site 를 찾을 수 없습니다.")
        pts = np.zeros((self.frames, 3), dtype=float)
        end_local = np.append(rw.end_site_offset, 1.0)
        for f in range(self.frames):
            Mw = self.world_matrix(rw, self.motion[f])
            p = Mw @ end_local
            pts[f, :] = p[:3]
        return pts

    def left_hand_end_positions(self):
        lw = self.joints_by_name.get("LeftWrist")
        if lw is None or lw.end_site_offset is None:
            raise KeyError("LeftWrist 또는 End Site 를 찾을 수 없습니다.")
        pts = np.zeros((self.frames, 3), dtype=float)
        end_local = np.append(lw.end_site_offset, 1.0)
        for f in range(self.frames):
            Mw = self.world_matrix(lw, self.motion[f])
            p = Mw @ end_local
            pts[f, :] = p[:3]
        return pts

    def pelvis_positions(self):
        if self.root is None:
            raise RuntimeError("루트 조인트가 없습니다.")

        root = self.root
        pts = np.zeros((self.frames, 3), dtype=float)

        if not root.channels:
            raise RuntimeError("루트에 채널 정보가 없습니다.")

        base = root.chan_start
        for f in range(self.frames):
            vals = self.motion[f]
            tx = ty = tz = 0.0
            for k, ch in enumerate(root.channels):
                v = vals[base + k]
                if ch == "Xposition":
                    tx = v
                elif ch == "Yposition":
                    ty = v
                elif ch == "Zposition":
                    tz = v
            pts[f, :] = [tx, ty, tz]
        return pts

    def right_shoulder_positions(self):
        rs = self.joints_by_name.get("RightShoulder")
        if rs is None:
            raise KeyError("RightShoulder 조인트를 찾을 수 없습니다.")
        return self._joint_world_positions(rs)

    def left_shoulder_positions(self):
        ls = self.joints_by_name.get("LeftShoulder")
        if ls is None:
            raise KeyError("LeftShoulder 조인트를 찾을 수 없습니다.")
        return self._joint_world_positions(ls)


def bvh_to_csv(bvh_path: str) -> str:
    """
    BVH 파일 하나를 받아서
      pelvis(3), L_shoulder(3), R_shoulder(3), L_hand(3), R_hand(3)
    순서로 CSV를 생성하고, CSV 경로를 리턴한다.

    예:
      input.bvh -> input.csv
      input_patched.bvh -> input_patched.csv
    """
    if not os.path.isfile(bvh_path):
        raise FileNotFoundError(f"BVH 파일을 찾을 수 없습니다: {bvh_path}")

    with open(bvh_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    bvh = BVH(text)

    right_end_xyz = bvh.right_hand_end_positions()
    left_end_xyz = bvh.left_hand_end_positions()
    pelvis_xyz = bvh.pelvis_positions()
    right_shoulder_xyz = bvh.right_shoulder_positions()
    left_shoulder_xyz = bvh.left_shoulder_positions()

    if not (pelvis_xyz.shape[0] == left_end_xyz.shape[0] == right_end_xyz.shape[0] ==
            left_shoulder_xyz.shape[0] == right_shoulder_xyz.shape[0]):
        raise ValueError("Pelvis / 손 / 어깨 프레임 수가 서로 다릅니다.")

    merged = np.hstack(
        (
            pelvis_xyz,
            left_shoulder_xyz,
            right_shoulder_xyz,
            left_end_xyz,
            right_end_xyz,
        )
    )

    base_name, _ = os.path.splitext(os.path.basename(bvh_path))
    out_dir = os.path.dirname(os.path.abspath(bvh_path))
    out_csv_path = os.path.join(out_dir, base_name + ".csv")

    header = (
        "pelvis_x,pelvis_y,pelvis_z,"
        "l_shoulder_x,l_shoulder_y,l_shoulder_z,"
        "r_shoulder_x,r_shoulder_y,r_shoulder_z,"
        "l_hand_x,l_hand_y,l_hand_z,"
        "r_hand_x,r_hand_y,r_hand_z"
    )

    np.savetxt(
        out_csv_path,
        merged,
        delimiter=",",
        header=header,
        comments="",
    )

    print(f"[BVH→CSV] 저장 완료: {out_csv_path}")
    return out_csv_path


# =========================
# r_hand L2 distance 계산
# =========================

def compare_r_hand(csv_path: str, frames=None):
    """
    csv_path : BVH에서 추출한 CSV (header 1줄 + 각 프레임 당 1행)
    frames   : (n1, n2) 형태의 튜플 (옵션)
               - n1, n2 는 BVH 프레임 번호(1-based)라고 가정
               - 실제 CSV 인덱스로는 (n1-1), (n2-1)을 사용

    동작:
      - 기준 프레임: data[1] (3행 = 두 번째 프레임)
      - 항상 '기준 ~ 마지막 프레임' L2 거리 계산 (기존 동작)
      - frames 가 지정되면:
          * 기준 ~ n1 프레임 L2 거리
          * 기준 ~ n2 프레임 L2 거리
        도 추가로 계산하여 출력
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    # header 한 줄을 건너뛰고 데이터만 로드
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    if data.ndim == 1:
        # 프레임이 1개일 때 대비 (1D -> 2D)
        data = data.reshape(1, -1)

    if data.shape[0] < 3:
        raise ValueError("프레임 수가 너무 적습니다. 최소 3행 이상이어야 합니다.")

    num_frames = data.shape[0]

    # r_hand_x, r_hand_y, r_hand_z는 마지막 3개의 컬럼
    # header:
    # pelvis(3), l_shoulder(3), r_shoulder(3), l_hand(3), r_hand(3) = 15
    r_hand_cols = [-3, -2, -1]  # 혹은 [12, 13, 14]

    # 기준 프레임: 3행(데이터 index 1, 실제 두 번째 프레임)
    start_idx = 1
    start_r_hand = data[start_idx, r_hand_cols]

    # 마지막 프레임
    last_idx = num_frames - 1
    end_r_hand = data[last_idx, r_hand_cols]

    # 기준 ~ 마지막 L2 거리 (기존 동작)
    diff_last = end_r_hand - start_r_hand
    l2_last = np.linalg.norm(diff_last)

    print("=== r_hand 위치 비교 ===")

    # --frames n1 n2 가 지정된 경우: 중간 프레임 2개에 대해서도 L2 거리 계산
    if frames is not None:
        n1, n2 = frames

        # BVH 프레임 번호(1-based) -> CSV 인덱스(0-based)
        idx1 = n1 - 1
        idx2 = n2 - 1

        # 인덱스 유효성 체크
        if not (0 <= idx1 < num_frames):
            raise ValueError(
                f"n1={n1} 가 유효 범위를 벗어났습니다. "
                f"(총 프레임 수: {num_frames}, 허용 BVH 프레임 번호: 1~{num_frames})"
            )
        if not (0 <= idx2 < num_frames):
            raise ValueError(
                f"n2={n2} 가 유효 범위를 벗어났습니다. "
                f"(총 프레임 수: {num_frames}, 허용 BVH 프레임 번호: 1~{num_frames})"
            )

        mid1_r_hand = data[idx1, r_hand_cols]
        mid2_r_hand = data[idx2, r_hand_cols]

        diff_1 = mid1_r_hand - start_r_hand
        diff_2 = mid2_r_hand - start_r_hand

        l2_1 = np.linalg.norm(diff_1)
        l2_2 = np.linalg.norm(diff_2)

        print("\n--- 중간 프레임 L2 거리 ---")
        # print(f"[n1 프레임] BVH 프레임 {n1} (CSV index {idx1}) r_hand (x, y, z): {mid1_r_hand}")
        print(f"\n시작 ~ n1({n1}) 프레임 L2 거리: {l2_1:.6f} cm")

        # print(f"\n[n2 프레임] BVH 프레임 {n2} (CSV index {idx2}) r_hand (x, y, z): {mid2_r_hand}")
        print(f"\n시작 ~ n2({n2}) 프레임 L2 거리: {l2_2:.6f} cm")

    # print(f"\n[기준 프레임] 3행(데이터 index {start_idx}) r_hand (x, y, z): {start_r_hand}")
    # print(f"[마지막 프레임] 행 index {last_idx} r_hand (x, y, z): {end_r_hand}")
    print(f"\n시작 ~ 마지막 프레임 L2 거리: {l2_last:.6f} cm")

    # 기존처럼 마지막 프레임 기준 L2 거리 하나를 리턴 (호환성 유지)
    return l2_last


def parse_frames_args(argv):
    """
    sys.argv 리스트에서 --frames 인자를 파싱한다.
    사용 예:
      python L2distance_with_csv.py input.bvh --frames 30 60
      python L2distance_with_csv.py input.csv --frames=30,60
    반환:
      (n1, n2) 또는 None
    """
    frames = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--frames":
            if i + 2 >= len(argv):
                raise ValueError("--frames 뒤에 n1 n2 두 개의 정수가 필요합니다.")
            n1 = int(argv[i + 1])
            n2 = int(argv[i + 2])
            frames = (n1, n2)
            i += 3
        elif arg.startswith("--frames="):
            # --frames=30,60 또는 --frames=30 60 형태 지원
            val = arg.split("=", 1)[1]
            parts = val.replace(",", " ").split()
            if len(parts) < 2:
                raise ValueError("--frames 에는 최소 두 개의 값(n1 n2)이 필요합니다.")
            n1 = int(parts[0])
            n2 = int(parts[1])
            frames = (n1, n2)
            i += 1
        else:
            i += 1
    return frames


if __name__ == "__main__":
    if len(sys.argv) < 2:
        script_name = os.path.basename(sys.argv[0])
        print("사용법:")
        print(f"  python {script_name} <input.bvh> [--frames n1 n2]")
        print(f"  python {script_name} <input.csv> [--frames n1 n2]")
        sys.exit(1)

    input_path = sys.argv[1]
    extra_args = sys.argv[2:]

    try:
        frames = parse_frames_args(extra_args)
    except Exception as e:
        print(f"[에러] {e}")
        sys.exit(1)

    # 확장자로 BVH / CSV 판별
    lower = input_path.lower()
    if lower.endswith(".bvh"):
        # 1) BVH → CSV 생성
        csv_path = bvh_to_csv(input_path)
    else:
        # 2) 이미 CSV 라고 가정
        csv_path = input_path

    # 3) r_hand L2 distance 계산
    compare_r_hand(csv_path, frames)
