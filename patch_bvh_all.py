# patch_bvh_all.py
import sys
import os
from typing import List, Dict, Optional, Tuple

import numpy as np


# --------------------------
# 간단한 BVH 파서 (계층 + 채널 인덱스만 필요)
# --------------------------

class Joint:
    def __init__(self, name: str, parent: Optional["Joint"]):
        self.name = name
        self.parent = parent
        self.children: List["Joint"] = []
        self.offset = (0.0, 0.0, 0.0)
        self.channels: List[str] = []   # 예: ["Xposition","Yposition","Zposition","Yrotation","Xrotation","Zrotation"]
        self.chan_start: int = -1       # 모션 배열에서 이 조인트의 첫 채널 인덱스

    def __repr__(self) -> str:
        return f"Joint({self.name}, ch={len(self.channels)}, start={self.chan_start})"


class BVH:
    def __init__(self, text: str):
        self.text = text
        self.root: Optional[Joint] = None
        self.joints_by_name: Dict[str, Joint] = {}
        self.frames: int = 0
        self.frame_time: float = 0.0
        self.motion: List[List[float]] = []   # List[프레임][채널값]
        self.total_channels: int = 0
        self._parse()

    # ---- 파서 ----
    def _parse(self) -> None:
        # 빈 줄 제거 + 양쪽 공백 제거
        lines = [ln.strip() for ln in self.text.splitlines() if ln.strip() != ""]
        if not lines or not lines[0].upper().startswith("HIERARCHY"):
            raise ValueError("BVH 파일이 아니거나 HIERARCHY 헤더가 없습니다.")
        i = 1

        channel_cursor = 0

        def parse_joint_block(joint: Joint, idx: int) -> int:
            nonlocal channel_cursor
            if lines[idx] != "{":
                raise ValueError(f"조인트 {joint.name} 블록에서 '{{'를 기대했으나: {lines[idx]}")
            idx += 1
            while idx < len(lines):
                ln = lines[idx]
                if ln.startswith("OFFSET"):
                    parts = ln.split()
                    nums = list(map(float, parts[1:4]))
                    joint.offset = tuple(nums)
                    idx += 1

                elif ln.startswith("CHANNELS"):
                    parts = ln.split()
                    n = int(parts[1])
                    joint.channels = parts[2:2 + n]
                    joint.chan_start = channel_cursor
                    channel_cursor += n
                    idx += 1

                elif ln.startswith("JOINT"):
                    # 자식 조인트
                    name = ln.split()[1]
                    child = Joint(name, joint)
                    joint.children.append(child)
                    self.joints_by_name[name] = child
                    idx += 1
                    idx = parse_joint_block(child, idx)

                elif ln.startswith("End Site"):
                    # End Site 블록 건너뛰기
                    idx += 1  # "End Site"
                    if lines[idx] != "{":
                        raise ValueError("End Site 뒤에 '{' 필요")
                    idx += 1
                    if not lines[idx].startswith("OFFSET"):
                        raise ValueError("End Site 블록에 OFFSET 필요")
                    idx += 1
                    if lines[idx] != "}":
                        raise ValueError("End Site 블록 종료 '}' 필요")
                    idx += 1

                elif ln == "}":
                    idx += 1
                    return idx

                else:
                    # 모르는 라인은 그냥 넘김
                    idx += 1
            return idx

        # ROOT
        if not lines[i].startswith("ROOT"):
            raise ValueError("HIERARCHY 다음에 ROOT가 나와야 합니다.")
        root_name = lines[i].split()[1]
        i += 1
        self.root = Joint(root_name, None)
        self.joints_by_name[root_name] = self.root
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
        parts = lines[i].split()
        self.frame_time = float(parts[-1])
        i += 1

        self.total_channels = channel_cursor

        # 각 프레임 모션 데이터
        self.motion = []
        for f in range(self.frames):
            if i + f >= len(lines):
                raise ValueError(f"프레임 {f} 데이터를 읽는 중 파일이 끝났습니다.")
            vals = list(map(float, lines[i + f].split()))
            if len(vals) != self.total_channels:
                raise ValueError(
                    f"Frame {f} 채널 수 불일치: {len(vals)}개, 기대값 {self.total_channels}"
                )
            self.motion.append(vals)


# --------------------------
# 1) 회전 각도 덮어쓰기 (+ 가우시안 노이즈, 여러 프레임)
# --------------------------
def apply_angle_patch_multi(
    bvh: BVH,
    special_joint_names: List[str],
    target_frame_indices: List[int],
    std1: Optional[float],
    std2: Optional[float],
) -> None:
    """
    - 기준 프레임: 2번째 프레임 (index=1)의 회전 값을 기준으로
    - target_frame_indices 에 포함된 프레임들의 회전 채널을 덮어쓰기
    - special_joint_names 에 포함된 조인트:
        → std1 표준편차의 노이즈
      그 외 모든 조인트:
        → std2 표준편차의 노이즈
    - std1, std2 가 None 이면 해당 그룹은 노이즈 없이 그냥 덮어쓰기만 수행
    """
    if not target_frame_indices:
        return

    if bvh.frames < 2:
        raise ValueError("프레임이 2개 미만입니다. (2번째 프레임이 존재하지 않음)")

    # 기준 프레임: 2번째 프레임
    ref_idx = 1
    ref_frame = bvh.motion[ref_idx]

    special_set = set(special_joint_names or [])

    def effective_std_for_joint(jname: str) -> float:
        if jname in special_set:
            return std1 if std1 is not None else 0.0
        else:
            return std2 if std2 is not None else 0.0

    # 중복된 프레임 인덱스 방지
    unique_targets = sorted(set(target_frame_indices))

    for frame_idx in unique_targets:
        if frame_idx < 0 or frame_idx >= bvh.frames:
            raise ValueError(f"frame_idx={frame_idx} 가 유효 범위를 벗어났습니다. (0~{bvh.frames-1})")

        # 현재 프레임 값 복사 후 수정
        frame_vals = bvh.motion[frame_idx][:]

        for j in bvh.joints_by_name.values():
            if j.chan_start < 0 or not j.channels:
                continue

            base = j.chan_start
            noise_std = effective_std_for_joint(j.name)

            for k, ch in enumerate(j.channels):
                # 회전 채널만 처리
                if "rotation" in ch or "Rotation" in ch:
                    src_val = ref_frame[base + k]

                    if noise_std > 0.0:
                        # 기존 스타일을 따라, 고정 시드로 항상 같은 랜덤 값 사용
                        np.random.seed(42)
                        unit = float(np.random.normal(loc=0.0, scale=1.0))
                        noise = unit * noise_std
                        frame_vals[base + k] = src_val + noise
                    else:
                        frame_vals[base + k] = src_val
                # position 채널 등은 건드리지 않음

        bvh.motion[frame_idx] = frame_vals


# --------------------------
# 2) pelvis position drift (position_vector_XX.csv) 적용
# --------------------------

def load_pelvis_drift_vector(csv_path: str) -> np.ndarray:
    """
    position_vector_XX.csv 를 읽어서 (3,) 형태의 pelvis 이동 벡터를 반환한다.
    - 헤더는 무시하고, 첫 번째 데이터 행의 x,y,z 3개 값만 사용.
    - 단위는 cm, BVH 좌표계와 이미 맞춰져 있다고 가정.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"pelvis drift CSV 를 찾을 수 없습니다: {csv_path}")

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    # 1행일 수도 있으므로 2D 로 강제 후 첫 행 사용
    data = np.atleast_2d(data)

    if data.shape[1] < 3:
        raise ValueError(f"CSV에 최소 3개 컬럼(x,y,z)이 필요합니다: {csv_path}")

    vec = data[0, :3].astype(float)   # (3,)
    return vec


def apply_pelvis_drift_to_frame(bvh: BVH, frame_idx: int, drift_vec: np.ndarray) -> None:
    """
    특정 프레임(frame_idx)의 pelvis(root) position 을
        = 2번째 프레임 pelvis position + drift_vec
    로 덮어쓴다.

    - frame_idx: 0-based (0 ~ frames-1)
    - drift_vec: (3,) = (dx, dy, dz), cm 단위
    - BVH 의 root 조인트의 X/Y/Zposition 채널만 수정.
    """
    if drift_vec is None:
        return

    if bvh.root is None:
        raise RuntimeError("BVH에 root 조인트가 없습니다.")

    if bvh.frames < 2:
        raise ValueError("프레임이 2개 미만입니다. (2번째 프레임이 존재하지 않음)")

    if not (0 <= frame_idx < bvh.frames):
        raise ValueError(f"frame_idx={frame_idx} 가 유효 범위를 벗어났습니다. (0~{bvh.frames-1})")

    root = bvh.root
    if root.chan_start < 0 or not root.channels:
        raise RuntimeError("root 조인트에 채널 정보가 없습니다.")

    base = root.chan_start

    # 2번째 프레임(인덱스 1)의 pelvis position 읽기
    frame2 = bvh.motion[1]
    pos2 = {"Xposition": None, "Yposition": None, "Zposition": None}
    for k, ch in enumerate(root.channels):
        if ch in pos2:
            pos2[ch] = frame2[base + k]

    if any(v is None for v in pos2.values()):
        raise RuntimeError(
            "root 조인트에서 Xposition/Yposition/Zposition 채널을 모두 찾지 못했습니다."
        )

    dx, dy, dz = float(drift_vec[0]), float(drift_vec[1]), float(drift_vec[2])
    new_pos = {
        "Xposition": pos2["Xposition"] + dx,
        "Yposition": pos2["Yposition"] + dy,
        "Zposition": pos2["Zposition"] + dz,
    }

    target_frame = bvh.motion[frame_idx][:]
    for k, ch in enumerate(root.channels):
        idx = base + k
        if ch in new_pos:
            target_frame[idx] = new_pos[ch]
        # 회전 채널 및 기타 채널은 그대로 둠

    bvh.motion[frame_idx] = target_frame


def apply_pelvis_drift_last_frame(bvh: BVH, drift_vec: np.ndarray) -> None:
    """
    마지막 프레임에 drift_vec 적용.
    """
    if drift_vec is None:
        return
    last_idx = bvh.frames - 1
    apply_pelvis_drift_to_frame(bvh, last_idx, drift_vec)


# --------------------------
# BVH 다시 쓰기
# --------------------------

def write_bvh_with_new_motion(original_text: str, bvh: BVH, out_path: str) -> None:
    """
    원본 텍스트에서 HIERARCHY 부분은 그대로 쓰고,
    MOTION 이후부터는 BVH 객체의 frames / frame_time / motion 데이터를 사용해 다시 쓴다.
    """
    raw_lines = original_text.splitlines(keepends=True)

    # MOTION 라인 위치 찾기
    motion_idx = None
    for i, ln in enumerate(raw_lines):
        if ln.strip().upper().startswith("MOTION"):
            motion_idx = i
            break

    if motion_idx is None:
        raise ValueError("원본 텍스트에서 MOTION 라인을 찾을 수 없습니다.")

    header_lines = raw_lines[:motion_idx]  # HIERARCHY ~ JOINT 정의 부분

    with open(out_path, "w", encoding="utf-8") as f:
        # 헤더 그대로 출력
        f.writelines(header_lines)

        # MOTION 헤더
        f.write("MOTION\n")
        f.write(f"Frames: {bvh.frames}\n")
        f.write(f"Frame Time: {bvh.frame_time:.6f}\n")

        # 모션 프레임들
        for frame_vals in bvh.motion:
            line = " ".join(f"{v:.6f}" for v in frame_vals)
            f.write(line + "\n")


# --------------------------
# main
# --------------------------

def parse_frames_arg(arg_values: List[str]) -> Tuple[int, int]:
    """
    --frames n1 n2 파싱용.
    n1, n2 는 1-based 프레임 번호로 받고,
    (n1, n2)를 정수로 반환한다.
    """
    if len(arg_values) < 2:
        raise ValueError("--frames 뒤에는 두 개의 정수(n1 n2)가 필요합니다.")
    try:
        n1 = int(arg_values[0])
        n2 = int(arg_values[1])
    except ValueError:
        raise ValueError("--frames 인자는 정수여야 합니다.")
    if n1 <= 0 or n2 <= 0:
        raise ValueError("프레임 번호는 1 이상의 정수여야 합니다.")
    return n1, n2


def main() -> None:
    if len(sys.argv) < 2:
        print("사용법:")
        print("  python patch_bvh_all.py input.bvh "
              "[--pelvis_only] "
              "[--pelvis_csv_01 position_vector_01.csv] "
              "[--pelvis_csv_02 position_vector_02.csv] "
              "[--pelvis_csv_03 position_vector_03.csv] "
              "[--frames n1 n2] "
              "[--std1 0.05] "
              "[--std2 0.01] "
              "--RightShoulder --RightElbow ...")
        sys.exit(1)

    bvh_path = sys.argv[1]
    if not os.path.isfile(bvh_path):
        print(f"BVH 파일을 찾을 수 없습니다: {bvh_path}")
        sys.exit(1)

    pelvis_csv_01: Optional[str] = None
    pelvis_csv_02: Optional[str] = None
    pelvis_csv_03: Optional[str] = None
    joint_names: List[str] = []
    std1: Optional[float] = None   # Bonename 지정 조인트용 노이즈
    std2: Optional[float] = None   # 나머지 모든 조인트용 노이즈
    frames_pair: Optional[Tuple[int, int]] = None
    pelvis_only: bool = False      # True면 회전값을 덮어쓰지 않고 pelvis position drift만 적용

    # '--조인트이름', '--pelvis_csv_XX 경로', '--frames n1 n2', '--std1/2 값' 형식의 인자를 모두 모음
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]

        # --- pelvis_only ---
        if arg == "--pelvis_only":
            pelvis_only = True
            i += 1

        # --- pelvis_csv_01 ---
        elif arg.startswith("--pelvis_csv_01="):
            pelvis_csv_01 = arg.split("=", 1)[1]
            i += 1
        elif arg == "--pelvis_csv_01":
            if i + 1 >= len(sys.argv):
                print("[에러] --pelvis_csv_01 뒤에 CSV 경로를 지정해야 합니다.")
                sys.exit(1)
            pelvis_csv_01 = sys.argv[i + 1]
            i += 2

        # --- pelvis_csv_02 ---
        elif arg.startswith("--pelvis_csv_02="):
            pelvis_csv_02 = arg.split("=", 1)[1]
            i += 1
        elif arg == "--pelvis_csv_02":
            if i + 1 >= len(sys.argv):
                print("[에러] --pelvis_csv_02 뒤에 CSV 경로를 지정해야 합니다.")
                sys.exit(1)
            pelvis_csv_02 = sys.argv[i + 1]
            i += 2

        # --- pelvis_csv_03 ---
        elif arg.startswith("--pelvis_csv_03="):
            pelvis_csv_03 = arg.split("=", 1)[1]
            i += 1
        elif arg == "--pelvis_csv_03":
            if i + 1 >= len(sys.argv):
                print("[에러] --pelvis_csv_03 뒤에 CSV 경로를 지정해야 합니다.")
                sys.exit(1)
            pelvis_csv_03 = sys.argv[i + 1]
            i += 2

        # 기존 patch_bvh.py 와의 호환용: --pelvis_csv 를 03으로 취급
        elif arg.startswith("--pelvis_csv="):
            pelvis_csv_03 = arg.split("=", 1)[1]
            i += 1
        elif arg == "--pelvis_csv":
            if i + 1 >= len(sys.argv):
                print("[에러] --pelvis_csv 뒤에 CSV 경로를 지정해야 합니다.")
                sys.exit(1)
            pelvis_csv_03 = sys.argv[i + 1]
            i += 2

        # --- frames: n1 n2 ---
        elif arg == "--frames":
            try:
                n1, n2 = parse_frames_arg(sys.argv[i + 1:i + 3])
            except Exception as e:
                print(f"[에러] {e}")
                sys.exit(1)
            frames_pair = (n1, n2)
            i += 3

        # (선택) --frames=n1,n2 꼴도 허용
        elif arg.startswith("--frames="):
            val = arg.split("=", 1)[1]
            parts = val.replace(",", " ").split()
            try:
                n1, n2 = parse_frames_arg(parts[:2])
            except Exception as e:
                print(f"[에러] {e}")
                sys.exit(1)
            frames_pair = (n1, n2)
            i += 1

        # --- std1 ---
        elif arg.startswith("--std1="):
            try:
                std1 = float(arg.split("=", 1)[1])
            except ValueError:
                # print("[에러] --std1 값이 실수 형태가 아닙니다.")
                sys.exit(1)
            i += 1
        elif arg == "--std1":
            if i + 1 >= len(sys.argv):
                # print("[에러] --std1 뒤에 값이 필요합니다.")
                sys.exit(1)
            try:
                std1 = float(sys.argv[i + 1])
            except ValueError:
                # print("[에러] --std1 값이 실수 형태가 아닙니다.")
                sys.exit(1)
            i += 2

        # --- std2 ---
        elif arg.startswith("--std2="):
            try:
                std2 = float(arg.split("=", 1)[1])
            except ValueError:
                # print("[에러] --std2 값이 실수 형태가 아닙니다.")
                sys.exit(1)
            i += 1
        elif arg == "--std2":
            if i + 1 >= len(sys.argv):
                # print("[에러] --std2 뒤에 값이 필요합니다.")
                sys.exit(1)
            try:
                std2 = float(sys.argv[i + 1])
            except ValueError:
                # print("[에러] --std2 값이 실수 형태가 아닙니다.")
                sys.exit(1)
            i += 2

        # --- (선택) 구버전 호환: --std 를 std1 으로 처리 ---
        elif arg.startswith("--std="):
            # print("[주의] --std 는 deprecated 입니다. --std1/--std2 를 사용하세요. (여기서는 std1 으로 사용)")
            try:
                std1 = float(arg.split("=", 1)[1])
            except ValueError:
                # print("[에러] --std 값이 실수 형태가 아닙니다.")
                sys.exit(1)
            i += 1
        elif arg == "--std":
            # print("[주의] --std 는 deprecated 입니다. --std1/--std2 를 사용하세요. (여기서는 std1 으로 사용)")
            if i + 1 >= len(sys.argv):
                # print("[에러] --std 뒤에 값이 필요합니다.")
                sys.exit(1)
            try:
                std1 = float(sys.argv[i + 1])
            except ValueError:
                # print("[에러] --std 값이 실수 형태가 아닙니다.")
                sys.exit(1)
            i += 2

        # --- 조인트 이름들: --RightShoulder 같은 형태 ---
        elif arg.startswith("--") and len(arg) > 2:
            joint_names.append(arg[2:])
            i += 1

        else:
            print(f"[주의] 알 수 없는 인자를 무시합니다: {arg}")
            i += 1

    with open(bvh_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    bvh = BVH(text)

    print(f"[정보] 프레임 수: {bvh.frames}, 채널 수: {bvh.total_channels}")
    # print(f"[정보] pelvis_only: {'ON (회전 패치 생략, pelvis position만 적용)' if pelvis_only else 'OFF (회전+pelvis 패치)'}")
    # print(f"[정보] 패치 대상(특정) 조인트: {', '.join(joint_names) if joint_names else '(없음)'}")
    print(f"[정보] pelvis CSV 01: {pelvis_csv_01 if pelvis_csv_01 else '(사용 안 함)'}")
    print(f"[정보] pelvis CSV 02: {pelvis_csv_02 if pelvis_csv_02 else '(사용 안 함)'}")
    print(f"[정보] pelvis CSV 03: {pelvis_csv_03 if pelvis_csv_03 else '(사용 안 함)'}")
    print(f"[정보] frames (중간 프레임): {frames_pair if frames_pair else '(지정 안 함)'}")
    # print(f"[정보] std1 (특정 조인트 노이즈): {std1 if std1 is not None else '(사용 안 함)'}")
    # print(f"[정보] std2 (나머지 조인트 노이즈): {std2 if std2 is not None else '(사용 안 함)'}")

    # if pelvis_only:
    #     if joint_names:
    #         print("[주의] --pelvis_only 가 켜져 있어 조인트 이름 인자(--RightShoulder 등)는 무시됩니다.")
    #     if std1 is not None or std2 is not None:
    #         print("[주의] --pelvis_only 가 켜져 있어 --std1/--std2 는 무시됩니다.")

    # --- 1) 회전 각도 덮어쓰기 (+ 노이즈) ---
    # 대상 프레임: 중간 프레임 1, 2 (옵션), 그리고 마지막 프레임
    if not pelvis_only:
        target_indices: List[int] = []
        last_idx = bvh.frames - 1
        target_indices.append(last_idx)

        if frames_pair is not None:
            n1, n2 = frames_pair
            idx1 = n1 - 1   # 1-based -> 0-based
            idx2 = n2 - 1

            if not (0 <= idx1 < bvh.frames) or not (0 <= idx2 < bvh.frames):
                print(f"[에러] --frames 에 지정한 프레임 번호가 범위를 벗어났습니다. "
                      f"(총 프레임: {bvh.frames}, 지정: {n1}, {n2})")
                sys.exit(1)

            target_indices.extend([idx1, idx2])

        apply_angle_patch_multi(
            bvh=bvh,
            special_joint_names=joint_names,
            target_frame_indices=target_indices,
            std1=std1,
            std2=std2,
        )

    # --- 2) pelvis position drift 적용 ---
    drift_01 = load_pelvis_drift_vector(pelvis_csv_01) if pelvis_csv_01 else None
    drift_02 = load_pelvis_drift_vector(pelvis_csv_02) if pelvis_csv_02 else None
    drift_03 = load_pelvis_drift_vector(pelvis_csv_03) if pelvis_csv_03 else None

    if frames_pair is not None:
        n1, n2 = frames_pair
        idx1 = n1 - 1
        idx2 = n2 - 1

        if drift_01 is not None:
            print(f"[정보] 프레임 {n1} (index {idx1}) 에 drift_01 적용")
            apply_pelvis_drift_to_frame(bvh, idx1, drift_01)
        else:
            print("[주의] drift_01 (pelvis_csv_01) 이 없어 프레임 n1 에는 pelvis 패치를 하지 않습니다.")

        if drift_02 is not None:
            print(f"[정보] 프레임 {n2} (index {idx2}) 에 drift_02 적용")
            apply_pelvis_drift_to_frame(bvh, idx2, drift_02)
        else:
            print("[주의] drift_02 (pelvis_csv_02) 이 없어 프레임 n2 에는 pelvis 패치를 하지 않습니다.")
    else:
        if drift_01 is not None or drift_02 is not None:
            print("[주의] --frames 가 지정되지 않아 01/02 벡터는 사용되지 않습니다.")

    # 마지막 프레임에 drift_03 적용
    if drift_03 is not None:
        print(f"[정보] 마지막 프레임(index {bvh.frames - 1}) 에 drift_03 적용")
        apply_pelvis_drift_last_frame(bvh, drift_03)

    base, ext = os.path.splitext(bvh_path)
    out_path = base + "_patched" + ext

    write_bvh_with_new_motion(text, bvh, out_path)

    # print(f"[완료] 패치된 BVH 저장: {out_path}")


if __name__ == "__main__":
    main()
