import sys
import os
from typing import List, Dict, Optional

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
        self.channels: List[str] = (
            []
        )  # 예: ["Xposition","Yposition","Zposition","Yrotation","Xrotation","Zrotation"]
        self.chan_start: int = -1  # 모션 배열에서 이 조인트의 첫 채널 인덱스

    def __repr__(self) -> str:
        return f"Joint({self.name}, ch={len(self.channels)}, start={self.chan_start})"


class BVH:
    def __init__(self, text: str):
        self.text = text
        self.root: Optional[Joint] = None
        self.joints_by_name: Dict[str, Joint] = {}
        self.frames: int = 0
        self.frame_time: float = 0.0
        self.motion: List[List[float]] = []  # List[프레임][채널값]
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
                raise ValueError(
                    f"조인트 {joint.name} 블록에서 '{{'를 기대했으나: {lines[idx]}"
                )
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
                    joint.channels = parts[2 : 2 + n]
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
        # 일반적으로 "Frame Time: 0.0083333" 형태
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
# 1) 마지막 프레임 각도 덮어쓰기 로직 (+ 가우시안 노이즈)
# --------------------------


def apply_angle_patch(
    bvh: BVH, joint_names: List[str], std: Optional[float] = None
) -> None:
    if not joint_names:
        return

    if bvh.frames < 2:
        raise ValueError("프레임이 2개 미만입니다. (2번째 프레임이 존재하지 않음)")

    start_idx = 1  # 두 번째 프레임 (0-based)
    end_idx = bvh.frames - 1  # 마지막 프레임

    start_frame = bvh.motion[start_idx]
    end_frame = bvh.motion[end_idx][:]  # 복사

    use_noise = std is not None and std > 0.0

    for name in joint_names:
        j = bvh.joints_by_name.get(name)
        if j is None:
            print(f"[경고] 조인트 '{name}' 를 찾을 수 없습니다. (무시)")
            continue

        if j.chan_start < 0 or not j.channels:
            print(f"[경고] 조인트 '{name}' 에 채널이 없습니다. (무시)")
            continue

        base = j.chan_start
        for k, ch in enumerate(j.channels):
            # 회전 채널만 수정
            if "rotation" in ch or "Rotation" in ch:
                src_val = start_frame[base + k]
                if use_noise:
                    np.random.seed(42)
                    noise = float(np.random.normal(loc=0.0, scale=std))
                    end_frame[base + k] = src_val + noise
                else:
                    end_frame[base + k] = src_val

    # 수정된 마지막 프레임을 되돌려 넣기
    bvh.motion[end_idx] = end_frame


# --------------------------
# 2) pelvis position drift (position_vector.csv) 적용
# --------------------------


def load_pelvis_drift_vector(csv_path: str) -> np.ndarray:
    """
    position_vector.csv 를 읽어서 (3,) 형태의 pelvis 이동 벡터를 반환한다.
    - 헤더는 무시하고, 첫 번째 데이터 행의 x,y,z 3개 값만 사용.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"pelvis drift CSV 를 찾을 수 없습니다: {csv_path}")

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    # 1행일 수도 있으므로 2D 로 강제 후 첫 행 사용
    data = np.atleast_2d(data)

    if data.shape[1] < 3:
        raise ValueError(f"CSV에 최소 3개 컬럼(x,y,z)이 필요합니다: {csv_path}")

    vec = data[0, :3].astype(float)  # (3,)
    return vec


def apply_pelvis_drift_last_frame(bvh: BVH, drift_vec: np.ndarray) -> None:
    """
    마지막 프레임 pelvis(root) position 을
        = 2번째 프레임 pelvis position + drift_vec
    로 덮어쓴다.

    - drift_vec: (3,) = (dx, dy, dz), cm 단위 (이미 BVH와 좌표계/단위 통일된 상태라고 가정)
    - BVH 의 root 조인트의 X/Y/Zposition 채널만 수정.
    """
    if drift_vec is None:
        return

    if bvh.root is None:
        raise RuntimeError("BVH에 root 조인트가 없습니다.")

    if bvh.frames < 2:
        raise ValueError("프레임이 2개 미만입니다. (2번째 프레임이 존재하지 않음)")

    root = bvh.root
    if root.chan_start < 0 or not root.channels:
        raise RuntimeError("root 조인트에 채널 정보가 없습니다.")

    base = root.chan_start

    # 2번째 프레임(인덱스 1)과 마지막 프레임(인덱스 frames-1)
    frame2 = bvh.motion[1]
    last_idx = bvh.frames - 1
    last_frame = bvh.motion[last_idx][:]  # 복사해서 수정

    # 2번째 프레임의 pelvis position 읽기
    # 채널 이름이 Xposition/Yposition/Zposition 이라고 가정
    pos2 = {"Xposition": None, "Yposition": None, "Zposition": None}
    for k, ch in enumerate(root.channels):
        if ch in pos2:
            pos2[ch] = frame2[base + k]

    if any(v is None for v in pos2.values()):
        raise RuntimeError(
            "root 조인트에서 Xposition/Yposition/Zposition 채널을 모두 찾지 못했습니다."
        )

    # 새 마지막 프레임 pelvis position = frame2 position + drift_vec
    dx, dy, dz = float(drift_vec[0]), float(drift_vec[1]), float(drift_vec[2])
    new_pos = {
        "Xposition": pos2["Xposition"] + dx,
        "Yposition": pos2["Yposition"] + dy,
        "Zposition": pos2["Zposition"] + dz,
    }

    for k, ch in enumerate(root.channels):
        idx = base + k
        if ch in new_pos:
            last_frame[idx] = new_pos[ch]
        # 회전 채널 및 기타 채널은 그대로 둠

    # 수정된 마지막 프레임 저장
    bvh.motion[last_idx] = last_frame


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


def main() -> None:
    if len(sys.argv) < 2:
        print("사용법:")
        print(
            "  python patch_bvh.py input.bvh "
            "[--pelvis_csv position_vector.csv] "
            "[--std 0.05] "
            "--RightShoulder --RightElbow ..."
        )
        sys.exit(1)

    bvh_path = sys.argv[1]
    if not os.path.isfile(bvh_path):
        print(f"BVH 파일을 찾을 수 없습니다: {bvh_path}")
        sys.exit(1)

    pelvis_csv_path: Optional[str] = None
    joint_names: List[str] = []
    std: Optional[float] = None

    # '--조인트이름', '--pelvis_csv 경로', '--std 값' 형식의 인자를 모두 모음
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.startswith("--pelvis_csv="):
            # --pelvis_csv=foo.csv
            pelvis_csv_path = arg.split("=", 1)[1]
            i += 1
        elif arg == "--pelvis_csv":
            # --pelvis_csv foo.csv
            if i + 1 >= len(sys.argv):
                print("[에러] --pelvis_csv 뒤에 CSV 경로를 지정해야 합니다.")
                sys.exit(1)
            pelvis_csv_path = sys.argv[i + 1]
            i += 2
        elif arg.startswith("--std="):
            # --std=0.05
            try:
                std = float(arg.split("=", 1)[1])
            except ValueError:
                sys.exit(1)
            i += 1
        elif arg == "--std":
            # --std 0.05
            if i + 1 >= len(sys.argv):
                sys.exit(1)
            try:
                std = float(sys.argv[i + 1])
            except ValueError:
                sys.exit(1)
            i += 2
        elif arg.startswith("--") and len(arg) > 2:
            # 그 외의 --Something 은 조인트 이름으로 처리
            joint_names.append(arg[2:])
            i += 1
        else:
            print(f"[주의] 알 수 없는 인자를 무시합니다: {arg}")
            i += 1

    # std 검증 및 0.1 미만으로 제한
    if std is not None:
        if std <= 0.0:
            std = None
        elif std >= 0.1:
            std = 0.0999

    with open(bvh_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    bvh = BVH(text)

    print(f"[정보] 프레임 수: {bvh.frames}, 채널 수: {bvh.total_channels}")
    print(
        f"[정보] 패치 대상 조인트: {', '.join(joint_names) if joint_names else '(없음)'}"
    )
    print(
        f"[정보] pelvis drift CSV: {pelvis_csv_path if pelvis_csv_path is not None else '(사용 안 함)'}"
    )

    # 1) 각도 패치 (+ 노이즈)
    apply_angle_patch(bvh, joint_names, std=std)

    # 2) pelvis position drift 적용 (마지막 프레임만)
    if pelvis_csv_path is not None:
        drift_vec = load_pelvis_drift_vector(pelvis_csv_path)
        print(f"[정보] pelvis drift 벡터: {drift_vec}")
        apply_pelvis_drift_last_frame(bvh, drift_vec)

    base, ext = os.path.splitext(bvh_path)
    out_path = base + "_patched" + ext

    write_bvh_with_new_motion(text, bvh, out_path)

    print(f"[완료] 패치된 BVH 저장: {out_path}")


if __name__ == "__main__":
    main()
