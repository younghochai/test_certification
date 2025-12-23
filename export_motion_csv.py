# export_motion_csv.py
import os
import sys
import csv
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# --------------------------
# --frames 파싱 (프로젝트 방식)
# --------------------------
def parse_frames_args(argv: List[str]) -> Optional[Tuple[int, int]]:
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


# --------------------------
# FK용 행렬 유틸
# --------------------------
def Rx(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[1, 0, 0, 0],
                     [0, c, -s, 0],
                     [0, s,  c, 0],
                     [0, 0, 0, 1]], dtype=float)

def Ry(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]], dtype=float)

def Rz(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, -s, 0, 0],
                     [s,  c, 0, 0],
                     [0,  0, 1, 0],
                     [0,  0, 0, 1]], dtype=float)

def T(tx: float, ty: float, tz: float) -> np.ndarray:
    M = np.eye(4, dtype=float)
    M[:3, 3] = [tx, ty, tz]
    return M


# --------------------------
# BVH 파서 (OFFSET + End Site OFFSET 포함)
# --------------------------
@dataclass
class Joint:
    name: str
    parent: Optional["Joint"]
    children: List["Joint"] = field(default_factory=list)

    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    end_site_offset: Optional[Tuple[float, float, float]] = None

    channels: List[str] = field(default_factory=list)
    chan_start: int = -1


class BVH:
    def __init__(self, text: str):
        self.text = text
        self.root: Optional[Joint] = None
        self.joints_by_name: Dict[str, Joint] = {}
        self.frames: int = 0
        self.frame_time: float = 0.0
        self.motion: List[List[float]] = []
        self.total_channels: int = 0
        self._parse()

    def _parse(self) -> None:
        lines = [ln.strip() for ln in self.text.splitlines() if ln.strip() != ""]
        if not lines or not lines[0].upper().startswith("HIERARCHY"):
            raise ValueError("BVH 파일이 아니거나 HIERARCHY 헤더가 없습니다.")

        i = 1
        channel_cursor = 0

        def parse_joint_block(j: Joint, idx: int) -> int:
            nonlocal channel_cursor
            if lines[idx] != "{":
                raise ValueError(f"조인트 {j.name} 블록에서 '{{'를 기대했으나: {lines[idx]}")
            idx += 1

            while idx < len(lines):
                ln = lines[idx]

                if ln.startswith("OFFSET"):
                    parts = ln.split()
                    nums = list(map(float, parts[1:4]))
                    j.offset = (nums[0], nums[1], nums[2])
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
                    child = Joint(name=name, parent=j)
                    j.children.append(child)
                    self.joints_by_name[name] = child
                    idx += 1
                    idx = parse_joint_block(child, idx)

                elif ln.startswith("End Site"):
                    # End Site 블록: OFFSET만 읽어서 parent joint에 저장
                    idx += 1
                    if lines[idx] != "{":
                        raise ValueError("End Site 뒤에 '{' 필요")
                    idx += 1
                    if not lines[idx].startswith("OFFSET"):
                        raise ValueError("End Site 블록에 OFFSET 필요")
                    parts = lines[idx].split()
                    nums = list(map(float, parts[1:4]))
                    j.end_site_offset = (nums[0], nums[1], nums[2])
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
            raise ValueError("HIERARCHY 다음에 ROOT가 나와야 합니다.")
        root_name = lines[i].split()[1]
        i += 1
        self.root = Joint(name=root_name, parent=None)
        self.joints_by_name[root_name] = self.root
        i = parse_joint_block(self.root, i)

        # MOTION
        if not lines[i].startswith("MOTION"):
            raise ValueError("MOTION 섹션을 찾을 수 없습니다.")
        i += 1

        if not lines[i].startswith("Frames:"):
            raise ValueError("Frames: 라인을 찾을 수 없습니다.")
        self.frames = int(lines[i].split()[1])
        i += 1

        if not lines[i].startswith("Frame Time:"):
            raise ValueError("Frame Time: 라인을 찾을 수 없습니다.")
        self.frame_time = float(lines[i].split()[-1])
        i += 1

        self.total_channels = channel_cursor

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

    # 채널값에서 (Tpos, Rrot) 구성 (채널 순서대로 회전 적용)
    def _local_tr(self, j: Joint, frame_vals: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        tx = ty = tz = 0.0
        R = np.eye(4, dtype=float)

        if j.channels and j.chan_start >= 0:
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
        return T(tx, ty, tz), R

    # 조인트 월드 행렬
    def world_matrix(self, j: Joint, frame_vals: List[float]) -> np.ndarray:
        if j.parent is None:
            Tloc, Rloc = self._local_tr(j, frame_vals)
            return Tloc @ Rloc
        Mp = self.world_matrix(j.parent, frame_vals)
        Tloc, Rloc = self._local_tr(j, frame_vals)
        ox, oy, oz = j.offset
        return Mp @ T(ox, oy, oz) @ Rloc @ Tloc

    # RightWrist End Site 월드 좌표
    def right_wrist_end_world(self, frame_vals: List[float]) -> Optional[np.ndarray]:
        rw = self.joints_by_name.get("RightWrist")
        if rw is None or rw.end_site_offset is None:
            return None
        Mw = self.world_matrix(rw, frame_vals)
        ex, ey, ez = rw.end_site_offset
        p = Mw @ np.array([ex, ey, ez, 1.0], dtype=float)
        return p[:3]


def build_channel_headers(bvh: BVH) -> List[str]:
    headers = [f"ch{i}" for i in range(bvh.total_channels)]
    for j in bvh.joints_by_name.values():
        if j.chan_start >= 0 and j.channels:
            for k, ch in enumerate(j.channels):
                headers[j.chan_start + k] = f"{j.name}_{ch}"
    return headers


def export_4frames_csv(bvh_path: str, n1: int, n2: int, out_csv: str) -> None:
    if not os.path.isfile(bvh_path):
        raise FileNotFoundError(f"BVH 파일을 찾을 수 없습니다: {bvh_path}")

    with open(bvh_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    bvh = BVH(text)

    if bvh.frames < 2:
        raise ValueError("BVH에는 최소 2프레임 이상이 필요합니다. (기준 프레임=2번째 프레임 사용)")

    base_idx = 1
    mid1_idx = n1 - 1
    mid2_idx = n2 - 1
    last_idx = bvh.frames - 1

    for idx, name in [(mid1_idx, "n1"), (mid2_idx, "n2")]:
        if idx < 0 or idx >= bvh.frames:
            raise ValueError(f"{name}={idx+1} 가 범위를 벗어났습니다. (허용: 1 ~ {bvh.frames})")

    pick_indices = [base_idx, mid1_idx, mid2_idx, last_idx]
    channel_headers = build_channel_headers(bvh)

    # ✅ 오른손 End Site 컬럼 추가
    extra_headers = ["rEndSite_Xposition", "rEndSite_Yposition", "rEndSite_Zposition"]

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame"] + channel_headers + extra_headers)

        for idx in pick_indices:
            frame_no = idx + 1
            vals = bvh.motion[idx]

            end_xyz = bvh.right_wrist_end_world(vals)
            if end_xyz is None:
                end_out = ["nan", "nan", "nan"]
            else:
                end_out = [f"{end_xyz[0]:.6f}", f"{end_xyz[1]:.6f}", f"{end_xyz[2]:.6f}"]

            w.writerow([frame_no] + [f"{v:.6f}" for v in vals] + end_out)

    # print(f"[완료] 4프레임 MOTION CSV 저장: {out_csv}")
    # print(f"  기준(frame=2), 중간1(frame={n1}), 중간2(frame={n2}), 마지막(frame={bvh.frames})")
    if bvh.joints_by_name.get("RightWrist") is None:
        print("[주의] BVH에 'RightWrist' 조인트가 없습니다. r_end_x/y/z는 nan으로 저장됩니다.")
    elif bvh.joints_by_name["RightWrist"].end_site_offset is None:
        print("[주의] RightWrist의 End Site OFFSET이 없습니다. r_end_x/y/z는 nan으로 저장됩니다.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        script = os.path.basename(sys.argv[0])
        print("사용법:")
        print(f"  python {script} input_patched.bvh --frames n1 n2 [--out out.csv]")
        sys.exit(1)

    bvh_path = sys.argv[1]
    extra = sys.argv[2:]

    frames_pair = parse_frames_args(extra)
    if frames_pair is None:
        raise ValueError("4프레임 추출을 위해 --frames n1 n2 가 필요합니다.")

    n1, n2 = frames_pair

    out_csv = None
    i = 0
    while i < len(extra):
        a = extra[i]
        if a == "--out":
            if i + 1 >= len(extra):
                raise ValueError("--out 뒤에 출력 CSV 경로가 필요합니다.")
            out_csv = extra[i + 1]
            i += 2
        elif a.startswith("--out="):
            out_csv = a.split("=", 1)[1]
            i += 1
        else:
            i += 1

    if out_csv is None:
        base, _ = os.path.splitext(os.path.basename(bvh_path))
        out_dir = os.path.dirname(os.path.abspath(bvh_path))
        out_csv = os.path.join(out_dir, f"{base}_4frames_motion.csv")

    export_4frames_csv(bvh_path=bvh_path, n1=n1, n2=n2, out_csv=out_csv)
