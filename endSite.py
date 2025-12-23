import re
import math
import os
import sys
from typing import List, Dict, Optional, Tuple

import numpy as np


# ---- 행렬 유틸 -------------------------------------------------------------
def Rx(deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array(
        [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=float
    )


def Ry(deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array(
        [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], dtype=float
    )


def Rz(deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array(
        [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
    )


def T(tx, ty, tz):
    M = np.eye(4)
    M[:3, 3] = [tx, ty, tz]
    return M


# ---- BVH 양식 --------------------------------------------------------------
class Joint:
    def __init__(self, name: str, parent: Optional["Joint"]):
        self.name = name
        self.parent = parent
        self.children: List["Joint"] = []
        self.offset = np.zeros(3, dtype=float)
        self.channels: List[str] = (
            []
        )  # ["Xposition","Yposition","Zposition","Yrotation","Xrotation","Zrotation"]
        self.chan_start: int = -1
        self.end_site_offset: Optional[np.ndarray] = None

    def __repr__(self):
        return f"Joint({self.name}, ch={len(self.channels)}, off={self.offset}, end={self.end_site_offset})"


# ---- BVH 파서 --------------------------------------------------------------
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

    def _parse(self):
        lines = [ln.strip() for ln in self.text.splitlines() if ln.strip() != ""]
        assert lines[0].upper().startswith("HIERARCHY")
        i = 1

        channel_cursor = 0

        def parse_joint_block(j: Joint, idx: int) -> int:
            nonlocal channel_cursor
            assert lines[idx] == "{"
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
                    j.channels = parts[2 : 2 + n]
                    j.chan_start = channel_cursor
                    channel_cursor += n
                    idx += 1
                elif ln.startswith("JOINT") or ln.startswith("End Site"):
                    if ln.startswith("JOINT"):
                        name = ln.split()[1]
                        child = Joint(name, j)
                        j.children.append(child)
                        self.joints_by_name[name] = child
                        idx += 1
                        idx = parse_joint_block(child, idx)
                    else:
                        # End Site
                        idx += 1
                        assert lines[idx] == "{"
                        idx += 1
                        assert lines[idx].startswith("OFFSET")
                        nums = list(map(float, lines[idx].split()[1:4]))
                        j.end_site_offset = np.array(nums, dtype=float)
                        idx += 1
                        assert lines[idx] == "}"
                        idx += 1
                elif ln == "}":
                    idx += 1
                    return idx
                else:
                    idx += 1
            return idx

        # ROOT
        assert lines[i].startswith("ROOT")
        root_name = lines[i].split()[1]
        i += 1
        self.root = Joint(root_name, None)
        self.joints_by_name[root_name] = self.root
        i = parse_joint_block(self.root, i)

        # MOTION
        assert lines[i].startswith("MOTION")
        i += 1
        assert lines[i].startswith("Frames:")
        self.frames = int(lines[i].split()[1])
        i += 1
        assert lines[i].startswith("Frame Time:")
        self.frame_time = float(lines[i].split()[2])
        i += 1

        # 총 채널 수 확정
        self.total_channels = channel_cursor

        # 모션 데이터
        for f in range(self.frames):
            vals = list(map(float, lines[i + f].split()))
            if len(vals) != self.total_channels:
                raise ValueError(
                    f"Frame {f} channel count mismatch: got {len(vals)} expected {self.total_channels}"
                )
            self.motion.append(vals)

    # 채널 값 → (pos, rot) 추출 (채널 순서 유지)
    def _local_tr(
        self, j: Joint, frame_vals: List[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        tx = ty = tz = 0.0
        R = np.eye(4)
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
                    raise ValueError(f"Unknown channel {ch}")
        return T(tx, ty, tz), R

    # 특정 관절의 월드 변환(4x4) 계산
    def world_matrix(self, j: Joint, frame_vals: List[float]) -> np.ndarray:
        if j.parent is None:
            # root: T(position) @ R(channels)
            Tloc, Rloc = self._local_tr(j, frame_vals)
            return Tloc @ Rloc
        else:
            M_parent = self.world_matrix(j.parent, frame_vals)
            Tloc, Rloc = self._local_tr(j, frame_vals)
            # BVH FK: parent @ T(OFFSET) @ R(channels) @ T(position)
            return M_parent @ T(*j.offset.tolist()) @ Rloc @ Tloc  # FIX: 순서 조정

    # 오른손 End Site 월드 좌표 반환 (프레임별)
    def right_hand_end_positions(self) -> np.ndarray:
        rw = self.joints_by_name.get("RightWrist")
        if rw is None or rw.end_site_offset is None:
            raise KeyError("RightWrist 또는 그 End Site를 찾을 수 없습니다.")
        pts = np.zeros((self.frames, 3), dtype=float)
        end_local = np.append(rw.end_site_offset, 1.0)
        for f in range(self.frames):
            Mw = self.world_matrix(rw, self.motion[f])
            p = Mw @ end_local
            pts[f, :] = p[:3]
        return pts

    # 왼손 End Site
    def left_hand_end_positions(self) -> np.ndarray:
        lw = self.joints_by_name.get("LeftWrist")
        if lw is None or lw.end_site_offset is None:
            raise KeyError("LeftWrist 또는 그 End Site를 찾을 수 없습니다.")
        pts = np.zeros((self.frames, 3), dtype=float)
        end_local = np.append(lw.end_site_offset, 1.0)
        for f in range(self.frames):
            Mw = self.world_matrix(lw, self.motion[f])
            p = Mw @ end_local
            pts[f, :] = p[:3]
        return pts

    # pelvis position(Xsens MVN 원본 그대로 값)
    def pelvis_positions(self) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("루트 조인트가 없습니다.")

        root = self.root
        pts = np.zeros((self.frames, 3), dtype=float)

        if not root.channels:
            raise RuntimeError("루트에 채널 정보가 없습니다.")

        base = root.chan_start

        # 루트 채널 중 X/Y/Zposition만 추출
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

    # 내부 유틸: 특정 조인트의 월드 좌표 (프레임별)
    def _joint_world_positions(self, j: Joint) -> np.ndarray:
        pts = np.zeros((self.frames, 3), dtype=float)
        origin = np.array([0.0, 0.0, 0.0, 1.0])
        for f in range(self.frames):
            M = self.world_matrix(j, self.motion[f])
            p = M @ origin
            pts[f, :] = p[:3]
        return pts

    # 오른쪽 어깨 월드 좌표 (프레임별)
    def right_shoulder_positions(self) -> np.ndarray:
        rs = self.joints_by_name.get("RightShoulder")
        if rs is None:
            raise KeyError("RightShoulder 조인트를 찾을 수 없습니다.")
        return self._joint_world_positions(rs)

    # 왼쪽 어깨 월드 좌표 (프레임별)
    def left_shoulder_positions(self) -> np.ndarray:
        ls = self.joints_by_name.get("LeftShoulder")
        if ls is None:
            raise KeyError("LeftShoulder 조인트를 찾을 수 없습니다.")
        return self._joint_world_positions(ls)


if __name__ == "__main__":

    # 사용법: python 이_스크립트.py input.bvh

    if len(sys.argv) != 2:
        script_name = os.path.basename(sys.argv[0])
        print(f"사용법: python {script_name} <input.bvh>")
        sys.exit(1)

    bvh_path = sys.argv[1]

    if not os.path.isfile(bvh_path):
        raise FileNotFoundError(f"지정한 BVH 파일을 찾을 수 없습니다: {bvh_path}")

    with open(bvh_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    bvh = BVH(text)

    # 2) 월드 좌표 계산
    right_end_xyz = bvh.right_hand_end_positions()
    left_end_xyz = bvh.left_hand_end_positions()
    pelvis_xyz = bvh.pelvis_positions()
    right_shoulder_xyz = bvh.right_shoulder_positions()
    left_shoulder_xyz = bvh.left_shoulder_positions()

    np.set_printoptions(precision=6, suppress=True)

    # 3) CSV 저장
    out_dir = os.path.dirname(os.path.abspath(bvh_path))

    # 입력 파일 이름에서 확장자 제거 후 .csv 붙이기
    base_name = os.path.splitext(os.path.basename(bvh_path))[0]
    out_csv_path = os.path.join(out_dir, base_name + ".csv")

    # 안전 확인: 프레임 수 일치 여부
    if not (
        pelvis_xyz.shape[0]
        == left_end_xyz.shape[0]
        == right_end_xyz.shape[0]
        == left_shoulder_xyz.shape[0]
        == right_shoulder_xyz.shape[0]
    ):
        raise ValueError(
            "Pelvis / Left / Right / Shoulder의 프레임 수가 서로 다릅니다."
        )

    # [pelvis(3), L_shoulder(3), R_shoulder(3), left(3), right(3)] -> (N, 15)
    merged = np.hstack(
        (pelvis_xyz, left_shoulder_xyz, right_shoulder_xyz, left_end_xyz, right_end_xyz)
    )

    header = (
        "pelvis_x,pelvis_y,pelvis_z,"
        "l_shoulder_x,l_shoulder_y,l_shoulder_z,"
        "r_shoulder_x,r_shoulder_y,r_shoulder_z,"
        "l_hand_x,l_hand_y,l_hand_z,"
        "r_hand_x,r_hand_y,r_hand_z"
    )

    np.savetxt(out_csv_path, merged, delimiter=",", header=header, comments="")

    print(f"\n[저장 완료] Pelvis / Left / Right CSV: {out_csv_path}")
