import sys
import os
import numpy as np

def compare_r_hand(csv_path: str):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    # header 한 줄을 건너뛰고 데이터만 로드
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    if data.shape[0] < 3:
        raise ValueError("프레임 수가 너무 적습니다. 최소 3행 이상이어야 합니다.")

    # r_hand_x, r_hand_y, r_hand_z는 마지막 3개의 컬럼 (총 15개 컬럼 중 12, 13, 14번 인덱스)
    # header:
    # pelvis(3), l_shoulder(3), r_shoulder(3), l_hand(3), r_hand(3) = 15
    r_hand_cols = [-3, -2, -1]  # 혹은 [12, 13, 14]

    # 3행(실제 데이터 기준 두 번째 프레임: index 1)
    start_r_hand = data[1, r_hand_cols]

    # 마지막 행
    end_r_hand = data[-1, r_hand_cols]

    # L2 거리 (유클리드 거리)
    diff = end_r_hand - start_r_hand
    l2_dist = np.linalg.norm(diff)

    print("=== r_hand 위치 비교 ===")
    print(f"3행 r_hand (x, y, z): {start_r_hand}")
    print(f"마지막 행 r_hand (x, y, z): {end_r_hand}")
    print(f"L2 거리: {l2_dist:.6f} cm")

    return l2_dist

if __name__ == "__main__":
    if len(sys.argv) != 2:
        script_name = os.path.basename(sys.argv[0])
        print(f"사용법: python {script_name} <input.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    compare_r_hand(csv_path)
