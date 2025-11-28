import argparse
import numpy as np
import os


def load_xyz_csv(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    data = np.loadtxt(path, delimiter=",", skiprows=1)

    # 데이터가 1행일 수도 있으므로 2D로 강제
    data = np.atleast_2d(data)

    if data.shape[1] < 3:
        raise ValueError(f"CSV에 최소 3개 컬럼(x,y,z)가 필요합니다: {path}")

    # x,y,z만 사용 (앞에 다른 컬럼이 없다면 그대로 3개일 것)
    return data[:, :3]


def pelvis_to_wanted(vec_p: np.ndarray) -> np.ndarray:
    """
    pelvis 좌표계 벡터 (x_p, y_p, z_p)를 wanted 좌표계로 변환.
    (x_w, y_w, z_w) = (-z_p, -y_p, -x_p)
    vec_p: (N, 3)
    """
    x_p = vec_p[:, 0]
    y_p = vec_p[:, 1]
    z_p = vec_p[:, 2]

    x_w = -z_p
    y_w = -y_p
    z_w = -x_p

    return np.stack([x_w, y_w, z_w], axis=1)


def main():
    parser = argparse.ArgumentParser(
        description="pelvis-test_result1.csv → pelvis-test_result2.csv 벡터를 wanted 좌표계로 변환하여 저장"
    )
    parser.add_argument(
        "--src1",
        type=str,
        required=True,
        help="pelvis-test_result1.csv 경로"
    )
    parser.add_argument(
        "--src2",
        type=str,
        required=True,
        help="pelvis-test_result2.csv 경로"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="wanted.csv",
        help="결과를 저장할 wanted.csv 경로 (기본값: ./position_vector.csv)"
    )

    args = parser.parse_args()

    p1 = load_xyz_csv(args.src1)  # (N, 3)
    p2 = load_xyz_csv(args.src2)  # (N, 3)

    if p1.shape != p2.shape:
        raise ValueError(
            f"두 CSV의 shape가 다릅니다: {p1.shape} vs {p2.shape}"
        )

    # pelvis 좌표계에서의 벡터 (2 - 1)
    diff_p = p2 - p1  # (N, 3)

    # wanted 좌표계로 변환
    diff_w = pelvis_to_wanted(diff_p)  # (N, 3)

    # 저장 (단위: cm)
    header = "x_w_cm,y_w_cm,z_w_cm"
    np.savetxt(
        args.out,
        diff_w,
        delimiter=",",
        fmt="%.3f",
        header=header,
        comments=""
    )

    print(f"저장 완료: {args.out}")
    print("첫 번째 벡터 (wanted 좌표계, cm):", diff_w[0])


if __name__ == "__main__":
    main()
