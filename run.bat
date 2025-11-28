@echo off
python calc_human.py --bg_pcd_path ./data/Calib.pcd --human_pcd_path ./data/test1.pcd --out_pcd_path ./data/result/test_result1.pcd
python calc_human.py --bg_pcd_path ./data/Calib.pcd --human_pcd_path ./data/test2.pcd --out_pcd_path ./data/result/test_result2.pcd

python calc_upperbody.py --human_pcd_path ./data/result/test_result1.pcd
python calc_upperbody.py --human_pcd_path ./data/result/test_result2.pcd

python convert_coordinate.py --src1 ./data/result/pelvis-test_result1.csv --src2 ./data/result/pelvis-test_result2.csv --out ./data/result/position_vector.csv

python patch_bvh.py ./data/BVH/input.bvh --pelvis_csv ./data/result/position_vector.csv --std 0.05 --Hips --Chest --RightCollar --RightShoulder --RightElbow --RightWrist
python endSite.py ./data//BVH/input_patched.bvh
python L2distance_with_csv.py ./data/BVH/input_patched.csv
pause