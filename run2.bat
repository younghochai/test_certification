@echo off
python calc_human.py --bg_pcd_path ./data/Calib.pcd --human_pcd_path ./data/test1.pcd --out_pcd_path ./data/result/test_result1.pcd
python calc_human.py --bg_pcd_path ./data/Calib.pcd --human_pcd_path ./data/test2.pcd --out_pcd_path ./data/result/test_result2.pcd
python calc_human.py --bg_pcd_path ./data/Calib.pcd --human_pcd_path ./data/test3.pcd --out_pcd_path ./data/result/test_result3.pcd
python calc_human.py --bg_pcd_path ./data/Calib.pcd --human_pcd_path ./data/test4.pcd --out_pcd_path ./data/result/test_result4.pcd

python calc_upperbody.py --human_pcd_path ./data/result/test_result1.pcd
python calc_upperbody.py --human_pcd_path ./data/result/test_result2.pcd
python calc_upperbody.py --human_pcd_path ./data/result/test_result3.pcd
python calc_upperbody.py --human_pcd_path ./data/result/test_result4.pcd

python convert_coordinate.py --src1 ./data/result/pelvis-test_result1.csv --src2 ./data/result/pelvis-test_result2.csv --out ./data/result/position_vector_01.csv
python convert_coordinate.py --src1 ./data/result/pelvis-test_result1.csv --src2 ./data/result/pelvis-test_result3.csv --out ./data/result/position_vector_02.csv
python convert_coordinate.py --src1 ./data/result/pelvis-test_result1.csv --src2 ./data/result/pelvis-test_result4.csv --out ./data/result/position_vector_03.csv

python patch_bvh_all.py ./data/BVH/input.bvh --frames 1500 3000 --pelvis_csv_01 ./data/result/position_vector_01.csv --pelvis_csv_02 ./data/result/position_vector_02.csv --pelvis_csv_03 ./data/result/position_vector_03.csv --pelvis_only

python endSite.py ./data/BVH/input_patched.bvh
python L2distance_with_csv.py ./data/BVH/input.bvh --frames 1500 3000
python L2distance_with_csv.py ./data/BVH/input_patched.bvh --frames 1500 3000
python visualization.py --midsphere --frames 1500 3000 --input_bvh ./data/BVH/input.bvh --patched_bvh ./data/BVH/input.bvh
python visualization.py --midsphere --frames 1500 3000 --input_bvh ./data/BVH/input.bvh --patched_bvh ./data/BVH/input_patched.bvh
pause