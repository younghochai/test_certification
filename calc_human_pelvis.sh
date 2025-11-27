python calc_human.py --bg_pcd_path ./data/test0.pcd --human_pcd_path ./data/test1.pcd --out_pcd_path ./data/result/test_result1.pcd
python calc_upperbody.py --human_pcd_path ./data/result/test_result1.pcd

python patch_bvh_angles.py ./data/BVH/input.bvh --Hips --RightCollar --RightShoulder --RightElbow --RightWrist
python endSite.py ./data//BVH/input_patched.bvh
python L2distance_with_csv.py ./data/BVH/input_patched.csv