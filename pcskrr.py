import numpy as np
import open3d as o3d
from pc_skeletor import LBC

pcd = o3d.io.read_point_cloud('human_only.pcd')

lbc = LBC(point_cloud=pcd,
          down_sample=0.008)
lbc.extract_skeleton()
lbc.extract_topology()

# Debug/Visualization
lbc.visualize()
lbc.export_results('./output')
lbc.animate(init_rot=np.asarray([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
            steps=300,
            output='./output')