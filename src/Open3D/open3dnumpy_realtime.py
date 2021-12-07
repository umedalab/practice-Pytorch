# https://github.com/isl-org/Open3D/issues/609

import copy
import numpy as np
import open3d as o3d

def play_motion(list_of_pcds: []):
    play_motion.vis = o3d.visualization.Visualizer()
    play_motion.index = 0

    def reset_motion(vis):
        play_motion.index = 0
        pcd.points = tensor_to_pcd(list_of_pcds[0]).points
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()
        time.sleep(.25)
        vis.register_animation_callback(forward)
        return False

    def backward(vis):
        pm = play_motion

        if pm.index > 0:
            pm.index -= 1
            pcd.points = tensor_to_pcd(list_of_pcds[pm.index]).points
            time.sleep(.05)
            vis.update_geometry()
            vis.poll_events()
            vis.update_renderer()
        else:
            vis.register_animation_callback(forward)

    def forward(vis):
        pm = play_motion
        if pm.index < len(list_of_pcds) - 1:
            pm.index += 1
            pcd.points = tensor_to_pcd(list_of_pcds[pm.index]).points
            time.sleep(.05)
            vis.update_geometry()
            vis.poll_events()
            vis.update_renderer()
        else:
            # vis.register_animation_callback(reset_motion)
            vis.register_animation_callback(backward)
        return False

    # Geometry of the initial frame
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(list_of_pcds[0].reshape(-1, 3))
    orange = np.array([255, 0, 0])
    pcd.colors = o3d.utility.Vector3dVector(np.ones(list_of_pcds[0].reshape(-1, 3).shape) * orange)

    # Initialize Visualizer and start animation callback
    vis = play_motion.vis
    vis.create_window()
    ctr = vis.get_view_control()
    ctr.rotate(0, -50)
    vis.add_geometry(pcd)
    vis.register_animation_callback(forward)
    vis.run()
    vis.destroy_window()

xyz = np.random.randint(5, size=(2, 3))
play_motion([xyz])
print('hello')