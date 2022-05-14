from compas.datastructures import Mesh
from compas.geometry import Pointcloud
from compas_view2.app import App


def add_to_viewer(elements: list, viewer):
    for element in elements:
        viewer.add(element)


def compas_show(data: dict, dist=3):
    pointclouds = []
    pc_dict = {}
    meshes = []


    if "keypoints" in data.keys():
        for pointcloud in data["keypoints"].values():
            pc_dict["points"] = pointcloud
            pointclouds.append(Pointcloud.from_data(pc_dict))

    if "fragments" in data.keys():
        for fragment in data["fragments"].values():
            meshes.append(fragment)

    viewer = App()

    add_to_viewer(pointclouds + meshes, viewer)
    viewer.view.camera.distance = dist
    viewer.show()

def compas_show_matches():
    #TODO: implement
    pass
