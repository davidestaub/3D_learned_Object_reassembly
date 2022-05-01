from compas.datastructures import Mesh
from compas.geometry import Pointcloud
from compas_view2.app import App


def add_to_viewer(elements: list, viewer):
    for element in elements:
        viewer.add(element)


def flatten_list(l_list:list):
    flat_list = []
    for _list in l_list:
        flat_list += _list
    return flat_list


def show(data: dict):
    pointclouds = []
    pc_dict = {}
    meshes = []
    viewer = App()

    if "keypoints" in data.keys():
        for pointcloud in data["keypoints"]:
            pc_dict["points"] = pointcloud
            pointclouds.append(Pointcloud.from_data(pc_dict))

    if "fragments" in data.keys():
        for fragment in data["fragments"]:
            meshes.append(fragment)

    add_to_viewer(pointclouds + meshes, viewer)
    viewer.view.camera.distance = 3
    viewer.show()
