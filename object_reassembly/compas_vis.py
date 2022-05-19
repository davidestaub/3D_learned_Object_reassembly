from compas.colors import Color
from compas.geometry import Pointcloud
from compas_view2.app import App


def add_to_viewer(elements: list, viewer, colors=None):
    for i, element in enumerate(elements):
        viewer.add(element, color=colors[i][1] if colors else None)


def compas_show(data: dict, dist=3):
    pointclouds = []
    pc_dict = {}
    meshes = []
    lines = []

    if "keypoints" in data.keys():
        for pointcloud in data["keypoints"].values():
            pc_dict["points"] = pointcloud
            pointclouds.append(Pointcloud.from_data(pc_dict))

    if "fragments" in data.keys():
        for fragment in data["fragments"].values():
            meshes.append(fragment)

    if "lines" in data.keys():
        lines = data["lines"]
        print(lines)

    viewer = App()

    colors = [
                 ('orange', Color.orange()),
                 ('yellow', Color.yellow()),
                 ('green', Color.green()),
                 ('red', Color.red()),
                 ('cyan', Color.cyan()),
                 ('blue', Color.blue()),
                 ('violet', Color.violet()),
                 ('pink', Color.pink()),
                 ('brown', Color.brown())
             ][:len(meshes)]

    for i, (name, c) in enumerate(colors):
        print(f"{i}: {name}, {c}")
    add_to_viewer(pointclouds + meshes, viewer, colors * 2)

    add_to_viewer(lines, viewer)

    viewer.view.camera.distance = dist
    viewer.show()


def compas_show_matches():
    # TODO: implement
    pass
