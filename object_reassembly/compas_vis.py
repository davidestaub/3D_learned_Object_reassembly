from typing import Dict

from compas.colors import Color
from compas.geometry import Pointcloud
from compas_view2.app import App


def add_to_viewer(elements: list, viewer: App, colors=None):
    for i, element in enumerate(elements):
        if colors:
            color = colors[i][1]
            pointcolor = color.darkened(50)
        else:
            color = None
            pointcolor = None
        viewer.add(element, pointcolor=pointcolor, facecolor=color, pointsize=8)


def compas_show(keypoints: Dict[int, Pointcloud] = None, fragments=None, lines=None, dist=3):
    pointclouds = []
    meshes = []

    if keypoints:
        for pointcloud in keypoints.values():
            pointclouds.append(Pointcloud(pointcloud))

    if fragments:
        for fragment in fragments.values():
            meshes.append(fragment)

    colors = [
        ('orange', Color.orange()),
        ('yellow', Color.yellow()),
        ('green', Color.green()),
        ('red', Color.red()),
        ('cyan', Color.cyan()),
        ('blue', Color.blue()),
        ('violet', Color.violet()),
        ('pink', Color.pink()),
        ('brown', Color.brown()),
        ('grey', Color.grey()),
        ('mint', Color.mint()),
        ('olive', Color.olive())
    ]
    colors = colors[:len(meshes)]

    for i, (name, c) in enumerate(colors):
        print(f"{i}: {name}, {c}")

    viewer = App()
    add_to_viewer(pointclouds, viewer, colors=colors)
    add_to_viewer(meshes, viewer, colors=colors)
    if lines:
        print(lines)
        add_to_viewer(lines, viewer)

    viewer.view.camera.distance = dist
    viewer.show()


def compas_show_matches():
    # TODO: implement
    pass
