from typing import Dict, Any

from compas.colors import Color
from compas.geometry import Pointcloud
from compas_view2.app import App


def add_to_viewer(elements: Dict[int, Any], viewer: App, colors=None):
    for idx, element in elements.items():
        if colors:
            color = colors[idx][1]
            pointcolor = color.darkened(50)
        else:
            color = None
            pointcolor = None
        viewer.add(element, pointcolor=pointcolor, facecolor=color, pointsize=8)


def compas_show(keypoints: Dict[int, Pointcloud] = None, fragments=None, lines=None, dist=3):
    cline = 0.1647, 0.7176, 0.7921
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

    for i, (name, c) in enumerate(colors):
        print(f"{i}: {name}, {c}")

    viewer = App(show_grid=False)
    if keypoints:
        add_to_viewer(keypoints, viewer, colors=colors)
    if fragments:
        add_to_viewer(fragments, viewer, colors=colors)

    line_color = cline
    if lines:
        for l in lines:
            viewer.add(l, color=line_color, linewidth=3)

    viewer.view.camera.distance = dist
    viewer.show()
