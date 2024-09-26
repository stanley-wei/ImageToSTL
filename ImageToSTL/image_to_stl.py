import argparse
import cv2
import math
import numpy as np
from stl import mesh
import sys


'''
    TODO:
        - Switch from numpy_stl to pyvista
        - Reduce triangulation of flat surfaces
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates a 3D model based on an image heightmap.")
    parser.add_argument('image',
                        help='File name of input image')
    parser.add_argument('output', nargs='?', const="output.stl", default="output.stl",
                        help='File name of output 3D model (Default: "output.stl")')
    parser.add_argument('-b', '--base', type=float, nargs='?', const=0, default=0,
                    help='Height of model base (Default = 0)')
    parser.add_argument('--ignore-zeroes', dest='ignore_zeroes', action='store_false',
                        help='Treat zero-valued pixels as non-solid (Default: True)')
    
    args = parser.parse_args();
    image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)

    base_height = args.base

    ignore_zeroes = args.ignore_zeroes
    if ignore_zeroes:
        ret, contours_mask = cv2.threshold(image,1,255,cv2.THRESH_BINARY)
        if not (contours_mask == 0).any():
            ignore_zeroes = False
        base_height += 1
    else:
        contours_mask = np.ones(image.shape, image.dtype)
    contours = list(cv2.findContours(contours_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0])

    # Top & Bottom
    top_bottom_faces = 4 * image.shape[0] * image.shape[1]  # Number of top/bottom faces
    if ignore_zeroes:
        top_bottom_faces -= 2 * np.count_nonzero(image == 0)
    meshed = mesh.Mesh(np.zeros(top_bottom_faces, dtype=mesh.Mesh.dtype))

    index = 0
    if not ignore_zeroes:
        for i in range(image.shape[0] - 1):
            for j in range(image.shape[1] - 1):
                meshed.vectors[index] = np.array([
                    [i, j+1, image[i, j+1]],
                    [i, j, image[i, j]],
                    [i+1, j, image[i+1, j]]
                ])
                meshed.vectors[index+1] = np.array([
                    [i+1, j+1, image[i+1, j+1]],
                    [i, j+1, image[i, j+1]],
                    [i+1, j, image[i+1, j]]
                ])
                meshed.vectors[index+2] = np.array([
                    [i, j+1, -base_height],
                    [i, j, -base_height],
                    [i+1, j, -base_height]
                ])
                meshed.vectors[index+3] = np.array([
                    [i+1, j+1, -base_height],
                    [i, j+1, -base_height],
                    [i+1, j, -base_height]
                ])
                index += 4
    else:
        for i in range(image.shape[0] - 1):
            for j in range(image.shape[1] - 1):
                corners = [
                    image[i, j],
                    image[i+1, j],
                    image[i, j+1],
                    image[i+1, j+1]
                ]
                if corners.count(0) > 1:
                    continue
                elif corners.count(0) == 1:
                    if corners[0] == 0:
                        meshed.vectors[index] = np.array([
                            [i+1, j+1, image[i+1, j+1]],
                            [i, j+1, image[i, j+1]],
                            [i+1, j, image[i+1, j]]
                        ])
                        meshed.vectors[index+1] = np.array([
                            [i+1, j+1, -base_height],
                            [i, j+1, -base_height],
                            [i+1, j, -base_height]
                        ])
                    elif corners[1] == 0:
                        meshed.vectors[index] = np.array([
                            [i+1, j+1, image[i+1, j+1]],
                            [i, j+1, image[i, j+1]],
                            [i, j, image[i, j]]
                        ])
                        meshed.vectors[index+1] = np.array([
                            [i+1, j+1, -base_height],
                            [i, j+1, -base_height],
                            [i, j, -base_height]
                        ])
                    elif corners[2] == 0:
                        meshed.vectors[index] = np.array([
                            [i+1, j, image[i+1, j]],
                            [i+1, j+1, image[i+1, j+1]],
                            [i, j, image[i, j]]
                        ])
                        meshed.vectors[index+1] = np.array([
                            [i+1, j, -base_height],
                            [i+1, j+1, -base_height],
                            [i, j, -base_height]
                        ])
                    else:
                        meshed.vectors[index] = np.array([
                            [i, j+1, image[i, j+1]],
                            [i, j, image[i, j]],
                            [i+1, j, image[i+1, j]]
                        ])
                        meshed.vectors[index+1] = np.array([
                            [i, j+1, -base_height],
                            [i, j, -base_height],
                            [i+1, j, -base_height]
                        ])
                    index += 2
                else:
                    meshed.vectors[index] = np.array([
                        [i, j+1, image[i, j+1]],
                        [i, j, image[i, j]],
                        [i+1, j, image[i+1, j]]
                    ])
                    meshed.vectors[index+1] = np.array([
                        [i+1, j+1, image[i+1, j+1]],
                        [i, j+1, image[i, j+1]],
                        [i+1, j, image[i+1, j]]
                    ])
                    meshed.vectors[index+2] = np.array([
                        [i, j+1, -base_height],
                        [i, j, -base_height],
                        [i+1, j, -base_height]
                    ])
                    meshed.vectors[index+3] = np.array([
                        [i+1, j+1, -base_height],
                        [i, j+1, -base_height],
                        [i+1, j, -base_height]
                    ])
                    index += 4
    meshed.data = meshed.data[0:index]

    # Sides
    contour_faces = 0                                       # Number of side faces
    for i in range(len(contours)):
        contours[i] = np.append(np.squeeze(contours[i]), contours[i][0], axis=0)
        contour_faces += 2 * contours[i].shape[0]
    sides = mesh.Mesh(np.zeros(contour_faces, dtype=mesh.Mesh.dtype))

    index = 0
    for contour in contours:
        for i in range(len(contour)-1):
            sides.vectors[index] = np.array([
                [contour[i][1], contour[i][0], -base_height],
                [contour[i+1][1], contour[i+1][0], -base_height],
                [contour[i][1], contour[i][0], image[contour[i][1], contour[i][0]]]
            ])
            sides.vectors[index+1] = np.array([
                [contour[i][1], contour[i][0], image[contour[i][1], contour[i][0]]],
                [contour[i+1][1], contour[i+1][0], -base_height],
                [contour[i+1][1], contour[i+1][0], image[contour[i+1][1], contour[i+1][0]]]
            ])
            index += 2

    meshed.data = np.concatenate((meshed.data, sides.data))
    meshed.save(args.output)
