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
def image_to_stl(file_name, base, x_scale, y_scale, z_scale, keep_zeroes, keep_transparent):
    image = cv2.imread(file_name, -1)

    base_height = base

    # Bitmap denoting which pixels are ignored (i.e. non-solid) vs. not-ignored
    contours_mask = np.ones(image.shape[0:2], dtype=image.dtype)
    if len(image.shape) == 3 and image.shape[2] == 4:
        # Ignore fully transparent pixels
        if keep_transparent:
            contours_mask[np.where(image[:, :, 3] == 0)] = 0
        else:
            contours_mask[np.where(image[:, :, 3] != 255)] = 0

    # Find contours to generate side walls for mesh
    image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    if not keep_zeroes:
        contours_mask[np.where(image == 0)] = 0 # Ignore zero-valued pixels
        base_height -= 1
    contours = list(cv2.findContours(contours_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0])

    # Top & Bottom
    top_bottom_faces = 4 * image.shape[0] * image.shape[1]  # Number of top/bottom faces
    if not keep_zeroes:
        top_bottom_faces -= 2 * np.count_nonzero(image == 0)
    meshed = mesh.Mesh(np.zeros(top_bottom_faces, dtype=mesh.Mesh.dtype))

    index = 0
    has_holes = True if (contours_mask == 0).any() else False
    if not has_holes:
        for i in range(image.shape[0] - 1):
            for j in range(image.shape[1] - 1):
                # 4 triangles (2 top, 2 bottom) for every 4-pixel square
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
                # Only generate triangles with non-ignored corners
                corners = [
                    contours_mask[i, j],
                    contours_mask[i+1, j],
                    contours_mask[i, j+1],
                    contours_mask[i+1, j+1]
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
    contour_faces = 0   # Number of side faces
    for i in range(len(contours)):
        contours[i] = np.append(np.squeeze(contours[i], axis=1), contours[i][0], axis=0)
        contour_faces += 2 * contours[i].shape[0]
    sides = mesh.Mesh(np.zeros(contour_faces, dtype=mesh.Mesh.dtype))

    index = 0
    for contour in contours:
        # Create two triangles for every pair of adjacent points in a contour
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

    # Combine top/bottom & side triangle sets
    meshed.data = np.concatenate((meshed.data, sides.data))
    meshed.vectors[:, :, 0] *= x_scale
    meshed.vectors[:, :, 1] *= y_scale
    meshed.vectors[:, :, 2] *= z_scale

    return meshed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates a 3D model based on an image heightmap.")
    parser.add_argument('image',
                        help='File name of input image')
    parser.add_argument('output', nargs='?', const="output.stl", default="output.stl",
                        help='File name of output 3D model (Default: "output.stl")')

    parser.add_argument('-b', '--base', type=float, nargs='?', const=0, default=0,
                    help='Height of model base (Default = 0)')
    parser.add_argument('--x-scale', type=float, nargs='?', const=1.0, default=1.0,
                    dest='x_scale', help='X scale of generated model')
    parser.add_argument('--y-scale', type=float, nargs='?', const=1.0, default=1.0,
                    dest='y_scale', help='Y scale of generated model')
    parser.add_argument('--z-scale', type=float, nargs='?', const=1.0, default=1.0,
                    dest='z_scale', help='Z scale of generated model')

    parser.add_argument('--keep-zeroes', dest='keep_zeroes', action='store_true',
                        help='Retain zero-valued pixels within the generated mesh')
    parser.add_argument('--keep-transparent', dest='keep_transparent', action='store_true',
                        help='Retain partially transparent pixels within the generated mesh')
    
    args = parser.parse_args();
    meshed = image_to_stl(args.image, args.base, args.x_scale, args.y_scale, args.z_scale, 
        args.keep_zeroes, args.keep_transparent)

    meshed.save(args.output)
