import sys
import numpy as np
from skimage import io

def mirror(image, axis):
    h, w = image.shape[:2]
    if axis in ['d', 'cd']:
        mirrored_image = np.zeros((w, h, 3), dtype=image.dtype)
    else:
        mirrored_image = np.zeros_like(image)

    if axis == 'h':
        for i in range(h):
            for j in range(w):
                mirrored_image[i, j] = image[h - 1 - i, j]
    elif axis == 'v':
        for i in range(h):
            for j in range(w):
                mirrored_image[i, j] = image[i, w - 1 - j]
    elif axis == 'd':
        for i in range(h):
            for j in range(w):
                mirrored_image[j, i] = image[i, j]
    elif axis == 'cd':
        for i in range(h):
            for j in range(w):
                mirrored_image[w - 1 - j, h - 1 - i] = image[i, j]
    return mirrored_image

def rotate(image, direction, angle):
    if direction == 'ccw':
        angle = -angle
    angle = angle % 360
    if angle == 90:
        rotated_image = np.zeros((image.shape[1], image.shape[0], 3), dtype=image.dtype)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rotated_image[j, image.shape[0] - 1 - i] = image[i, j]
    elif angle == 180:
        rotated_image = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rotated_image[image.shape[0] - 1 - i, image.shape[1] - 1 - j] = image[i, j]
    elif angle == 270:
        rotated_image = np.zeros((image.shape[1], image.shape[0], 3), dtype=image.dtype)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rotated_image[image.shape[1] - 1 - j, i] = image[i, j]
    else:
        rotated_image = image
    return rotated_image

def extract(image, left_x, top_y, width, height):
    h, w = image.shape[:2]
    extracted_image = np.zeros((height, width, 3), dtype=image.dtype)
    for i in range(height):
        for j in range(width):
            y = top_y + i
            x = left_x + j
            if 0 <= y < h and 0 <= x < w:
                extracted_image[i, j] = image[y, x]
    return extracted_image

def autocontrast(image):
    min_pixel = np.min(image)
    max_pixel = np.max(image)

    if min_pixel == max_pixel:
        # Изображение полностью однородно, контрастирование невозможно
        result_image = np.zeros_like(image)
    else:
        result_image = (image - min_pixel) * (1.0 / (max_pixel - min_pixel)) * 255.0
    return result_image.astype(np.uint8)    

def compute_horizontal_variation(image):
    variation = np.sum(np.abs(np.diff(image, axis=0)))
    return variation

def fixinterlace(image):
    swapped_image = np.copy(image)
    swapped_image[::2, :], swapped_image[1::2, :] = image[1::2, :], image[::2, :]

    original_variation = compute_horizontal_variation(image)
    swapped_variation = compute_horizontal_variation(swapped_image)

    if swapped_variation < original_variation:
        return swapped_image
    else:
        return image

def main():
    command = sys.argv[1]
    input_file = sys.argv[-2]
    output_file = sys.argv[-1]

    image = io.imread(input_file)

    if command == 'mirror':
        axis = sys.argv[2]
        result_image = mirror(image, axis)
    elif command == 'rotate':
        direction = sys.argv[2]
        angle = int(sys.argv[3])
        result_image = rotate(image, direction, angle)
    elif command == 'extract':
        left_x = int(sys.argv[2])
        top_y = int(sys.argv[3])
        width = int(sys.argv[4])
        height = int(sys.argv[5])
        result_image = extract(image, left_x, top_y, width, height)
    elif command == 'autocontrast':
        result_image = autocontrast(image)
    elif command == 'fixinterlace':
        result_image = fixinterlace(image)
    else:
        print("Unknown command")
        return

    io.imsave(output_file, result_image)

if __name__ == "__main__":
    main()