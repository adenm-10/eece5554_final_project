import os
import cv2
import numpy as np


def apply_gaussian_blur(img, ksize=7, sigma=100):
    """
    Simulate loss of sharpness using Gaussian blur.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    ksize : int
        Blur kernel size. Must be odd for OpenCV.
    sigma : float
        Standard deviation of the Gaussian kernel.

    Returns
    -------
    blurred : np.ndarray
        Blurred image.
    """
    if ksize % 2 == 0:
        ksize += 1

    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)


def blur_first_200_images(input_folder, output_folder, ksize=7, sigma=100):
    """
    Load the first 200 images from input_folder, apply Gaussian blur,
    and save them into output_folder.

    Parameters
    ----------
    input_folder : str
        Path to the folder containing the original images.
    output_folder : str
        Path to the folder where blurred images will be saved.
    ksize : int
        Gaussian blur kernel size.
    sigma : float
        Gaussian blur sigma.
    """
    os.makedirs(output_folder, exist_ok=True)

    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    image_files = sorted(
        [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]
    )

    first_200 = image_files[:200]

    if len(first_200) == 0:
        raise FileNotFoundError(f"No image files found in: {input_folder}")

    for filename in first_200:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Skipping unreadable file: {filename}")
            continue

        blurred = apply_gaussian_blur(img, ksize=ksize, sigma=sigma)

        success = cv2.imwrite(output_path, blurred)
        if success:
            print(f"Saved: {output_path}")
        else:
            print(f"Failed to save: {output_path}")


input_folder = "/Users/sammorrisroe/Desktop/dataset-room1_512_16/mav0/cam1/data/"
output_folder = "/Users/sammorrisroe/Desktop/dataset-room1_ks19/mav0/cam1/data/"

blur_first_200_images(
    input_folder=input_folder,
    output_folder=output_folder,
    ksize=19,
    sigma=100
)