import cv2
import numpy as np
import matplotlib.pyplot as plt

def _preserve_dtype(img_float, original_dtype):
    """
    After doing math in float form, clip the values back into the valid range
    for the original image type and convert back to that type.

    This is important because TUM-VI images may be stored as 16-bit grayscale.
    If we do operations like brightness scaling or noise addition, pixel values
    can go below 0 or above the allowed max value, so we need to clamp them.
    """
    if np.issubdtype(original_dtype, np.integer):
        # For integer images like uint8 or uint16, get valid min/max
        info = np.iinfo(original_dtype)
        img_float = np.clip(img_float, info.min, info.max)
    else:
        # For float images, assume values should stay in [0, 1]
        img_float = np.clip(img_float, 0.0, 1.0)

    # Convert back to the original image type
    return img_float.astype(original_dtype)


def apply_occlusion(img, severity=0.25, mode="right_to_left", value=0):
    """
    Simulate a camera occlusion by covering part of the image with a constant value.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image.
    severity : float
        Fraction of image width or height to block.
        Example: 0.25 means block 25% of the image.
    mode : str
        Direction of occlusion:
        - "right_to_left"
        - "left_to_right"
        - "top_to_bottom"
        - "bottom_to_top"
        - "center_box"
    value : int
        Pixel value used for the blocked region.
        Usually 0 for black occlusion.

    Returns
    -------
    out : np.ndarray
        Degraded image with the occlusion applied.
    """
    # Make a copy so we do not modify the original image
    out = img.copy()

    # Get image height and width
    h, w = out.shape[:2]

    # Horizontal occlusion cases
    if mode in ["right_to_left", "left_to_right"]:
        occ_w = int(w * severity)  # width of blocked region

        if mode == "right_to_left":
            # Block the right side of the image
            out[:, w - occ_w:] = value
        else:
            # Block the left side of the image
            out[:, :occ_w] = value

    # Vertical occlusion cases
    elif mode in ["top_to_bottom", "bottom_to_top"]:
        occ_h = int(h * severity)  # height of blocked region

        if mode == "top_to_bottom":
            # Block the top portion
            out[:occ_h, :] = value
        else:
            # Block the bottom portion
            out[h - occ_h:, :] = value

    # Center box occlusion case
    elif mode == "center_box":
        box_w = int(w * severity)
        box_h = int(h * severity)

        # Compute center box coordinates
        x0 = (w - box_w) // 2
        y0 = (h - box_h) // 2

        # Fill the center region with the occlusion value
        out[y0:y0 + box_h, x0:x0 + box_w] = value

    return out


def apply_gaussian_blur(img, ksize=7, sigma=2.0):
    """
    Simulate loss of sharpness using Gaussian blur.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    ksize : int
        Blur kernel size. Must be odd for OpenCV.
        Larger values produce stronger blur.
    sigma : float
        Standard deviation of the Gaussian kernel.

    Returns
    -------
    blurred : np.ndarray
        Blurred image.
    """
    # OpenCV requires odd kernel sizes, so fix it if needed
    if ksize % 2 == 0:
        ksize += 1

    # Apply Gaussian blur
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)


def apply_brightness_drop(img, factor=0.5):
    """
    Simulate an underexposed or darkened image.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    factor : float
        Brightness scaling factor.
        1.0 = no change, 0.5 = half brightness.

    Returns
    -------
    out : np.ndarray
        Darkened image in the original dtype.
    """
    original_dtype = img.dtype

    # Convert to float so multiplication is safe and precise
    out = img.astype(np.float32) * factor

    # Clip and cast back to original type
    return _preserve_dtype(out, original_dtype)


def apply_additive_gaussian_noise(img, sigma=20.0):
    """
    Simulate sensor noise by adding zero-mean Gaussian noise.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    sigma : float
        Noise standard deviation.
        For 16-bit images, this can be much larger than for 8-bit images.

    Returns
    -------
    out : np.ndarray
        Noisy image in the original dtype.
    """
    original_dtype = img.dtype

    # Convert to float for noise addition
    out = img.astype(np.float32)

    # Create random Gaussian noise with same shape as image
    noise = np.random.normal(0, sigma, out.shape).astype(np.float32)

    # Add noise to image
    out = out + noise

    # Clip and convert back
    return _preserve_dtype(out, original_dtype)

def example_show_all_degradations_on_tum_image(image_path):
    """
    Show original + all degradations in one subplot figure.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"Could not load image at: {image_path}")

    occluded = apply_occlusion(img, severity=0.35)
    blurred = apply_gaussian_blur(img, ksize=19, sigma=100)
    darkened = apply_brightness_drop(img, factor=0.5)
    noisy = apply_additive_gaussian_noise(img, sigma=5000.0)

    images = [
        ("Original", img),
        ("Occlusion", occluded),
        ("Blur", blurred),
        ("Darkened", darkened),
        ("Noise", noisy),
    ]

    # Use one common display scale for all images
    vmin = 0
    if np.issubdtype(img.dtype, np.integer):
        vmax = np.iinfo(img.dtype).max
    else:
        vmax = 1.0

    plt.figure(figsize=(12, 6))

    for i, (title, im) in enumerate(images):
        plt.subplot(2, 3, i + 1)
        plt.imshow(im, cmap='gray', vmin=vmin, vmax=vmax)
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

example_show_all_degradations_on_tum_image(
    "/Users/sammorrisroe/Desktop/dataset-room1_512_16/mav0/cam1/data/1520530308199447626.png"
)