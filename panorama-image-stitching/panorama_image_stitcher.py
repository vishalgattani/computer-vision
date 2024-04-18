import cv2
import argparse
import cv2
from pathlib import Path

def detectAndDescribe(image, method=None):
    """Compute key points and feature descriptors using an specific method

    Args:
        image (_type_): image
        method (_type_, optional): 'sift', 'surf', 'orb' descriptors. Defaults to None.

    Returns:
        _type_: keypoints and features
    """
    assert method is not None, "Values are: 'sift', 'surf', 'orb'"

    # detect and extract features from the image
    if method == 'sift':
        descriptor = cv2.xfeatures2d.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()

    # get keypoints and descriptors
    (kps, features) = descriptor.detectAndCompute(image, None)

    return kps, features

def createMatcher(method,crossCheck):
    """Create and return a Matcher Object

    Args:
        method (_type_): 'sift', 'surf', 'orb' descriptors.
        crossCheck (_type_): crossCheck bool parameter indicates whether the two
            features have to match each other to be considered valid. In other words,
            for a pair of features (f1, f2) to considered valid, f1 needs to match f2
            and f2 has to match f1 as the closest match as well. This procedure ensures
            a more robust set of matching features and is described in the original SIFT paper.

    Returns:
        _type_: matcher
    """
    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf

if __name__ == "__main__":
    image_directory = Path("./scottsdale")
    print(list(image_directory.glob('*.jpg')))
    ...