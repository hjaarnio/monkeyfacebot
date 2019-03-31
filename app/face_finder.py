# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for dlib-based alignment."""

import dlib
import numpy as np
import random
import nudged
from skimage import transform
from skimage import morphology
from PIL import Image
from PIL import ImageDraw
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.color import rgb2gray
from skimage.transform import match_histograms


class AlignDlib:
    """
    Use `dlib's landmark estimation <http://blog.dlib.net/2014/08/real-time-face-pose-estimation.html>`_ to align faces.

    The alignment preprocess faces for input into a neural network.
    Faces are resized to the same size (such as 96x96) and transformed
    to make landmarks (such as the eyes and nose) appear at the same
    location on every image.

    Normalized landmarks:

    .. image:: ../images/dlib-landmark-mean.png
    """

    #: Landmark indices.
    INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
    OUTER_EYES_AND_NOSE = [36, 45, 33]

    def __init__(self, facePredictor):
        """
        Instantiate an 'AlignDlib' object.

        :param facePredictor: The path to dlib's
        :type facePredictor: str
        """
        assert facePredictor is not None

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(facePredictor)

    def getAllFaceBoundingBoxes(self, rgbImg):
        """
        Find all face bounding boxes in an image.

        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :return: All face bounding boxes in an image.
        :rtype: dlib.rectangles
        """
        assert rgbImg is not None

        try:
            return self.detector(rgbImg, 1)
        except Exception as e:
            print("Warning: {}".format(e))
            # In rare cases, exceptions are thrown.
            return []

    def getLargestFaceBoundingBox(self, rgbImg, skipMulti=False):
        """
        Find the largest face bounding box in an image.

        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :param skipMulti: Skip image if more than one face detected.
        :type skipMulti: bool
        :return: The largest face bounding box in an image, or None.
        :rtype: dlib.rectangle
        """
        assert rgbImg is not None

        faces = self.getAllFaceBoundingBoxes(rgbImg)
        if (not skipMulti and len(faces) > 0) or len(faces) == 1:
            return max(faces, key=lambda rect: rect.width() * rect.height())
        else:
            return None

    def findLandmarks(self, rgbImg, bb):
        """
        Find the landmarks of a face.

        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :param bb: Bounding box around the face to find landmarks for.
        :type bb: dlib.rectangle
        :return: Detected landmark locations.
        :rtype: list of (x,y) tuples
        """
        assert rgbImg is not None
        assert bb is not None

        points = self.predictor(rgbImg, bb)
        return list(map(lambda p: (p.x, p.y), points.parts()))


finder = AlignDlib("shape_predictor_5_face_landmarks.dat")


def coords_from_pil_image(image):
    np_img = np.array(image)
    return finder.getAllFaceBoundingBoxes(np_img)


def landmarks_from_pil_image(image):
    coords = coords_from_pil_image(image)
    np_img = np.array(image)
    return [
        finder.findLandmarks(np_img, coord)
        for coord in coords
    ]


def generate_histogram(image, mask):
    mask_pixels = mask.getdata()
    image_pixels = [
        imgage_pixel
        for (imgage_pixel, mask_pixel)
        in zip(image.getdata(), mask_pixels)
        if mask_pixel > 0
    ]
    histogram_bg = Image.new('RGB', image.size)
    histogram_bg_pixels = histogram_bg.load()
    for i in range(histogram_bg.size[0]):
        for j in range(histogram_bg.size[1]):
            histogram_bg_pixels[i, j] = image_pixels[random.randint(0, len(image_pixels)-1)]
    return Image.composite(image, histogram_bg, mask)


def swap_faces(img1, img2):
    # get landmark points from the images
    landmarks1 = landmarks_from_pil_image(img1)
    landmarks2 = landmarks_from_pil_image(img2)
    img1_index = random.randint(0, len(landmarks1) - 1)
    img2_index = random.randint(0, len(landmarks2) - 1)
    landmarks1 = landmarks1[img1_index]
    landmarks2 = landmarks2[random.randint(0, len(landmarks2) - 1)]
    bb = coords_from_pil_image(img1)[img1_index]
    bb = coords_from_pil_image(img2)[img2_index]
    # calculate transformation matrix to go from one set of points to the other
    trans_matrix1 = nudged.estimate(landmarks2, landmarks1).get_matrix()

    trans1 = transform.ProjectiveTransform(matrix=np.array(trans_matrix1))
    # transform the images to be on top of each other
    img1_transformed = transform.warp(
        np.array(img1),
        trans1,
        output_shape=np.array(img2).shape[:2]
    ).dot(255).astype('uint8')

    s = np.linspace(0, 2*np.pi, 400)
    height = (bb.bottom()-bb.top())
    rad_y = height/2*1.2
    x = (bb.left()+bb.right())/2 + (bb.right()-bb.left())/2*np.cos(s)
    y = (bb.top()+bb.bottom())/2 + rad_y*np.sin(s) - height*0.15
    init = np.array([x, y]).T
    #init = init.dot(trans_matrix1)
    shape = active_contour(
        gaussian(rgb2gray(img1_transformed), 3),
        init, alpha=0.08, beta=1, gamma=0.001, max_iterations=height*0.1, max_px_move=1
    )
    #shape = init
    output = Image.fromarray(img1_transformed)
    mask = Image.new('L', output.size)
    draw = ImageDraw.Draw(mask)
    draw.polygon([tuple(coord)[:2] for coord in shape], fill="white")
    #draw = ImageDraw.Draw(img2)
    #draw.polygon([tuple(coord)[:2] for coord in shape], outline="red")
    #return img2

    # Match histograms on the selected areas
    print(morphology.disk(int(height*0.1)))
    hist_mask = Image.fromarray(morphology.dilation(np.array(mask), morphology.disk(int(height*0.1))))
    img2_histogram = generate_histogram(img2, hist_mask)
    img1_histogram = generate_histogram(Image.fromarray(img1_transformed), hist_mask)
    matched = Image.fromarray(match_histograms(
        np.asarray(img1_histogram),
        np.asarray(img2_histogram),
        multichannel=True
    ))

    mask = gaussian(np.array(mask), height*0.1)
    mask = Image.fromarray(mask.dot(255).astype('uint8'))
    output = Image.composite(matched, img2, mask)
    return output
