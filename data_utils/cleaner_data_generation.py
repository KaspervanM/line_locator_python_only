#!/usr/bin/env python
from sys import exit
import math
import random
from abc import ABC, abstractmethod
import warnings
import numpy as np
import cv2
import matplotlib.pyplot as plt


class NumberGeneratorInterface(ABC):
    """Abstract class for a number generator.

    Attributes:
        limit_left: The left limit of the distribution.
        limit_right: The right limit of the distribution.
    """
    limit_left = None
    limit_right = None

    @abstractmethod
    def generate(self):
        """Generates a random number.

        Returns:
            The generated random number.
        """
        pass


class NumberGenerator(NumberGeneratorInterface):
    """Number generator that generates a constant number or a number from a list of numbers in order.

    Attributes:
        number: The number or list of numbers to generate.
    """
    _index = -1

    def __init__(self, number):
        """Inits the number generator with a number or list of numbers.

        Args:
            number: The number or list of numbers to generate.
        """
        if isinstance(number, list):
            self._index = 0
        self.number = number
        self.limit_left = number
        self.limit_right = number

    def generate(self):
        """Generates the (next) number.

        Returns:
            The generated number.

        Raises:
            IndexError: If the number generator has no more numbers to generate.
        """
        if self._index >= 0:
            try:
                number = self.number[self._index]
            except IndexError:
                raise IndexError("The number generator has no more numbers to generate.")
            self._index += 1
            return number
        return self.number


class LinearRNG(NumberGeneratorInterface):
    """Number generator according to the uniform distribution.

    Attributes:
        limit_left: The left limit of the distribution.
        limit_right: The right limit of the distribution.
    """

    def __init__(self, limit_left=-1.0, limit_right=1.0):
        """Inits the number generator with the left and right limit of the distribution.

        Args:
            limit_left: The left limit of the distribution.
            limit_right: The right limit of the distribution.

        Raises:
            ValueError: If the left limit is larger than the right limit.
        """
        if limit_left > limit_right:
            raise ValueError("The left limit should be smaller than the right limit")
        self.limit_left = limit_left
        self.limit_right = limit_right

    def generate(self):
        """Generates a random number according to the uniform distribution.

        Returns:
            The generated random number.
        """
        return random.uniform(self.limit_left, self.limit_right)


class GaussianRNG(NumberGeneratorInterface):
    """Number generator according to the gaussian distribution.

    Attributes:
        mean: The mean of the distribution.
        standard_deviation: The standard deviation of the distribution.
        limit_left: The left limit of the distribution.
        limit_right: The right limit of the distribution.
    """

    def __init__(self, mean=0.0, standard_deviation=1.0, limit_left=None, limit_right=None):
        """Inits the number generator with the mean and standard deviation of the distribution.

        Args:
            mean: The mean of the distribution.
            standard_deviation: The standard deviation of the distribution.
            limit_left: The left limit of the distribution.
            limit_right: The right limit of the distribution.

        Raises:
            ValueError: If the left limit is larger than the right limit.
            ValueError: If the standard deviation is smaller than 0.
        """
        if limit_left is not None and limit_right is not None and limit_left > limit_right:
            raise ValueError("The left limit should be smaller than the right limit")
        if standard_deviation <= 0:
            raise ValueError("The standard deviation should be larger than 0")
        self.mean = mean
        self.standard_deviation = standard_deviation
        self.limit_left = limit_left if limit_left is not None else -np.inf
        self.limit_right = limit_right if limit_right is not None else np.inf

    def generate(self):
        """Generates a random number according to the gaussian distribution.

        Returns:
            The generated random number.
        """
        random_number = random.gauss(self.mean, self.standard_deviation)
        return max(self.limit_left, min(self.limit_right, random_number))


class UniformCategoricalRNG(NumberGeneratorInterface):
    """Number generator according to the uniform categorical distribution.

    Attributes:
        categories: A list of categories.
        limit_left: The left limit of the distribution.
        limit_right: The right limit of the distribution.
    """

    def __init__(self, categories, limit_left=None, limit_right=None):
        """Inits the number generator with the categories.

        Args:
            categories: A list of categories.
            limit_left: The left limit of the distribution.
            limit_right: The right limit of the distribution.

        Raises:
            ValueError: If the left limit is larger than the right limit.
            ValueError: If the categories are not a list.
            ValueError: If the categories are an empty list.
        """
        if limit_left is not None and limit_right is not None and limit_left > limit_right:
            raise ValueError("The left limit should be smaller than the right limit")
        if not isinstance(categories, list):
            raise ValueError("The categories should be a list")
        if len(categories) == 0:
            raise ValueError("The categories should not be an empty list")
        self.categories = categories
        self.limit_left = limit_left if limit_left is not None else -np.inf
        self.limit_right = limit_right if limit_right is not None else np.inf

    def generate(self):
        """Generates a random number according to the uniform categorical distribution.

        Returns:
            The generated random number.
        """
        return random.choice(self.categories)


class GrayscaleNumberGenerator(NumberGeneratorInterface):
    def __init__(self, number_generator):
        if not isinstance(number_generator, NumberGeneratorInterface):
            raise ValueError("The number generator should be an instance of NumberGeneratorInterface")
        if isinstance(number_generator, GaussianRNG):
            if number_generator.mean < 0 or number_generator.mean > 1:
                warnings.warn("A mean outside [0,1] will result in many values outside of the range [0, 1]" +
                              " which will therefore be clipped. This will result in a buildup at 0 and 1.")
            if number_generator.standard_deviation >= 1:
                warnings.warn(
                    "A standard deviation of 1 or larger will result in many values outside of the range [0, 1]" +
                    " and will therefore be clipped. This will result in a buildup at 0 and 1.")
        if isinstance(number_generator, NumberGenerator):
            if isinstance(number_generator.number, list):
                for number in number_generator.number:
                    if number < 0 or number > 1:
                        raise ValueError("The list of background_grayscale values should only contain values in [0, 1]")
            if isinstance(number_generator.number, float):
                if number_generator.number < 0 or number_generator.number > 1:
                    raise ValueError("The background_grayscale value should be in [0, 1]")
        if isinstance(number_generator, UniformCategoricalRNG):
            for category in number_generator.categories:
                if category < 0 or category > 1:
                    raise ValueError("The list of background_grayscale values should only contain values in [0, 1]")

        self.number_generator = number_generator
        self.number_generator.limit_left = 0.0
        self.number_generator.limit_right = 1.0

    def generate(self):
        """Generates a number according to number generator.

        Returns:
            The generated number.
        """
        return self.number_generator.generate()


class AngleNumberGenerator(NumberGeneratorInterface):
    def __init__(self, number_generator):
        if not isinstance(number_generator, NumberGeneratorInterface):
            raise ValueError("The number generator should be an instance of NumberGeneratorInterface")
        if isinstance(number_generator, GaussianRNG):
            if number_generator.mean < -0.5 or number_generator.mean > 0.5:
                warnings.warn("A mean outside [-0.5, 0.5] will result in many values outside of the range [-0.5, 0.5]" +
                              " which will therefore be clipped. This will result in a buildup at -0.5 and 0.5.")
            if number_generator.standard_deviation >= 1:
                warnings.warn(
                    "A standard deviation of 1 or larger will result in many values outside of the range [-0.5, 0.5]" +
                    " and will therefore be clipped. This will result in a buildup at -0.5 and 0.5.")
        self.number_generator = number_generator
        self.number_generator.limit_left = -0.5
        self.number_generator.limit_right = 0.5

    def generate(self):
        """Generates a random number according to number generator.

        Returns:
            The generated number.
        """
        return self.number_generator.generate()


class Gradient:
    """Gradient of values by position interpolation.

    Attributes:
        values: A list of values.
    """

    def __init__(self, values: (NumberGeneratorInterface, float, list[float]),
                 nr_of_colors: (NumberGeneratorInterface, int) = None):
        """Inits the gradient with a list of values.

        Args:
            values: A number generator, a float or a list of floats.
            nr_of_colors: A number generator for the number of colors or an integer.

        Raises:
            ValueError: If the values are a number generator and the number of colors is not provided.
            ValueError: If the values are a list and the list is empty.
            ValueError: If the values are a list and the list contains values that are not floats.
            ValueError: If the values are not a number generator, a float or a list of floats.
        """
        if isinstance(values, NumberGeneratorInterface):
            tmp_values = []
            if nr_of_colors is None:
                raise ValueError("If the values are a number generator, the number of colors should be provided")
            if isinstance(nr_of_colors, NumberGeneratorInterface):
                nr_of_colors = int(nr_of_colors.generate())
            if nr_of_colors < 1:
                raise ValueError("The number of colors should be at least 1")
            for i in range(nr_of_colors):
                tmp_values.append(values.generate())
            values = tmp_values
        elif isinstance(values, list):
            if len(values) == 0:
                raise ValueError("The list of values should contain at least one value")
            for value in values:
                if not isinstance(value, float):
                    raise ValueError("The list of values should only contain floats")
        elif isinstance(values, float):
            values = [values]
        else:
            print(type(values))
            raise ValueError("The values should be a number generator, a float or a list of floats")
        self.values = values

    def get_value(self, position):
        """Returns the interpolated value at a given position.

        Args:
            position: A float between 0 and 1 (can be out of range).

        Returns:
            The interpolated color at the given position.
            If the position is out of range, the gradient will behave as if mirrored after itself.
        """
        if len(self.values) == 1:
            return self.values[0]

        position = abs(position)
        if position > 1:
            position = position % 1 if position % 2 < 1 else (1 - position % 1)

        first_color_index = int(position * (len(self.values) - 1))
        second_color_index = first_color_index + 1
        if second_color_index > len(self.values) - 1:
            second_color_index = len(self.values) - 2

        first_color = self.values[first_color_index]
        second_color = self.values[second_color_index]
        position_between = position * (len(self.values) - 1) - first_color_index
        color = first_color * (1 - position_between) + second_color * position_between

        return color


class GrayscaleGradient(Gradient):
    """Gradient of background_grayscale values between 0 and 1.

    Attributes:
        values: A list of background_grayscale values in [0, 1]
    """

    def __init__(self, values: (GrayscaleNumberGenerator, float, list[float]),
                 nr_of_colors: (NumberGeneratorInterface, int) = None):
        """Inits the gradient with a list of background_grayscale values.

        Args:
            values: A number generator, a float or a list of floats.
            nr_of_colors: A number generator for the number of colors or an integer.

        Raises:
            ValueError: If the values are or contain a number not in [0, 1].
            ValueError: If the values are a number generator and the number of colors is not provided.
            ValueError: If the values are a list and the list is empty.
            ValueError: If the values are a list and the list contains values that are not floats.
            ValueError: If the values are not a number generator, a float or a list of floats.
        """
        if isinstance(values, float):
            if values < 0 or values > 1:
                raise ValueError("The background_grayscale value should be in [0, 1]")
        if isinstance(values, list):
            for value in values:
                if value < 0 or value > 1:
                    raise ValueError("The list of background_grayscale values should only contain values in [0, 1]")
        super().__init__(values, nr_of_colors)


def generate_noise_image(size: tuple[int, int], grayscale_generator: (GrayscaleNumberGenerator, list[float], float)):
    """Generates an image with noise according to some number generator.

    Args:
        size: A tuple of the size of the image (width, height).
        grayscale_generator: The number generator or float to generate the background_grayscale values.

    Returns:
        The generated image.

    Raises:
        ValueError: If the grayscale generator is a float and not in [0, 1].
    """
    if isinstance(grayscale_generator, float):
        grayscale_generator = GrayscaleNumberGenerator(NumberGenerator(grayscale_generator))
    if isinstance(grayscale_generator, list):
        grayscale_generator = GrayscaleNumberGenerator(UniformCategoricalRNG(grayscale_generator))
    image = np.zeros((size[1], size[0]), np.float16)
    for i in range(size[1]):
        for j in range(size[0]):
            color = grayscale_generator.generate()
            image[i, j] = color

    return image


def line_on_image(background_image: np.ndarray[np.dtype[np.floating]],
                  left_coordinate: tuple[int, int],
                  right_coordinate: tuple[int, int], grayscale_gradient: GrayscaleGradient,
                  width_gradient: GrayscaleGradient):
    left_coordinate = np.array(left_coordinate)
    right_coordinate = np.array(right_coordinate)
    coordinates = np.linspace(left_coordinate, right_coordinate, np.max(np.abs(left_coordinate - right_coordinate)) + 1)

    for i in range(len(coordinates)):
        color = grayscale_gradient.get_value(i / len(coordinates)) * 255
        width = max(1, width_gradient.get_value(i / len(coordinates)) * math.hypot(background_image.shape[1],
                                                                                   background_image.shape[0]))
        cv2.circle(background_image, np.round(coordinates[i]).astype(np.uint8), math.floor(width / 2), color, -1)

    return background_image.astype(np.float16) / 255


def show_image(image):
    """Shows an image.

    Args:
        image: The image to show.
    """
    plt.imshow(image, cmap="gray", norm=plt.Normalize(vmin=0, vmax=1))
    plt.axis('off')
    plt.show()


# diagonal = math.hypot(28, 28)
# my_img = line_on_image(np.zeros((28, 28)), (0, 10), (20, 5), GrayscaleGradient([0.5, 0.1, 1.0]),
#                        GrayscaleGradient([0.5 / diagonal, 8 / diagonal]))
# show_image(my_img)
# exit(0)


def scale_image(image: np.ndarray[np.dtype[np.floating]], size: tuple[int, int]):
    """Scales an image to a given size.

    Args:
        image: The image to scale.
        size: The size to scale the image to.

    Returns:
        The scaled image.
    """
    image = (image * 255).astype(np.uint8)
    return cv2.resize(image, size).astype(np.float16) / 255


def scale_coordinates(coordinates, original_size, new_size):
    """Scales coordinates so the coordinates of a point on the original image correspond with those on the new image.

    Args:
        coordinates: The coordinates to scale.
        original_size: The original size of the image.
        new_size: The new size of the image.

    Returns:
        The scaled coordinates.
    """
    original_width, original_height = original_size
    new_width, new_height = new_size

    width_scale = new_width / original_width
    height_scale = new_height / original_height

    scaled_coordinates = (coordinates[0] * width_scale, coordinates[1] * height_scale)

    return scaled_coordinates


def generate_data(nr_of_images: (NumberGeneratorInterface, int), size: tuple[int, int],
                  background_grayscale: (GrayscaleNumberGenerator, float, list[float]), seed: int,
                  min_line_length: float, nr_of_line_colors: (NumberGeneratorInterface, int),
                  line_grayscale: (GrayscaleNumberGenerator, float, list[float]),
                  nr_of_widths: (NumberGeneratorInterface, int),
                  line_width_gradient: (GrayscaleNumberGenerator, int),
                  prescale_image_size: (
                          NumberGeneratorInterface,
                          tuple[NumberGeneratorInterface, NumberGeneratorInterface],
                          tuple[int, int]) = None):
    if isinstance(nr_of_images, NumberGeneratorInterface):
        nr_of_images = nr_of_images.generate()
    if nr_of_images < 1:
        raise ValueError("The number of images should be at least 1")

    if prescale_image_size is not None:
        if isinstance(prescale_image_size, tuple):
            if isinstance(prescale_image_size[0], int) and isinstance(prescale_image_size[1], int):
                if prescale_image_size[0] < 1 or prescale_image_size[1] < 1:
                    raise ValueError("The prescale image size should be at least 1")
                prescale_image_size = (
                    NumberGenerator(prescale_image_size[0]),
                    NumberGenerator(prescale_image_size[1])
                )
            elif isinstance(prescale_image_size[0], NumberGeneratorInterface) and isinstance(prescale_image_size[1],
                                                                                             NumberGeneratorInterface):
                if prescale_image_size[0].limit_left < 1 or prescale_image_size[1].limit_left < 1:
                    raise ValueError("The prescale image size should be at least 1")
        elif isinstance(prescale_image_size, NumberGeneratorInterface):
            if prescale_image_size.limit_left < 1:
                raise ValueError("The prescale image size should be at least 1")
            prescale_image_size = (prescale_image_size, prescale_image_size)
        else:
            raise ValueError("Prescale image size should be a number generator or a tuple of number generators or ints")
    else:
        prescale_image_size = (NumberGenerator(size[0]), NumberGenerator(size[1]))

    random.seed(seed)
    images = []
    for i in range(nr_of_images):
        prescaled_size = (int(prescale_image_size[0].generate()), int(prescale_image_size[1].generate()))
        image = generate_noise_image(prescaled_size, background_grayscale)

        x1 = y1 = x2 = y2 = 0
        while math.dist((x1, y1), (x2, y2)) < min_line_length:
            x1 = round(LinearRNG(0, prescaled_size[0]).generate())
            y1 = round(LinearRNG(0, prescaled_size[1]).generate())
            x2 = round(LinearRNG(0, prescaled_size[0]).generate())
            y2 = round(LinearRNG(0, prescaled_size[1]).generate())

        if x1 < x2:
            left_coordinate = (x1, y1)
            right_coordinate = (x2, y2)
        elif x2 < x1:
            left_coordinate = (x2, y2)
            right_coordinate = (x1, y1)
        elif y1 < y2:
            left_coordinate = (x1, y1)
            right_coordinate = (x2, y2)
        else:
            left_coordinate = (x2, y2)
            right_coordinate = (x1, y1)

        image = line_on_image((image * 255).astype(np.uint8), left_coordinate, right_coordinate,
                              GrayscaleGradient(line_grayscale, nr_of_line_colors),
                              GrayscaleGradient(line_width_gradient, nr_of_widths))

        left_coordinate = (left_coordinate[0] / prescaled_size[0], left_coordinate[1] / prescaled_size[1])
        left_coordinate = scale_coordinates(left_coordinate, prescaled_size, size)
        right_coordinate = (right_coordinate[0] / prescaled_size[0], right_coordinate[1] / prescaled_size[1])
        right_coordinate = scale_coordinates(right_coordinate, prescaled_size, size)
        image = scale_image(image, size)
        images.append((image, (*left_coordinate, *right_coordinate)))

    return images


def save_dataset(dataset, file_path):
    images = [item[0] for item in dataset]
    lines = [item[1] for item in dataset]

    file_path = file_path if file_path.endswith('.npz') else file_path + '.npz'
    np.savez_compressed(f'{file_path}', images=images, lines=lines)


def load_dataset(file_path):
    file_path = file_path if file_path.endswith('.npz') else file_path + '.npz'
    loaded_dataset = np.load(file_path)

    return loaded_dataset['images'], loaded_dataset['lines']
