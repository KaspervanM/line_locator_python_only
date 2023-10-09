from data_utils.data_generation import generate_data, LinearRNG, GaussianRNG, GrayscaleNumberGenerator, save_dataset
import math
import numpy as np

dataset_size = 1000
image_size = (28, 28)
image_nr_background_colors = LinearRNG(1, 5)
image_background_grays = GrayscaleNumberGenerator(GaussianRNG(0.5, 0.1))
seed = 1
diagonal = math.hypot(*image_size)
line_min_length = 4 / diagonal
line_nr_colors = LinearRNG(1, 5)
line_grays = GrayscaleNumberGenerator(GaussianRNG(0.4, 0.05))
line_nr_widths = LinearRNG(1, 5)
line_widths = GaussianRNG(1 / diagonal, 0.5 / diagonal, limit_left=0, limit_right=4 / diagonal)
prescale_image_size = (LinearRNG(image_size[0], image_size[0] * 20), LinearRNG(image_size[1], image_size[1] * 20))

dataset_list = generate_data(dataset_size, image_size, image_nr_background_colors, image_background_grays, seed,
                             line_min_length, line_nr_colors, line_grays, line_nr_widths, line_widths,
                             prescale_image_size=prescale_image_size)

save_dataset(dataset_list, f"data/dataset-{dataset_size}_size-{image_size[0]}x{image_size[1]}"
                           f"_bg-{image_background_grays.number_generator.mean:.3f}"
                           f"±{image_background_grays.number_generator.standard_deviation:.3f}_seed-{seed}"
                           f"_line-{line_grays.number_generator.mean:.3f}"
                           f"±{line_grays.number_generator.standard_deviation:.3f}"
                           f"_width-{line_widths.mean:.3f}"
                           f"±{line_widths.standard_deviation:.3f}")
