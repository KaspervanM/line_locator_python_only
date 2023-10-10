from data_utils.cleaner_data_generation import generate_data, LinearRNG, GaussianRNG, save_dataset
import math

dataset_size = 5000
image_size = (28, 28)
image_background_grays = GaussianRNG(0.5, 0.1, limit_left=0.45, limit_right=1)
seed = 2
diagonal = math.hypot(*image_size)
line_min_length = 4 / diagonal
line_nr_colors = LinearRNG(1, 5)
line_grays = GaussianRNG(0.3, 0.05, limit_left=0, limit_right=0.4)
line_nr_widths = LinearRNG(1, 5)
line_widths = GaussianRNG(1 / diagonal, 0.5 / diagonal, limit_left=0, limit_right=4 / diagonal)
prescale_image_size = image_size  # (LinearRNG(image_size[0], image_size[0] * 30), LinearRNG(image_size[1], image_size[1] * 30))
dataset_list = generate_data(dataset_size, image_size, image_background_grays, seed,
                             line_min_length, line_nr_colors, line_grays, line_nr_widths, line_widths,
                             prescale_image_size=prescale_image_size)

save_dataset(dataset_list, f"data/cleaner_dataset-{dataset_size}_size-{image_size[0]}x{image_size[1]}"
                           f"_bg-{image_background_grays.mean:.3f}"
                           f"±{image_background_grays.standard_deviation:.3f}"
                           f"({image_background_grays.limit_left:.3f}to{image_background_grays.limit_right:.3f})"
                           f"_seed-{seed}"
                           f"_line-{line_grays.mean:.3f}"
                           f"±{line_grays.standard_deviation:.3f}"
                           f"({line_grays.limit_left:.3f}to({line_grays.limit_right:.3f})"
                           f"_width-{line_widths.mean:.3f}"
                           f"±{line_widths.standard_deviation:.3f}")
