from ..config.configLoader import GlobalConfig
if GlobalConfig.backend == "jax":
    import jax.numpy as np
if GlobalConfig.backend == "numpy":
    import numpy as np
if GlobalConfig.backend == "tensor":
    import tensorflow as np



def generate_sampling_points(x_existing, max_samples=30):
    """
    Generate intelligent sampling points based on existing x and y values.

    Parameters:
    - x_existing (array-like): The existing x-values that have already been sampled.
    - y_existing (array-like, optional): The existing y-values that correspond to x_existing.
    - max_samples (int): The maximum number of additional samples to generate.

    Returns:
    - new_sampling_points (array): The new x-values where sampling should be conducted.
    """

    # Sort the existing x-values
    x_existing = np.sort(x_existing)
    # Calculate the existing gaps between x-values
    gaps = np.diff(x_existing)
    # Normalize the gaps to their sum
    normalized_gaps = gaps / np.sum(gaps)
    # Calculate the number of points to insert in each gap
    points_to_insert = np.round(normalized_gaps * max_samples).astype(int)
    # Generate new sampling points
    new_points = []
    for x1, x2, n_points in zip(x_existing[:-1], x_existing[1:], points_to_insert):
        new_points.extend(
            np.linspace(x1, x2, n_points + 2)[1:-1]
        )  # Exclude the end points
    # Ensure that the number of new points does not exceed max_samples
    new_points = new_points[:max_samples]
    return np.array(new_points)


def filter_extreme_outliers(x, y, threshold=3.0):
    """
    Filters extreme outliers from the data based on Z-score.

    Parameters:
    - x, y (array-like): The input x and y data arrays.
    - threshold (float): The Z-score threshold for outlier removal.

    Returns:
    - filtered_x, filtered_y (arrays): The x and y data arrays with outliers removed.
    """
    mean = np.mean(y)
    std_dev = np.std(y)
    z_scores = [(elem - mean) / std_dev for elem in y]
    filtered_x = np.array(
        [x_val for x_val, z in zip(x, z_scores) if np.abs(z) < threshold]
    )
    filtered_y = np.array(
        [y_val for y_val, z in zip(y, z_scores) if np.abs(z) < threshold]
    )
    return filtered_x, filtered_y


def select_chunk_around_max(x, y, zoom_factor=0.5):
    """
    Selects a chunk of data centered around the maximum value within a dynamically mapped interval.
    The function ensures that the range of selected x's is centered around the maximum.

    Parameters:
    - x, y (array-like): The input x and y data arrays.
    - zoom_factor (float): A factor between 0 and 1 indicating the proportion of data to select around the max.

    Returns:
    - selected_x, selected_y (arrays): The selected chunks of x and y data.
    """
    max_idx = np.argmax(y)

    # Calculate the number of elements to select on each side
    left_count = int((max_idx) * zoom_factor)
    right_count = int((len(y) - max_idx - 1) * zoom_factor)

    # Determine start and end indices
    start_idx = max(0, max_idx - left_count)
    end_idx = min(len(y), max_idx + right_count + 1)  # +1 to include the max_idx itself

    # Balance sides to make it centered around max
    max_iterations = min(
        max_idx, len(y) - max_idx
    )  # The maximum number of iterations needed to balance
    for _ in range(max_iterations):
        if end_idx - max_idx < max_idx - start_idx and end_idx < len(y):
            end_idx += 1
        elif end_idx - max_idx > max_idx - start_idx and start_idx > 0:
            start_idx -= 1
        else:
            break  # Break the loop if it's balanced or can't be balanced further

    selected_x = x[start_idx:end_idx]
    selected_y = y[start_idx:end_idx]

    return selected_x, selected_y


def find_knee_index(y):
    # Calculate the second derivative
    second_derivative = np.diff(np.diff(y))
    # Find the index at which the second derivative changes significantly
    # Here, I'm assuming that the "knee" is where the second derivative is minimum, indicating the curve is flattening out
    knee_index = np.argmin(np.abs(second_derivative))
    return knee_index


def identify_curve_shape(x, y, verbose=False):
    n = len(y)
    if n < 3:  # Insufficient data to make a conclusion
        return {"Insufficient Data": True}, None, None

    # Find the index and value of the maximum
    max_index = np.argmax(y)
    max_x = x[max_index]

    # Calculate the "distance" on either side of the max value in terms of y
    left_distance = np.sum(np.abs(np.diff(y[: max_index + 1])))
    right_distance = np.sum(np.abs(np.diff(y[max_index:])))

    # Calculate ratio to decide the type of the curve
    min_distance = min(left_distance, right_distance)
    max_distance = max(left_distance, right_distance)

    if max_distance == 0:
        return {"Insufficient Data": True}, None, None

    center_ratio = min_distance / max_distance

    # Dynamic thresholds based on center_ratio
    semi_threshold = 0.3
    half_threshold = 0.1

    # Initialize curve_shape dictionary
    curve_shape = {
        "Healthy Parabola": False,
        "Left Semi-Parabola": False,
        "Right Semi-Parabola": False,
        "Left Half-Parabola": False,
        "Right Half-Parabola": False,
    }

    # Classify curve shape based on distance ratio
    if center_ratio > semi_threshold:
        shape = "Healthy Parabola"
    elif center_ratio > half_threshold:
        shape = (
            "Left Semi-Parabola"
            if left_distance < right_distance
            else "Right Semi-Parabola"
        )
    else:
        shape = (
            "Left Half-Parabola"
            if left_distance < right_distance
            else "Right Half-Parabola"
        )

    if shape == "Left Half-Parabola" or shape == "Right Half-Parabola":
        # Compute the differences to find where the curve stagnates
        differences = np.abs(np.diff(y))

        # Identify knee of the curve
        knee_index = find_knee_index(y)

        # Decide whether to use max_index or knee_index
        if knee_index < max_index and shape == "Left Half-Parabola":
            new_point = knee_index
        elif knee_index > max_index and shape == "Right Half-Parabola":
            new_point = knee_index
        else:
            new_point = max_index  # Fallback to maximum point if knee is not in the expected direction

        # Update max_x to be the x at the new point
        max_x = x[new_point]

    curve_shape[shape] = True

    # Recalculate new_range, lb_centered, and ub_centered with the updated max_x
    new_range = min(max_x - min(x), max(x) - max_x)
    lb_centered = max_x - new_range
    ub_centered = max_x + new_range

    # Update dictionary and calculate symmetric adjustment

    if verbose:
        print(f"\033[94mThe curve is identified as a \033[92m{shape}\033[94m.")
        print(
            f"\033[94mNew bounds centered around the maximum are \033[93m[{lb_centered}, {ub_centered}]\033[0m"
        )

    return curve_shape, max_x, lb_centered, ub_centered


# x = [1, 2, 3, 4, 5, 6, 7]
# y = [1, 2, 5, 9, 5, 2, 1]
# shape_dict, max_x, lb,lu = identify_curve_shape(x, y, verbose=True)

# prf_res_p = prf_res['profiles']
# profile  = prf_res_a[psi_key]
# print(identify_curve_shape(prf_res_p[psi_key], profile,verbose = True))  # Output: "Healthy Parabola"
