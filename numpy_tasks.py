"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    matrices_array = np.asarray(matrices)
    vectors_array = np.asarray(vectors)
    if vectors_array.ndim == 2:
        vectors_array = vectors_array[..., np.newaxis]
    return np.sum(matrices_array @ vectors_array, axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return (np.asarray(matrix) > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [np.unique(row).tolist() for row in np.asarray(matrix)]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    array = np.asarray(matrix)
    return [np.unique(array[:, column]).tolist() for column in range(array.shape[1])]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    generator = np.random.default_rng(seed)
    matrix = generator.normal(loc=mean, scale=std, size=(rows, columns))
    return MatrixStatistics(
        matrix=matrix,
        row_means=np.mean(matrix, axis=1),
        column_means=np.mean(matrix, axis=0),
        row_variances=np.var(matrix, axis=1),
        column_variances=np.var(matrix, axis=0),
    )


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    row_indices, column_indices = np.indices((rows, columns))
    return np.where((row_indices + column_indices) % 2 == 0, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[...] = background_color

    top = image_height // 2 - height // 2
    left = image_width // 2 - width // 2
    bottom = top + height
    right = left + width
    top_clip, bottom_clip = max(top, 0), min(bottom, image_height)
    left_clip, right_clip = max(left, 0), min(right, image_width)
    if top_clip < bottom_clip and left_clip < right_clip:
        image[top_clip:bottom_clip, left_clip:right_clip] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[...] = background_color
    if semi_axis_x <= 0 or semi_axis_y <= 0:
        return image

    y, x = np.indices((image_height, image_width))
    center_x = image_width // 2
    center_y = image_height // 2
    inside = ((x - center_x) / semi_axis_x) ** 2 + ((y - center_y) / semi_axis_y) ** 2 <= 1
    image[inside] = shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    array = np.asarray(values)
    local_maxima = np.where((array[1:-1] > array[:-2]) & (array[1:-1] > array[2:]))[0] + 1
    local_minima = np.where((array[1:-1] < array[:-2]) & (array[1:-1] < array[2:]))[0] + 1
    moving_average = np.convolve(array, np.ones(window) / window, mode="valid")
    return TimeSeriesStatistics(
        mean=float(np.mean(array)),
        variance=float(np.var(array)),
        std=float(np.std(array)),
        local_maxima_indices=local_maxima,
        local_minima_indices=local_minima,
        moving_average=moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    labels_array = np.asarray(labels, dtype=int)
    if labels_array.ndim != 1:
        raise ValueError("labels must be a one-dimensional vector")
    if np.any(labels_array < 0):
        raise ValueError("labels must be non-negative")
    if class_count is None:
        class_count = int(labels_array.max()) + 1 if labels_array.size else 0
    if class_count < 0 or np.any(labels_array >= class_count):
        raise ValueError("class_count must cover every label")
    result = np.zeros((labels_array.size, class_count), dtype=int)
    result[np.arange(labels_array.size), labels_array] = 1
    return result
