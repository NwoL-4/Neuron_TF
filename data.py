from numba import njit, prange, jit, float64, int64

from numpy import zeros, array, pi, ndarray, int64, vectorize
from numpy.random import uniform
from scipy.constants import epsilon_0
from tqdm import tqdm


@njit(cache=True, parallel=True)
def tensor_of_spatial_charge_fields(coordinate,
                                    enlargement_coefficient,
                                    charge,
                                    grid_x,
                                    grid_y,
                                    grid_z
                                    ) -> ndarray:
    coefficient = 1 / (4 * pi * epsilon_0)

    numbers_particle = coordinate.shape[1]

    tensor = zeros(shape=(3, len(grid_x), len(grid_y), len(grid_z)), dtype=float64)

    for i in prange(grid_x.shape[0]):
        for j in prange(grid_y.shape[0]):
            for k in prange(grid_z.shape[0]):
                field = zeros(3)
                grid_note = array([grid_x[i], grid_y[j], grid_z[k]])

                for particle in prange(numbers_particle):
                    field += enlargement_coefficient * charge * coefficient / (
                            ((coordinate[0, particle] - grid_note[0]) ** 2 +
                             (coordinate[1, particle] - grid_note[1]) ** 2 +
                             (coordinate[2, particle] - grid_note[2]) ** 2) ** (3 / 2)) * array(
                        [coordinate[0, particle] - grid_note[0],
                         coordinate[1, particle] - grid_note[1],
                         coordinate[2, particle] - grid_note[2]
                         ])

                tensor[:, i, j, k] = field
    return tensor


@njit(parallel=True, cache=True)
def spawn_particle(number_added_particle: int64,  # Число добавляемых частиц
                   height: float64,  # Высота
                   width: float64,  # Ширина
                   length: float64,  # Средняя скорость влета
                   ) -> ndarray:
    new_coordinate = zeros(shape=(3, number_added_particle), dtype=float)

    new_coordinate[0, :] = uniform(low=0, high=width, size=number_added_particle)
    new_coordinate[1, :] = uniform(low=0, high=height, size=number_added_particle)
    new_coordinate[2, :] = uniform(low=0, high=length, size=number_added_particle)

    return new_coordinate
