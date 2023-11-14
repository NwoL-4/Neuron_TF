import gc
import os
from os import chdir, makedirs

from tqdm import tqdm

from numpy import array, zeros, append, float64, ndarray, sqrt, int8, int64, zeros_like, linspace, meshgrid, save, savez
from scipy.constants import speed_of_light, elementary_charge, Boltzmann, electron_mass, neutron_mass, proton_mass

import data

if __name__ == '__main__':

    paths: str = 'C:\\Vsykoe\\Neuron\\Dataset'
    makedirs(paths, exist_ok=True)
    chdir(paths)

    ro: float = -0.1

    charge: float64 = elementary_charge

    mass = electron_mass

    wight: float64 = float64(1e-4)
    height: float64 = float64(1e-4)
    length: float64 = float64(1e-4)

    len_database: int64 = int64(5)
    number_particle = 10_000
#                       x    y    z
    grid_step = array([100, 100, 10])

    grid_e_x = linspace(0, wight, grid_step[0])
    grid_e_y = linspace(0, height, grid_step[1])
    grid_e_z = linspace(0, length, grid_step[1])

    enlargement_coefficient: float64 = float64(ro * wight * height * length / (charge * number_particle))



    for frame in tqdm(range(len_database)):
        coordinate = data.spawn_particle(number_added_particle=number_particle,
                                         height=height, width=wight, length=length)

        database = data.tensor_of_spatial_charge_fields(coordinate=coordinate,
                                                        enlargement_coefficient=enlargement_coefficient,
                                                        charge=charge,
                                                        grid_x=grid_e_x,
                                                        grid_y=grid_e_z,
                                                        grid_z=grid_e_z)

        len_db = len(os.listdir())
        savez(file=f'{len_db + 1}', coordinate=coordinate, dataset=database)
