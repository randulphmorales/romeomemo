#!/usr/bin/env python
# coding=utf-8

import numpy as np
from romeomemo.utilities import projection

from scipy.stats import linregress
from scipy.spatial.distance import squareform, pdist

class Grid:

    def __init__(self, nx, nz, dx, dz, xmin, zmin):
        """
        Parameters
        ----------
        dx : float
            EASTERLY size of a gridcell in meters
        dy : float
            NORTHLY size of a gridcell in meters
        nx : int
            Number of cells in EASTERLY direction
        ny : int
            Number of cells in NORTHLY direction
        xmin : float
            EASTERLY distance of bottom left gridpoint in meters
        ymin : float
            NORTHLY distance of bottom left gridpoint in meters
        """

        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz
        self.xmin = xmin
        self.zmin = zmin

    def cell_corners(self, i, j):
        """Return the corners of the cell with indices (i,j).

        See also the docstring of Grid.cell_corners.

        Parameters
        ----------
        i : int
        j : int

        Returns
        -------
        tuple(np.array(shape=(4,), dtype=float),
              np.array(shape=(4,), dtype=float))
            Arrays containing the x and y coordinates of the corners

        """
        x1, z1 = [self.xmin + i * self.dx, self.zmin + j * self.dz]
        x2, z2 = [self.xmin + (i + 1) * self.dx, self.zmin + (j + 1) * self.dz]

        cell_x = np.array([x2, x2, x1, x1])
        cell_z = np.array([z2, z1, z1, z2])

        return cell_x, cell_z

    def xdist_range(self):
        """Return an array containing horizontal points on the grid.

        Returns
        -------
        np.array(shape=(nx,), dtype=float)
        """
        return np.array([self.xmin + i * self.dx + self.dx / 2 for i in range(self.nx)])

    def alt_range(self):
        """Return an array containing all altitude points on the grid.

        Returns
        -------
        np.array(shape=(nz,), dtype=float)
        """
        return np.array([self.zmin + i * self.dz + self.dz / 2 for i in range(self.nz)])

    def indices_of_point(self, x, z):
        """Return the index of each measurement point assigned to the
        target grid

        Parameters
        ----------
        x : float
           distance (m) from anchor point
        z : float
           altitude (m) from anchor point

        Returns
        -------
        tuple(int, int)
           index of the measurement point placed in the grid cell
        """

        indx = np.floor((x - self.xmin) / self.dx)
        indz = np.floor((z - self.zmin) / self.dz)

        if indx < 0 or indz < 0 or indx > self.nx - 1 or indz > self.nz -1:
            raise IndexError("Point lies outside the target grid!")

        return int(indx), int(indz)


    def square_matrix(self):
        
        range_x = self.xdist_range()
        range_z = self.alt_range()

        coords = []
        for x in range_x:
            for z in range_z:
                points = [x,z]
                coords.append(points)

        coods = np.asarray(coords)
        dist = pdist(coords)
        square_matrix = squareform(dist)

        return square_matrix



class ProcessCoordinates:

    def __init__(self, lon, lat):

        self.lon = lon
        self.lat = lat

    def regression_coeffs(self):

        x = np.array(self.lon)
        y = np.array(self.lat)
        m, b, r_value, p_value, std_err = linregress(x, y)

        self.m = m # slope of the regression line
        self.b = b # y-intercept of the regression line
        self.r_value = r_value

        return m, b, r_value

    def regression_vector(self):
        """Projects transect measurement points to a regression line

        Parameters
        ----------
        lon : float
            Longitude in WGS84
        lat : float
            Latitude in WGS84

        Returns
        -------
        tuple(float, float)
            contains the projected longitude and latitude into the regression line
        """

        x = np.array(self.lon)
        y = np.array(self.lat)

        m, b, r_value, p_value, std_err = linregress(x, y)

        perp_slope = -1 / m
        y_perp_int = -perp_slope * x + y
        proj_lon = (y_perp_int - b) / (m - perp_slope)
        proj_lat = (perp_slope * proj_lon + y_perp_int)

        return proj_lon, proj_lat


    def anchorpoint(self, proj_lon, proj_lat):
        """Determines the starting point of the transect

        Parameters
        ----------
        lon : float
            Longitude in WGS84
        lat : float
            Latitude in WGS84

        Returns
        -------
        tuple(float, float)
            contains the projected lon and projected lat to which the transect
            should start
        """

        self.proj_lon = proj_lon
        self.proj_lat = proj_lat

        lon_min = min(self.proj_lon)
        lon_max = max(self.proj_lon)

        lat_min = min(self.proj_lat)
        lat_max = max(self.proj_lat)

        x_min, y_min = projection.gps2proj(lon_min, lat_min)
        x_max, y_max = projection.gps2proj(lon_max, lat_max)

        diff_x = x_max - x_min
        diff_y = y_max - y_min

        if diff_x > diff_y:
            lon_index = list(self.proj_lon).index(min(self.proj_lon)) # Take the eastern most part
            proj_lon_anchor = min(self.proj_lon)
            proj_lat_anchor = self.proj_lat[lon_index]
        elif diff_x < diff_y:                     # Take the northern most part
            lat_index = list(self.proj_lat).index(min(self.proj_lat))
            proj_lat_anchor = min(self.proj_lat)
            proj_lon_anchor = self.proj_lon[lat_index]

        return proj_lon_anchor, proj_lat_anchor


    def projected_distance(self, proj_lon_anchor, proj_lat_anchor):

        self.proj_lon_anchor = proj_lon_anchor
        self.proj_lat_anchor = proj_lat_anchor

        utm_x, utm_y = projection.gps2proj(self.proj_lon, self.proj_lat)
        utm_x_anchor, utm_y_anchor = projection.gps2proj(proj_lon_anchor, proj_lat_anchor)

        x = (utm_x - utm_x_anchor)
        y = (utm_y - utm_y_anchor)

        distance = np.sqrt(x**2 + y**2)

        return distance


def perpendicular_distance(lon, lat, source_lon, source_lat):

    proc_gps = ProcessCoordinates(lon, lat)
    proj_lon, proj_lat = proc_gps.regression_vector()
    source_x, source_y = projection.gps2proj(source_lon, source_lat)
    utm_x, utm_y = projection.gps2proj(proj_lon, proj_lat)

    proc_utm = ProcessCoordinates(utm_x, utm_y)
    m, b, _ = proc_utm.regression_coeffs()

    perp_distance = np.abs((-m * source_x) + source_y - b) / np.sqrt(m**2)

    return perp_distance
