#!/usr/bin/env python
# coding=utf-8

import pyproj

####################################
# Module to convert map            #
# coordinates to different         #
# projections and vice-versa       #
####################################

# R. Morales - Aug. 2018

proj = "+proj=utm +zone=35 +ellps=WGS84 +datum=WGS84 +units=m +nodefs"

def proj2gps(x, y, proj=proj, inverse=False):
    '''
    x: band 0 from APEX geo location
    y: band 1 from APEX geo location

    proj: proj4 description (e.g. from: http://spatialreference.org)

    Examples for proj:
    Swiss Coordinate System = "+proj=somerc +lat_0=46.95240555555556
    +lon_0=7.439583333333333 +k_0=1 +x_0=600000 +y_0=200000 +ellps=bessel
    +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs"
    Switzerland: "SWISS LV03"
    Rotterdam:  "+proj=utm +zone=31 +ellps=WGS84 +datum=WGS84 +units=m +nodefs"
    '''
    p = pyproj.Proj(proj)
    lon, lat = p(x, y, inverse=(not inverse))

    return lon, lat


def gps2proj(lon, lat, proj=proj, inverse=False):
    '''
    x: band 0 from APEX geo location
    y: band 1 from APEX geo location

    proj: proj4 description (e.g. from: http://spatialreference.org)

    Examples for proj:
    Swiss Coordinate System = "+proj=somerc +lat_0=46.95240555555556
    +lon_0=7.439583333333333 +k_0=1 +x_0=600000 +y_0=200000 +ellps=bessel
    +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs"
    Switzerland: "SWISS LV03"
    Rotterdam:   "+proj=utm +zone=31 +ellps=WGS84 +datum=WGS84 +units=m +nodefs"
    '''
    p = pyproj.Proj(proj)
    x,y = p(lon, lat, inverse=inverse)

    return x, y
