#!/usr/bin/env python
# coding=utf-8

## Measurement data
memo_file = "sample_data/EMP_QCLAS_312_03.csv"
rug_file = "sample_data/RUG_ACORE_312_03.csv"
meteo_file = "sample_data/agg_20200312.csv"
# fit_file = "/project/mrp/DUREX/312/CURTAIN_3/FIT/FIT_312_03.nc"

## Target path for model results
model_path = "sample_data/results/"

source_lon = 8.603447
source_lat = 47.390561

## Transect time
start = "2020-03-12 14:58:47"
end = "2020-03-12 15:09:50"

## Grid specifications
dx = 0.5
dz = 0.5

## Parameters for REBS background correction
## No need to change
rebs_NoXP = 150.0
rebs_b =  3.0

flight_code = "312_03"

## Micrometeorology parameters
wind_z = 2.0
wind_OL = 13.05969
wind_u_star = 0.2488
