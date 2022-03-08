#!/usr/bin/env python
# coding=utf-8

import numpy as np
import pandas as pd


def read_usa(usa_file, start, end):

    usa_df = pd.read_csv(usa_file, sep=",", index_col=0, parse_dates=True)
    usa_df = usa_df[["uu", "vv", "ww", "wd", "ws", "pp", "u.star", "L "]]
    # usa_df = usa_df[["uu", "vv", "ww", "wd", "ws", "u.star"]]

    usa_df = usa_df[(usa_df.index >= start) & (usa_df.index <= end)]

    u_star = usa_df["u.star"]
    # u_wind = usa_df["uu"]
    # v_wind = usa_df["vv"]
    # w_wind = usa_df["ww"]
    ws = usa_df["ws"]
    wd = usa_df["wd"]

    ws = np.mean(ws)
    u_star = np.mean(u_star)
    wd = np.mean(wd)

    return ws, u_star, wd


def read_heidelberg(usa_file, start, end):

    usa_df = pd.read_csv(usa_file, encoding="ISO-8859-1", index_col="UTC")
    usa_df.index = pd.DatetimeIndex(usa_df.index)

    usa_df = usa_df[(usa_df.index >= start) & (usa_df.index <= end)]
    u_star = usa_df["ustar"]
    ws = usa_df["vel"].astype(float)
    wd = usa_df["dir"]

    ws = np.mean(ws)
    u_star = np.mean(u_star)
    wd = np.mean(wd)

    return ws, u_star, wd


def read_dronemeteo(usa_file, start, end):


    usa_df = pd.read_csv(usa_file, index_col=0, parse_dates=True)
    usa_df = usa_df[["ws", "wd", "u.star"]]

    usa_df = usa_df[(usa_df.index >= start) & (usa_df.index <= end)]

    u_star = usa_df["u.star"]
    ws = usa_df["ws"]
    wd = usa_df["wd"]

    ws = np.mean(ws)
    u_star = np.mean(u_star)
    wd = np.mean(wd)

    return ws, u_star, wd 


def meteorology_data(meteo_file, start, end):

    usa_df = pd.read_csv(meteo_file, index_col=0, parse_dates=True)
    usa_df = usa_df[["uu", "vv", "ws", "wd", "u.star"]]

    usa_df = usa_df[(usa_df.index >= start) & (usa_df.index <= end)]

    return usa_df


def meteorology_heidelberg(usa_file, start, end):

    usa_df = pd.read_csv(usa_file, encoding="ISO-8859-1", index_col="UTC")
    usa_df.index = pd.DatetimeIndex(usa_df.index)

    usa_df = usa_df[["u", "v", "vel", "dir", "ustar"]]
    usa_df = usa_df[(usa_df.index >= start) & (usa_df.index <= end)]

    rename_cols = {"u":"uu", "v":"vv", "vel":"ws", "dir":"wd", "ustar":"u.star"}
    usa_df = usa_df.rename(columns=rename_cols)
    usa_df["ws"] = usa_df["ws"].astype(float)

    return usa_df


def minute_wind(usa_file, start, end):

    usa_df = pd.read_csv(usa_file, sep=",", index_col=0, parse_dates=True)
    usa_df = usa_df[["uu", "vv", "ww", "wd", "ws", "pp", "u.star", "L "]]
    # usa_df = usa_df[["uu", "vv", "ww", "wd", "ws", "u.star"]]

    usa_df = usa_df[(usa_df.index >= start) & (usa_df.index <= end)]

    ws = usa_df["ws"]
    wd = usa_df["wd"]

    return ws, wd

