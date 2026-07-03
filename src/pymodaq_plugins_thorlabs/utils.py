# -*- coding: utf-8 -*-
"""
Created the 31/08/2023

@author: Sebastien Weber
"""
from pathlib import Path

from pymodaq.utils.config import BaseConfig, USER, GlobalConfig


class Config(BaseConfig):
    """Main class to deal with configuration values for this plugin"""
    config_template_path = Path(__file__).parent.joinpath('resources/config_template.toml')
    config_name = f"config_{__package__.split('pymodaq_plugins_')[1]}"

# function that are useful but may be should not be in this precise file?

import numpy as np


class BeamAnalysisCache:
    def __init__(self):
        self.shape = None
        self.x = None
        self.y = None
        self.x2 = None
        self.y2 = None

    def update(self, shape):
        if self.shape != shape:
            H, W = shape
            self.shape = shape
            self.x = np.arange(W, dtype=np.float64)
            self.y = np.arange(H, dtype=np.float64)
            self.x2 = self.x ** 2
            self.y2 = self.y ** 2


def analyze_beam_moments(
    img,
    cache: BeamAnalysisCache,
    threshold_mode='Auto (Percentage)',
    threshold_percent=10.0,
    threshold_value=10.0,
    min_signal=1e-9,
):
    """
    Analyse d'un faisceau par moments d'ordre 1 et 2.

    Retourne un dictionnaire :
    {
        'detected': bool,
        'centroid_x': float,
        'centroid_y': float,
        'waist_u': float,
        'waist_v': float,
        'theta_deg': float,
        'power': float,
    }
    """

    img = np.asarray(img)

    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}")

    H, W = img.shape
    cache.update((H, W))

    # Conversion en float32 pour limiter le coût mémoire.
    # Les sommes critiques sont ensuite faites en float64.
    img_f = img.astype(np.float32, copy=False)

    img_max = float(np.max(img_f))

    if threshold_mode == 'Auto (Percentage)':
        th = img_max * threshold_percent / 100.0
    else:
        th = threshold_value

    # Une seule allocation principale
    I = np.where(img_f >= th, img_f, 0.0).astype(np.float32, copy=False)

    P = float(I.sum(dtype=np.float64))

    if P <= min_signal:
        return {
            'detected': False,
            'centroid_x': 0.0,
            'centroid_y': 0.0,
            'waist_u': 0.0,
            'waist_v': 0.0,
            'theta_deg': 0.0,
            'power': 0.0,
        }

    x = cache.x
    y = cache.y
    x2 = cache.x2
    y2 = cache.y2

    x_profile = I.sum(axis=0, dtype=np.float64)
    y_profile = I.sum(axis=1, dtype=np.float64)

    x_bar = float(np.dot(x_profile, x) / P)
    y_bar = float(np.dot(y_profile, y) / P)

    mu_xx = float(np.dot(x_profile, x2) / P - x_bar ** 2)
    mu_yy = float(np.dot(y_profile, y2) / P - y_bar ** 2)

    mu_xx = max(mu_xx, 0.0)
    mu_yy = max(mu_yy, 0.0)

    # Terme croisé sans meshgrid :
    # sum_y sum_x I[y,x] * y * x
    xy = float(np.dot(y, I @ x))
    mu_xy = xy / P - x_bar * y_bar

    diff = mu_xx - mu_yy
    term = np.sqrt(diff ** 2 + 4.0 * mu_xy ** 2)

    w_u = 2.0 * np.sqrt(max((mu_xx + mu_yy + term) / 2.0, 0.0))
    w_v = 2.0 * np.sqrt(max((mu_xx + mu_yy - term) / 2.0, 0.0))

    theta = 0.5 * np.arctan2(2.0 * mu_xy, diff)

    # Attention : ceci est une puissance relative caméra, pas une puissance physique calibrée.
    power = float(P)

    return {
        'detected': True,
        'centroid_x': x_bar,
        'centroid_y': y_bar,
        'waist_u': float(w_u),
        'waist_v': float(w_v),
        'theta_deg': float(np.degrees(theta)),
        'power': power,
    }
