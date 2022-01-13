/**function module.
This module contains necessary functions for calculating the features
in the `features` module.
Attributes:
    frequency_to_mel: Converting the frequency to Mel scale.
        This is necessary for filterbank energy calculation.
    mel_to_frequency: Converting the Mel to frequency scale.
        This is necessary for filterbank energy calculation.
    triangle: Creating a triangle for filterbanks.
        This is necessary for filterbank energy calculation.
    zero_handling: Handling zero values due to the possible
        issues regarding the log functions.
*/
/*from __future__ import division
import numpy as np
from . import processing
from scipy.fftpack import dct
import math*/
use ndarray::{Array1, Array2};

/**
 * converting from frequency to Mel scale.
    :param f: The frequency values(or a single frequency) in Hz.
    :returns: The mel scale values(or a single mel).
*/
pub fn frequency_to_mel(f: f64) -> f64 {
    1127 * (1 + f / 700.).log()
}
//Note: may want to try this crate https://github.com/SuperFluffy/rust-expm
/**
 * converting from Mel scale to frequency.
    :param mel: The mel scale values(or a single mel).
    :returns: The frequency values(or a single frequency) in Hz.
*/
pub fn mel_to_frequency(mel: Array2<f64>) -> Array2<f64> {
    mel.map(|v| 700 * ((v / 1127.0).exp() - 1))
}

pub fn triangle(x: Array1<f64>, left: f32, middle: f32, right: f32) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(x.shape());
    out[x <= left] = 0;
    out[x >= right] = 0;

    let first_half = np.logical_and(left < x, x <= middle);
    out[first_half] = (x[first_half] - left) / (middle - left);
    let second_half = np.logical_and(middle <= x, x < right);
    out[second_half] = (right - x[second_half]) / (right - middle);
    out
}
/**
 *
    This function handle the issue with zero values if the are exposed
    to become an argument for any log function.
    :param x: The vector.
    :return: The vector with zeros substituted with epsilon values.
*/
pub fn zero_handling(x: Array1<f64>) -> Array1<f64> {
    x.mapv(|x| if x == 0.0 { std::f64::EPSILON } else { x })
}
