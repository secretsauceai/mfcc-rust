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
use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut2};

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
pub fn mel_to_frequency(mel: Array1<f64>) -> Array1<f64> {
    mel.map(|v| 700 * ((v / 1127.0).exp() - 1))
}

pub fn triangle(arr: ArrayViewMut2<f32>, x: Array1<f32>, left: usize, middle: usize, right: usize) {
    //original function: https://github.com/astorfi/speechpy/blob/master/speechpy/functions.py#L44
    //-> ArrayView1<&mut f32> {
    //let mut arr = Array1::<f64>::zeros(x.shape());
    //arr[x <= left] = 0;
    //arr[x >= right] = 0;
    arr.iter().enumerate().for_each(|(i, v)| {
        if let left..=right = v {
            //TODO: fix range bounds to be exclusive
            if v <= middle {
                *v = (x[i] - left) / (middle - left);
            } //NOTE: depending on whether the double <= >= is intended or not, may be simplified to just else
            if middle <= x {
                *v = (right - x[i]) / (right - middle)
            }
        } else {
            *v = 0.0
        }
    });
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
