/**feature module.
This module provides functions for calculating the main speech
features that the package is aimed to extract as well as the required
elements.
Functions:
    filterbanks: Compute the Mel-filterbanks
                 The filterbanks must be created for extracting
                 speech features such as MFCC.
    mfcc: Extracting Mel Frequency Cepstral Coefficient feature.
    mfe: Extracting Mel Energy feature.
    lmfe: Extracting Log Mel Energy feature.
    extract_derivative_feature: Extract the first and second derivative
        features. This finction, directly use the ``derivative_extraction``
        function in the ``processing`` module.
*/
use crate::functions::{frequency_to_mel, mel_to_frequency, triangle, zero_handling};
use crate::processing::{power_spectrum, stack_frames};
use crate::util::ArrayLog;

use ndarray::{
    concatenate, s, Array, Array1, Array2, Array3, ArrayBase, ArrayViewMut1, Axis, NewAxis,
    OwnedRepr,
};

/*from __future__ import division
import numpy as np
from . import processing
from scipy.fftpack import dct
from . import functions*/

/**Compute the Mel-filterbanks. Each filter will be stored in one rows.
The columns correspond to fft bins.
Args:
    num_filter (int): the number of filters in the filterbank, default 20.
    coefficients (int): (fftpoints//2 + 1). Default is 257.
    sampling_freq (float): the samplerate of the signal we are working
        with. It affects mel spacing.
    low_freq (float): lowest band edge of mel filters, default 0 Hz
    high_freq (float): highest band edge of mel filters,
        default samplerate/2
Returns:
        array: A numpy array of size num_filter x (fftpoints//2 + 1)
            which are filterbank
*/
pub fn filterbanks(
    num_filter: usize,
    coefficients: usize,
    sampling_freq: f64,
    low_freq: Option<f64>,
    high_freq: Option<f64>,
) -> ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> {
    let high_freq = high_freq.unwrap_or(sampling_freq / 2.0);
    let low_freq = low_freq.unwrap_or(300.0);
    assert!(
        high_freq <= sampling_freq / 2.0,
        "High frequency cannot be greater than half of the sampling frequency!"
    );
    assert!(low_freq >= 0.0, "low frequency cannot be less than zero!");

    // Computing the Mel filterbank
    // converting the upper and lower frequencies to Mels.
    // num_filter + 2 is because for num_filter filterbanks we need
    // num_filter+2 point.
    let mels = Array1::<f64>::linspace(
        frequency_to_mel(low_freq),
        frequency_to_mel(high_freq),
        num_filter + 2,
    );

    // we should convert Mels back to Hertz because the start and end-points
    // should be at the desired frequencies.

    // The frequency resolution required to put filters at the
    // exact points calculated above should be extracted.
    //  So we should round those frequencies to the closest FFT bin.
    let freq_index = mel_to_frequency(mels)
        .map(|x| ((coefficients as i32 + 1) as f64 * x / sampling_freq) as usize);

    // Initial definition
    let filterbank = Array2::zeros([num_filter, coefficients]);

    // The triangular function for each filter
    for i in 0..num_filter {
        let left = freq_index[i] as f64;
        let middle = freq_index[i + 1] as f64;
        let right = freq_index[i + 2] as f64;
        let z = Array1::<f64>::linspace(left, right, right as usize - left as usize + 1);

        {
            let mut s: ArrayViewMut1<f64> =
                filterbank.slice_mut(s![i, left as usize..right as usize + 1]);
            triangle(s, z, left, middle, right);
        }
    }

    filterbank
}

/**Compute MFCC features from an audio signal.
    Args:
         signal (array): the audio signal from which to compute features.
             Should be an N x 1 array
         sampling_frequency (int): the sampling frequency of the signal
             we are working with.
         frame_length (float): the length of each frame in seconds.
             Default is 0.020s
         frame_stride (float): the step between successive frames in seconds.
             Default is 0.02s (means no overlap)
         num_filters (int): the number of filters in the filterbank,
             default 40.
         fft_length (int): number of FFT points. Default is 512.
         low_frequency (float): lowest band edge of mel filters.
             In Hz, default is 0.
         high_frequency (float): highest band edge of mel filters.
             In Hz, default is samplerate/2
         num_cepstral (int): Number of cepstral coefficients.
         dc_elimination (bool): hIf the first dc component should
             be eliminated or not.
    Returns:
        array: A numpy array of size (num_frames x num_cepstral) containing mfcc features.
*/
fn mfcc(
    signal: Array1<f64>,
    sampling_frequency: usize,
    frame_length: f64,           // =0.020,
    frame_stride: f64,           // =0.01,
    num_cepstral: usize,         // =13,
    num_filters: usize,          // =40,
    fft_length: usize,           // =512,
    low_frequency: f64,          // =0,
    high_frequency: Option<f64>, // =None,
    dc_elimination: bool,        //True
) -> Array1<f64> {
    let (feature, energy) = mfe(
        signal,
        sampling_frequency,
        frame_length,
        frame_stride,
        num_filters,
        fft_length,
        low_frequency,
        high_frequency,
    );

    if feature.len() == 0 {
        return Array2::zeros((0_usize, num_cepstral));
    }
    feature = feature.log();

    feature = rustdct::DctPlanner::new()
        .plan_dct2(
            feature, /*type=*/ 2, /*axis=*/ -1, /*norm=*/ "ortho",
        ) //#TODO: fix this
        .slice(s![.., ..num_cepstral]);

    // replace first cepstral coefficient with log of frame energy for DC
    // elimination.
    if dc_elimination {
        feature.slice_mut(s![.., 0]) = energy.log();
    }
    return feature;
}
fn f_it(x: usize) -> ArrayBase<OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> {
    Array1::<f64>::ones(x as usize)
}
/**
* """Compute Mel-filterbank energy features from an audio signal.

   Args:
        signal (array): the audio signal from which to compute features.
            Should be an N x 1 array
        sampling_frequency (int): the sampling frequency of the signal
            we are working with.
        frame_length (float): the length of each frame in seconds.
            Default is 0.020s
        frame_stride (float): the step between successive frames in seconds.
            Default is 0.02s (means no overlap)
        num_filters (int): the number of filters in the filterbank,
            default 40.
        fft_length (int): number of FFT points. Default is 512.
        low_frequency (float): lowest band edge of mel filters.
            In Hz, default is 0.
        high_frequency (float): highest band edge of mel filters.
            In Hz, default is samplerate/2
   Returns:
             array: features - the energy of fiterbank of size num_frames x num_filters. The energy of each frame: num_frames x 1
   */
fn mfe(
    signal: Array1<f64>,
    sampling_frequency: usize,
    frame_length: f64,           /*=0.020*/
    frame_stride: f64,           /*=0.01*/
    num_filters: usize,          /*=40*/
    fft_length: usize,           /*=512*/
    low_frequency: f64,          /*=0*/
    high_frequency: Option<f64>, /*None*/
) -> (Array1<f64>, Array1<f64>) {
    // Convert to float
    //let signal = signal.type(float);
    let f = |x: i32| -> ArrayBase<OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> {
        Array1::<f64>::ones(x as usize)
    };
    // Stack frames
    let frames = stack_frames(
        signal,
        sampling_frequency,
        frame_length,
        frame_stride,
        f_it,
        false, /*=False*/
    );

    // getting the high frequency
    let high_frequency = high_frequency.unwrap_or(sampling_frequency as f64 / 2);

    // calculation of the power sprectum
    let power_spectrum = crate::processing::power_spectrum(frames, fft_length);
    let coefficients = power_spectrum.shape()[1];
    // this stores the total energy in each frame
    let frame_energies = power_spectrum.sum_axis(Axis(1));

    // Handling zero enegies.
    let frame_energies = zero_handling(frame_energies);

    // Extracting the filterbank
    let filter_banks = filterbanks(
        num_filters as usize,
        coefficients,
        sampling_frequency.into(),
        Some(low_frequency),
        Some(high_frequency),
    );

    // Filterbank energies
    let features = power_spectrum.dot(&filter_banks.reversed_axes());
    let features = crate::functions::zero_handling(features);

    (features, frame_energies)
}

/**
*     """Compute log Mel-filterbank energy features from an audio signal.
   Args:
        signal (array): the audio signal from which to compute features.
            Should be an N x 1 array
        sampling_frequency (int): the sampling frequency of the signal
            we are working with.
        frame_length (float): the length of each frame in seconds.
            Default is 0.020s
        frame_stride (float): the step between successive frames in seconds.
            Default is 0.02s (means no overlap)
        num_filters (int): the number of filters in the filterbank,
            default 40.
        fft_length (int): number of FFT points. Default is 512.
        low_frequency (float): lowest band edge of mel filters.
            In Hz, default is 0.
        high_frequency (float): highest band edge of mel filters.
            In Hz, default is samplerate/2
   Returns:
             array: Features - The log energy of fiterbank of size num_frames x num_filters frame_log_energies. The log energy of each frame num_frames x 1
*/
fn lmfe(
    signal: Array1<f64>,
    sampling_frequency: usize,
    frame_length: f64,           /*=0.020*/
    frame_stride: f64,           /*=0.01*/
    num_filters: usize,          /*=40*/
    fft_length: usize,           /*=512*/
    low_frequency: f64,          /*=0*/
    high_frequency: Option<f64>, /*None*/
) -> Array1<f64> {
    let (feature, _frame_energies) = mfe(
        signal,
        sampling_frequency,
        frame_length,
        frame_stride,
        num_filters,
        fft_length,
        low_frequency,
        high_frequency,
    );
    feature.log()
}

/**
    This function extracts temporal derivative features which are
        first and second derivatives.
    Args:
        feature (array): The feature vector which its size is: N x M
    Return:
          array: The feature cube vector which contains the static, first and second derivative features of size: N x M x 3
*/
fn extract_derivative_feature(feature: Array2<f64>) -> Array3<f64> {
    let first_derivative_feature = crate::processing::derivative_extraction(feature, 2);
    let second_derivative_feature =
        crate::processing::derivative_extraction(first_derivative_feature, 2);

    // Creating the future cube for each file
    //Note about numpy syntax in equivalent function
    //https://stackoverflow.com/a/1408435/11019565
    let feature_cube = concatenate![
        Axis(2),
        feature.slice(s![.., .., NewAxis]),
        first_derivative_feature.slice(s![.., .., NewAxis]),
        second_derivative_feature.slice(s![.., .., NewAxis])
    ];

    feature_cube
}
