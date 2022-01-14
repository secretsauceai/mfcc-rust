// -*- coding: utf-8 -*-
/*Processing module for signal processing operations.
This module demonstrates documentation for the signal processing
function which are required as internal computations in the package.
Attributes:
    preemphasis: Preemphasising on the signal. This is a preprocessing step.
    stack_frames: Create stacking frames from the raw signal.
    fft_spectrum: Calculation of the Fast Fourier Transform.
    power_spectrum: Power Spectrum calculation.
    log_power_spectrum: Log Power Spectrum calculation.
    derivative_extraction: Calculation of the derivative of the extracted featurs.
    cmvn: Cepstral mean variance normalization. This is a post processing operation.
    cmvnw: Cepstral mean variance normalization over the sliding window. This is a post processing operation.
*/

const __license__: &str = "MIT";
const __author__: &str = " Amirsina Torfi";
const __docformat__: &str = "reStructuredText";

/*import decimal
import numpy as np
import math*/

use crate::util::{pad, tile};
use ndarray::{s, Array1, Array2, Axis};
use ndrustfft::{ndfft_r2c, Complex, R2cFftHandler};
//use realfft::RealFftPlanner;
//use rustfft::FftPlanner;

// 1.4 becomes 1 and 1.6 becomes 2. special case: 1.5 becomes 2.
// fn round_half_up(number: i32) -> i32 {
//     return decimal
//         .Decimal(number)
//         .quantize(decimal.Decimal('1'), rounding = decimal.ROUND_HALF_UP) as i32;
// }

/**
 *  preemphasising on the signal.
    Args:
        signal (array): The input signal.
        shift (int): The shift step.
        cof (float): The preemphasising coefficient. 0 equals to no filtering.
    Returns:
           array: The pre-emphasized signal.
*/
fn preemphasis(signal: Array1<f32>, shift: i32 /*1*/, cof: f32 /*=0.98*/) -> Array1<f32> {
    //Note: https://github.com/rust-ndarray/ndarray/issues/281

    //let rolled_signal = np.roll(signal, shift);
    let mut rolled_signal = Array1::<f32>::zeros(signal.shape());
    {
        let mut s = rolled_signal.slice_mut(s![1..-1, ..]);
        s += &signal.slice(s![shift.., ..]);
        s -= &signal.slice(s![..-shift, ..]);
    }
    signal - (cof * rolled_signal)
}

/**
 * Frame a signal into overlapping frames.
    Args:
        sig (array): The audio signal to frame of size (N,).
        sampling_frequency (int): The sampling frequency of the signal.
        frame_length (float): The length of the frame in second.
        frame_stride (float): The stride between frames.
        filter (array): The time-domain filter for applying to each frame.
            By default it is one so nothing will be changed.
        zero_padding (bool): If the samples is not a multiple of
            frame_length(number of frames sample), zero padding will
            be done for generating last frame.
    Returns:
            array: Stacked_frames-Array of frames of size (number_of_frames x frame_len).
*/
pub fn stack_frames(
    sig: Array1<f32>,
    sampling_frequency: i32,
    frame_length: f32, /*=0.020*/
    frame_stride: f32, /*=0.020*/
    filter: fn(i32) -> Array1<f32>, /*=lambda x: np.ones(
                       (x,
                        ))*/
    zero_padding: bool, /*=True*/
) -> Array2<f32> {
    // Check dimension
    assert!(
        sig.ndim == 1,
        format!(
            "Signal dimention should be of the format of (N,) but it is {:?} instead",
            sig.shape
        )
    );

    // Initial necessary values
    let length_signal = sig.len();
    let frame_sample_length = (sampling_frequency as f32 * frame_length).round(); // Defined by the number of samples
    let frame_stride = (sampling_frequency as f32 * frame_stride).round();
    let mut len_sig = 0;
    let mut numframes = 0;

    // Zero padding is done for allocating space for the last frame.
    let signal = if zero_padding {
        // Calculation of number of frames
        numframes = ((length_signal - frame_sample_length) / frame_stride).ceil() as i32;
        println!(
            "{} {} {} {}",
            numframes, length_signal, frame_sample_length, frame_stride
        );

        // Zero padding
        len_sig = (numframes * frame_stride + frame_sample_length) as i32;
        let additive_zeros = ndarray::ArrayBase::zeros((len_sig - length_signal,));
        ndarray::concatenate![Axis(0), sig, additive_zeros];
    } else {
        // No zero padding! The last frame which does not have enough
        // samples(remaining samples <= frame_sample_length), will be dropped!
        numframes = (length_signal - frame_sample_length) / frame_stride;

        // new length
        let len_sig = ((numframes - 1) * frame_stride + frame_sample_length) as i32;
        sig[0..len_sig]
    };

    // Getting the indices of all frames.
    let indices = tile(
        ndarray::Array::range(0, frame_sample_length),
        (numframes, 1),
    ) + tile(
        ndarray::Array::range(0, numframes * frame_stride, frame_stride),
        (frame_sample_length, 1),
    )
    .transpose();
    indices = Array1::from::<i32>(indices);

    // Extracting the frames based on the allocated indices.
    let frames = signal[indices];

    // Apply the windows function
    let window = tile(filter(frame_sample_length), (numframes, 1));
    frames * window // Extracted frames
}

/*    This function computes the one-dimensional n-point discrete Fourier
    Transform (DFT) of a real-valued array by means of an efficient algorithm
    called the Fast Fourier Transform (FFT). Please refer to
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.rfft.html
    for further details.
    Args:
        frames (array): The frame array in which each row is a frame.
        fft_points (int): The length of FFT. If fft_length is greater than frame_len, the frames will be zero-padded.
    Returns:
            array: The fft spectrum.
            If frames is an num_frames x sample_per_frame matrix, output
            will be num_frames x FFT_LENGTH.
*/
fn fft_spectrum(frames: Array2<f32>, fft_points: i32 /*=512*/) {
    //SPECTRUM_VECTOR = np.fft.rfft(frames, n = fft_points, axis = -1, norm = None)
    //in case of fire see https://github.com/secretsauceai/mfcc-rust/issues/2
    // let real2comp = RealFftPlanner::new()
    //     .real_planner()
    //     .plan_fft_forward(fft_points);
    // let mut input_vec = real2comp.make_input_vec();
    // let mut spectrum_vector = real2comp.make_output_vec();
    // real2comp.process(&input_vec, &output_vec);
    let col_size = frames.shape()[1];
    let mut handler = R2cFftHandler::<f32>::new(fft_points);
    let mut spectrum_vector = Array2::<Complex<f32>>::zeros((fft_points / 2 + 1, col_size));
    ndfft_r2c(&frames, &mut spectrum_vector, &mut handler, -1);
    //would this work?
    //spectrum_vector.abs()
    spectrum_vector.map(|v| v.abs())
}

/**
 * Power spectrum of each frame.
    Args:
        frames (array): The frame array in which each row is a frame.
        fft_points (int): The length of FFT. If fft_length is greater than frame_len, the frames will be zero-padded.
    Returns:
            array: The power spectrum.
            If frames is an num_frames x sample_per_frame matrix, output
            will be num_frames x fft_length.
*/
pub fn power_spectrum(frames: Array1<f32>, fft_points: i32 /*=512*/) -> Array2<f32> {
    (1.0 / fft_points as f32) * ndarray::ArrayBase::square(fft_spectrum(frames, fft_points))
}

/**
 * Log power spectrum of each frame in frames.
    Args:
        frames (array): The frame array in which each row is a frame.
        fft_points (int): The length of FFT. If fft_length is greater than
            frame_len, the frames will be zero-padded.
        normalize (bool): If normalize=True, the log power spectrum
            will be normalized.
    Returns:
           array: The power spectrum - If frames is an
           num_frames x sample_per_frame matrix, output will be
           num_frames x fft_length.
*/
fn log_power_spectrum(
    frames: Array2<f32>,
    fft_points: i32, /*=512*/
    normalize: bool, /*=True*/
) -> Array2<f32> {
    let log_power_spec = power_spectrum(frames, fft_points).map(|x| {
        if *x > 1e-20 {
            10 * x.log10()
        } else {
            -200.0 //10*log10(1e-20)
        }
    });
    if normalize {
        log_power_spec - log_power_spec.max()
    } else {
        log_power_spec
    }
}

/*
This function the derivative features.
    Args:
        feat (array): The main feature vector(For returning the second
             order derivative it can be first-order derivative).
        DeltaWindows (int): The value of  DeltaWindows is set using
            the configuration parameter DELTAWINDOW.
    Returns:
           array: Derivative feature vector - A NUMFRAMESxNUMFEATURES numpy
           array which is the derivative features along the features.
*/
pub fn derivative_extraction(feat: Array2<f32>, DeltaWindows: i32) -> Array2<f32> {
    // Getting the shape of the vector.
    let (rows, cols) = feat.shape();

    // Difining the vector of differences.
    let mut DIF = Array2::<f32>::zeros(feat.shape());
    let Scale = 0;

    // Pad only along features in the vector.
    let FEAT = pad(feat, ((0, 0), (DeltaWindows, DeltaWindows)), "edge");
    for i in 0..DeltaWindows {
        // Start index
        let offset = DeltaWindows;

        // The dynamic range
        let Range = i + 1;

        let dif = Range * FEAT.slice(s![.., offset + Range..offset + Range + cols])
            - FEAT.slice(s![.., offset - Range..offset - Range + cols]);

        Scale += 2 * Range.pow(2);
        DIF += dif;
    }

    DIF / Scale
}
//NOTE: determine if they are using tiling for broadcasting or
//if so this code may be simplified
//see note attached to definition of numpy tile
//see https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#broadcasting
/**
 * This function is aimed to perform global cepstral mean and
        variance normalization (CMVN) on input feature vector "vec".
        The code assumes that there is one observation per row.
    Args:
        vec (array): input feature matrix
            (size:(num_observation,num_features))
        variance_normalization (bool): If the variance
            normilization should be performed or not.
    Return:
          array: The mean(or mean+variance) normalized feature vector.
*/
fn cmvn(vec: Array2<f32>, variance_normalization: bool /*=False*/) -> Array2<f32> {
    let eps = 2.0f32.powf(-30);
    let (rows, cols) = vec.shape();

    // Mean calculation
    let norm = ndarray::Array1::mean_axis(vec, 0).unwrap();
    let norm_vec = tile(norm, (rows, 1));

    // Mean subtraction
    let mean_subtracted = vec - norm_vec;

    // Variance normalization
    if variance_normalization {
        let stdev = ndarray::ArrayBase::std_axis(mean_subtracted, 0);
        let stdev_vec = tile(stdev, (rows, 1));
        mean_subtracted / (stdev_vec + eps)
    } else {
        mean_subtracted
    }
}

/**
    This function is aimed to perform local cepstral mean and
    variance normalization on a sliding window. The code assumes that
    there is one observation per row.
    Args:
        vec (array): input feature matrix
            (size:(num_observation,num_features))
        win_size (int): The size of sliding window for local normalization.
            Default=301 which is around 3s if 100 Hz rate is
            considered(== 10ms frame stide)
        variance_normalization (bool): If the variance normilization should
            be performed or not.
    Return:
            array: The mean(or mean+variance) normalized feature vector.
*/
fn cmvnw(
    vec: Array1<f32>,
    win_size: i32,                /*=301*/
    variance_normalization: bool, /*=False*/
) {
    // Get the shapes
    let eps = 2 * *-30;
    let (rows, cols) = vec.shape;

    // Windows size must be odd.
    //assert isinstance(win_size, int), "Size must be of type 'int'!"
    assert!(win_size % 2 == 1, "Windows size must be odd!");

    // Padding and initial definitions
    let pad_size = (win_size - 1) / 2;
    //NOTE: see https://github.com/rust-ndarray/ndarray/issues/823#issuecomment-942392888
    let vec_pad = pad(vec, ((pad_size, pad_size), (0, 0)), "symmetric");
    let mut mean_subtracted = ndarray::ArrayBase::<f32>::zeros(vec);

    for i in 0..rows {
        let window = vec_pad.slice(s![i..i + win_size, ..]);
        let window_mean = ndarray::ArrayBase::mean_axis(window, 0);
        mean_subtracted.slice(s![i, ..]) = vec.slice(s![i, ..]) - window_mean;
    }

    // Variance normalization
    if variance_normalization {
        // Initial definitions.
        let variance_normalized = Array2::<f32>::zeros(vec.shape());
        let vec_pad_variance = pad(mean_subtracted, ((pad_size, pad_size), (0, 0)), "symmetric");

        // Looping over all observations.
        for i in 0..rows {
            let window = vec_pad_variance.slice(s![i..i + win_size, ..]);
            let window_variance = ndarray::ArrayBase::std_axis(window, 0);
            variance_normalized.slice_mut(s![i, ..]) =
                mean_subtracted.slice(s![i, ..]) / (window_variance + eps)
        }

        variance_normalized
    } else {
        mean_subtracted
    }
}
