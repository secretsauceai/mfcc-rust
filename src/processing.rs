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

const __LICENSE__: &str = "MIT";
//const __docformat__: &str = "reStructuredText";

use std::ops::Mul;

use crate::util::{pad, tile, PadType};
use ndarray::{azip, s, Array1, Array2, Axis, Dimension, Ix1, Ix2};
use ndrustfft::{ndfft_r2c, Complex, R2cFftHandler};

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
fn preemphasis(signal: Array1<f64>, shift: i32 /*1*/, cof: f64 /*=0.98*/) -> Array1<f64> {
    //Note: https://github.com/rust-ndarray/ndarray/issues/281

    //let rolled_signal = np.roll(signal, shift);
    let mut rolled_signal = Array1::<f64>::zeros(signal.shape()[0]);
    {
        rolled_signal += &signal.slice(s![shift..]);
        rolled_signal -= &signal.slice(s![..-shift]);
    }
    signal - (cof * rolled_signal)
}
//TODO: change the return type, the returned matrix will have varying dimensions
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
    sig: Array1<f64>,
    sampling_frequency: usize,
    frame_length: f64, /*=0.020*/
    frame_stride: f64, /*=0.020*/
    filter: fn(usize) -> Array1<f64>, /*=lambda x: np.ones(
                       (x,
                        ))*/
    zero_padding: bool, /*=True*/
) -> Array2<f64> {
    // Check dimension
    //not necessary due to function type signature
    // assert!(
    //     sig.ndim() == 1,
    //     &format!(
    //         "Signal dimention should be of the format of (N,) but it is {:?} instead",
    //         sig.shape()
    //     )
    // );

    // Initial necessary values
    let length_signal = sig.len();
    let frame_sample_length = (sampling_frequency as f64 * frame_length).round() as usize; // Defined by the number of samples
    let frame_stride = (sampling_frequency as f64 * frame_stride).round();
    let mut len_sig: usize;
    let mut numframes: usize;

    //TODO: once the code is working simplify this section, handle sig directly and
    //let the below if else declare the last index
    // Zero padding is done for allocating space for the last frame.
    let signal = if zero_padding {
        // Calculation of number of frames
        numframes = ((length_signal - frame_sample_length) as f64 / frame_stride).ceil() as usize;
        println!(
            "{} {} {} {}",
            numframes, length_signal, frame_sample_length, frame_stride
        );

        // Zero padding
        len_sig = (numframes as f64 * frame_stride) as usize + frame_sample_length;
        let additive_zeros =
            ndarray::Array::<f64, Ix1>::zeros(((len_sig - length_signal) as usize,));
        ndarray::concatenate![Axis(0), sig, additive_zeros]
    } else {
        // No zero padding! The last frame which does not have enough
        // samples(remaining samples <= frame_sample_length), will be dropped!
        numframes = ((length_signal - frame_sample_length) as f64 / frame_stride) as usize;

        // new length
        let len_sig =
            ((numframes - 1) as f64 * frame_stride) as usize + frame_sample_length as usize;
        sig.slice_move(s![0..len_sig])
    };

    // Getting the indices of all frames.
    let indices = tile::<f64, Ix1>(
        &ndarray::Array::range(0., frame_sample_length as f64, 1.),
        vec![numframes, 1],
    ) + tile::<f64, Ix1>(
        &ndarray::Array::range(0., numframes as f64 * frame_stride, frame_stride),
        vec![frame_sample_length, 1],
    )
    .t();
    let indices = indices.mapv(|v| v as usize);

    // Extracting the frames based on the allocated indices.
    let frames = indices
        .map(|i| {
            *signal.get(*i).expect(&format!(
                "code panicked when trying to access element {} of ndarray signal.\n\n signal:\n{}",
                i, signal
            ))
        })
        .into_dimensionality::<Ix2>()
        .expect("failed to convert frames into 2D array");

    let window = tile::<f64, Ix1>(&filter(frame_sample_length), vec![numframes, 1])
        .into_dimensionality::<Ix2>()
        .expect("failed to convert window into 2d array");

    //NOTE: frames is Nx1, window is Mx1, so result is MxN
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
fn fft_spectrum(frames: Array2<f64>, fft_points: usize /*=512*/) -> Array2<f64> {
    //SPECTRUM_VECTOR = np.fft.rfft(frames, n = fft_points, axis = -1, norm = None)
    //in case of fire see https://github.com/secretsauceai/mfcc-rust/issues/2
    // let real2comp = RealFftPlanner::new()
    //     .real_planner()
    //     .plan_fft_forward(fft_points);
    // let mut input_vec = real2comp.make_input_vec();
    // let mut spectrum_vector = real2comp.make_output_vec();
    // real2comp.process(&input_vec, &output_vec);
    let col_size = frames.shape()[1];
    let mut handler = R2cFftHandler::<f64>::new(fft_points);
    let mut spectrum_vector = Array2::<Complex<f64>>::zeros((fft_points / 2 + 1, col_size));
    ndfft_r2c(
        &frames,
        &mut spectrum_vector,
        &mut handler,
        frames.shape()[1], //this is the last axis right
    );
    //would this work?
    //spectrum_vector.abs()
    spectrum_vector.map(|v: &Complex<f64>| -> f64 { (v.re.powf(2.) + v.im.powf(2.)).sqrt() as f64 })
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
pub fn power_spectrum(frames: Array2<f64>, fft_points: usize /*=512*/) -> Array2<f64> {
    fft_spectrum(frames, fft_points).map(|x| (1. / fft_points as f64) * *x)
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
    frames: Array2<f64>,
    fft_points: usize, /*=512*/
    normalize: bool,   /*=True*/
) -> Array2<f64> {
    let mut mx = 1e-20 as f64; //had to do this because of trait constraints on max
    let log_power_spec = power_spectrum(frames, fft_points).map_mut(|x| {
        if *x > 1e-20 {
            *x = 10. * x.log10();
            mx = *x;
            *x
        } else {
            -200.0 //10*log10(1e-20)
        }
    });
    if normalize {
        log_power_spec - mx
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
pub fn derivative_extraction(feat: &Array2<f64>, DeltaWindows: usize) -> Array2<f64> {
    // Getting the shape of the vector.
    let cols = feat.shape()[1];

    // Difining the vector of differences.
    let mut DIF = Array2::<f64>::zeros(feat.raw_dim());
    let mut Scale = 0.;

    // Pad only along features in the vector.
    let FEAT = pad(
        &feat,
        vec![[0, 0], [DeltaWindows, DeltaWindows]],
        0.,
        PadType::Edge,
    );
    for i in 0..DeltaWindows {
        // Start index
        let offset = DeltaWindows;

        // The dynamic range
        let Range = i + 1;

        let dif = FEAT
            .slice(s![.., offset + Range..offset + Range + cols])
            .mul(Range as f64)
            - FEAT.slice(s![.., offset - Range..offset - Range + cols]);

        Scale += 2. * (Range as f64).powf(2.);
        DIF = DIF + dif;
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
fn cmvn(vec: Array2<f64>, variance_normalization: bool /*=False*/) -> Array2<f64> {
    let eps = 2.0f64.powf(-30.);
    let rows = vec.shape()[0];

    // Mean calculation
    let norm = &vec.mean_axis(Axis(0)).unwrap();
    let norm_vec = tile::<f64, Ix1>(norm, vec![rows, 1]);

    // Mean subtraction
    let mean_subtracted = vec - norm_vec;

    // Variance normalization
    if variance_normalization {
        let stdev = mean_subtracted.std_axis(Axis(0), 0.);

        let stdev_vec = tile::<f64, ndarray::IxDyn>(&stdev, vec![rows, 1]);

        (mean_subtracted / (stdev_vec + eps))
            .into_dimensionality::<Ix2>()
            .expect("error shaping output of cmvn with variance normalization")
    } else {
        mean_subtracted
            .into_dimensionality::<Ix2>()
            .expect("error shaping output of cmvn with variance normalization")
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
    vec: Array2<f64>,
    win_size: usize,              /*=301*/
    variance_normalization: bool, /*=False*/
) -> Array2<f64> {
    //TODO: verify shape of output
    // Get the shapes
    let eps = 2f64.powf(-30.);
    let rows = vec.shape()[0];

    // Windows size must be odd.
    //assert isinstance(win_size, int), "Size must be of type 'int'!"
    assert!(win_size % 2 == 1, "Windows size must be odd!");

    // Padding and initial definitions
    let pad_size = ((win_size - 1) / 2) as usize;
    //NOTE: see https://github.com/rust-ndarray/ndarray/issues/823#issuecomment-942392888
    let vec_pad = pad(
        &vec,
        vec![[pad_size, pad_size], [0, 0]],
        0.,
        PadType::Symmetric,
    );
    let mut mean_subtracted = ndarray::Array2::<f64>::zeros(vec.raw_dim());

    (0..rows).for_each(|i| {
        let window = vec_pad.slice(s![i..i + win_size, ..]); //NOTE: we have to fix pad before fixing this error
                                                             //TODO: preallocate window mean
        let window_mean = window.mean_axis(Axis(0)).unwrap();
        azip!((a in &mut mean_subtracted.slice_mut(s![i, ..]),&b in &vec.slice(s![i, ..]),c in &window_mean)*a=b-c); //this took way too long to figure out lol
    });

    // Variance normalization
    if variance_normalization {
        // Initial definitions.
        let mut variance_normalized = Array2::<f64>::zeros(vec.raw_dim());
        let vec_pad_variance = pad(
            &mean_subtracted,
            vec![[pad_size, pad_size], [0, 0]],
            0.,
            PadType::Symmetric,
        );

        // Looping over all observations.
        (0..rows).for_each(|i| {
            let window = vec_pad_variance.slice(s![i..i + win_size, ..]); //currently the return type is wrapped around &&str?
            let window_variance = window.std_axis(Axis(0), 0.);
            azip!((a in &mut variance_normalized.slice_mut(s![i, ..]),
                &b in &mean_subtracted.slice(s![i, ..]) ,c in &window_variance) *a=b/(c+eps))
            //error related to return type of pad
        });

        variance_normalized
    } else {
        mean_subtracted
    }
}
