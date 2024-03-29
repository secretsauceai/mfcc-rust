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

use std::ops::Mul;

use crate::util::{pad, repeat_axis, PadType};
use ndarray::azip;
use ndarray::s;
use ndarray::{stack, Array1, Array2, ArrayView1, ArrayView2, Axis, Ix1, Ix2};
use ndrustfft::{ndfft_r2c, Complex, R2cFftHandler};

/// preemphasising on the signal.
/// Args:
///     signal (array): The input signal.
///     shift (int): The shift step.
///     cof (float): The preemphasising coefficient. 0 equals to no filtering.
/// Returns:
///     The pre-emphasized signal.
pub fn preemphasis(
    signal: Array1<f32>,
    shift: isize, /*1*/
    cof: f32,     /*=0.98*/
) -> Array1<f32> {
    //Note: https://github.com/rust-ndarray/ndarray/issues/281
    //lets use an example of a signal with 5 elements and a shift of 2
    let mut rolled_signal = Array1::<f32>::zeros(signal.shape()[0]);
    //the original array will be divided into two chunks
    {
        //first assign the right chunk of the result from the left chunk of the input
        // in the example 0,1,2,3
        let mut rolled_slice = rolled_signal.slice_mut(s![..shift]);
        rolled_slice.assign(&signal.slice(s![-shift..]));
    }
    {
        //second assign the left chunk of the result for the right chunk of the input
        //in the example 4,5
        let mut rolled_slice = rolled_signal.slice_mut(s![shift..]);
        rolled_slice.assign(&signal.slice(s![..-shift]));
    }
    signal - (cof * rolled_signal)
}

/// Frame a signal into overlapping frames.
/// Args:
///     signal : The audio signal to frame of size (N,).
///     sample rate (int): The sampling frequency of the signal.
///     frame_length (float): The length of the frame in second.
///     frame_stride (float): The stride between frames.
///     filter (array): The time-domain filter for applying to each frame. By default it is one so nothing will be changed.
///     zero_padding (bool): If the samples is not a multiple of frame_length(number of frames sample), zero padding will be done for generating last frame.
/// Returns:
///     Stacked_frames-Array of frames of size (number_of_frames x frame_len).
pub fn stack_frames(
    signal: ArrayView1<f32>,
    sample_rate: usize,
    frame_length: f32, /*=0.020*/
    frame_stride: f32, /*=0.020*/
    filter: Option<fn(usize) -> Array2<f32>>, /*=lambda x: np.ones(
                       (x,
                        ))*/
    zero_padding: bool, /*=True*/
) -> Array2<f32> {
    // Initial necessary values
    let length_signal = signal.len();
    let frame_sample_length = (sample_rate as f32 * frame_length).round() as usize; // Defined by the number of samples
    let frame_step_size = ((sample_rate as f32 * frame_stride).round()) as usize;
    let len_sig: usize;
    let numframes: usize;

    //TODO: once the code is working simplify this section, handle sig directly and
    //let the below if else declare the last index
    // Zero padding is done for allocating space for the last frame.
    let signal = if zero_padding {
        // TODO: simplify once this makes its way to stable:
        // https://github.com/rust-lang/rust/issues/88581
        // TODO: check why the suggested edit (for including the last frame length) in issue below causes error
        // https://github.com/astorfi/speechpy/issues/34
        // see if implementation needs to change to address it
        numframes = ((length_signal - (frame_sample_length)) as f32 / frame_step_size as f32).ceil()
            as usize;

        // Zero padding
        len_sig = (numframes * frame_step_size) + frame_sample_length;
        let additive_zeros = ndarray::Array::<f32, Ix1>::zeros(((len_sig - length_signal),));
        ndarray::concatenate![Axis(0), signal, additive_zeros]
    } else {
        // No zero padding! The last frame which does not have enough
        // samples(remaining samples <= frame_sample_length), will be dropped!
        numframes = ((length_signal - frame_sample_length) as f32 / frame_step_size as f32).floor()
            as usize;

        // new length
        let len_sig = ((numframes - 1) * frame_step_size) + frame_sample_length;
        signal.slice(s![0..len_sig]).to_owned()
    };
    let _slice_len = numframes * frame_sample_length;

    let mut frames = Array2::zeros((numframes, frame_sample_length));

    let doubled_sig = stack![Axis(0), signal, signal];
    let sig_chunks = doubled_sig.exact_chunks((numframes, 2));
    frames
        .exact_chunks_mut((numframes, 2))
        .into_iter()
        .zip(sig_chunks)
        .for_each(|(mut frame_chunk, sig_chunk)| {
            frame_chunk.assign(&sig_chunk);
        });

    if let Some(f) = filter {
        let filt = f(frame_sample_length);
        let window = repeat_axis(filt.view(), Axis(0), numframes);
        return frames * window;
    }

    frames
}

/// This function computes the one-dimensional n-point discrete Fourier
/// Transform (DFT) of a real-valued array by means of an efficient algorithm
/// called the Fast Fourier Transform (FFT). Please refer to
/// https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.rfft.html
/// for further details.
///Args:
///    frames : The frame array in which each row is a frame.
///    fft_points : The length of FFT. If fft_length is greater than frame_len, the frames will be zero-padded.
/// Returns:
///     ndarray representing the The fft spectrum.
/// If frames is an num_frames x sample_per_frame matrix, output
/// will be num_frames x FFT_LENGTH.
fn fft_spectrum(frames: Array2<f32>, fft_points: usize /*=512*/) -> Array2<f32> {
    let row_size = frames.shape()[0];
    let col_size = frames.shape()[1];
    let mut handler = R2cFftHandler::<f32>::new(fft_points);
    let frames = if col_size < fft_points {
        pad(
            &frames,
            vec![[0, 0], [0, fft_points - col_size]],
            0.,
            PadType::Constant,
        )
    } else {
        frames
    };
    let mut spectrum_vector = Array2::<Complex<f32>>::zeros((row_size, fft_points / 2 + 1));

    ndfft_r2c(
        &frames.view(),
        &mut spectrum_vector.view_mut(),
        &mut handler,
        1,
    );

    //would this work?
    //spectrum_vector.abs()
    spectrum_vector.map(|v: &Complex<f32>| -> f32 { (v.re.powf(2.) + v.im.powf(2.)).sqrt() })
}

/// Power spectrum of each frame.
/// Args:
///     frames : The frame array in which each row is a frame.
///     fft_points : The length of FFT. If fft_length is greater than frame_len, the frames will be zero-padded.
/// Returns:
///         array representing The power spectrum.
/// If frames is an num_frames x sample_per_frame matrix, output
/// will be num_frames x fft_length.
pub fn power_spectrum(frames: Array2<f32>, fft_points: usize /*=512*/) -> Array2<f32> {
    fft_spectrum(frames, fft_points).map(|x| (1. / fft_points as f32) * *x)
}

/// Log power spectrum of each frame in frames.
/// Args:
///     frames : The frame array in which each row is a frame.
///     fft_points : The length of FFT. If fft_length is greater than
///         frame_len, the frames will be zero-padded.
///     normalize : If true, the log power spectrum
///         will be normalized.
/// Returns:
///         array representing The power spectrum  
/// If frames is an (num_frames x sample_per_frame) matrix, output will be
/// num_frames x fft_length.
fn log_power_spectrum(
    frames: Array2<f32>,
    fft_points: usize, /*=512*/
    normalize: bool,   /*=True*/
) -> Array2<f32> {
    let mut mx = 1e-20_f32; //had to do this because of trait constraints on max
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

/// This function extracts the derivative features.
/// Args:
///     feat : The main feature vector(For returning the second order derivative it can be first-order derivative).
///     DeltaWindows : The value of  DeltaWindows is set using the configuration parameter DELTAWINDOW.
/// Returns:
///     A NUMFRAMESxNUMFEATURES array which is the derivative features along the features.
pub fn derivative_extraction(feat: &Array2<f32>, delta_windows: usize) -> Array2<f32> {
    // Getting the shape of the vector.
    let cols = feat.shape()[1];

    // Difining the vector of differences.
    let mut difference_vector = Array2::<f32>::zeros(feat.raw_dim());
    let mut scale = 0.;

    // Pad only along features in the vector.
    let features = pad(
        feat,
        vec![[0, 0], [delta_windows, delta_windows]],
        0.,
        PadType::Edge,
    );
    for i in 0..delta_windows {
        // Start index
        let offset = delta_windows;

        // The dynamic range
        let Range = i + 1;

        let dif = features
            .slice(s![.., offset + Range..offset + Range + cols])
            .mul(Range as f32)
            - features.slice(s![.., offset - Range..offset - Range + cols]);

        scale += 2. * (Range as f32).powf(2.);
        difference_vector = difference_vector + dif;
    }

    difference_vector / scale
}

/// This function is aimed to perform global cepstral mean and
/// variance normalization (CMVN) on input feature vector "vec".
/// The code assumes that there is one observation per row.
/// Args:
///     vec (array): input feature matrix
///         (size:(num_observation,num_features))
/// variance_normalization (bool): If the variance normalization should be performed or not.
/// Return:
///     array: The mean(or mean+variance) normalized feature vector.
pub fn cmvn(vec: ArrayView2<f32>, variance_normalization: bool /*=False*/) -> Array2<f32> {
    let eps = 2.0f32.powf(-30.);
    let rows = vec.shape()[0];

    // Mean calculation
    let norm = vec.mean_axis(Axis(0)).unwrap();
    let norm_len = norm.len();
    let norm_vec = repeat_axis(
        norm.into_shape((1, norm_len)).unwrap().view(),
        Axis(0),
        rows,
    );

    // Mean subtraction
    let mean_subtracted = &vec - norm_vec;

    // Variance normalization
    if variance_normalization {
        let stdev = mean_subtracted.std_axis(Axis(0), 0.);
        let stdev_len = stdev.len();

        let stdev_vec = repeat_axis(
            stdev.into_shape((1, stdev_len)).unwrap().view(),
            Axis(0),
            rows,
        );

        (mean_subtracted / (stdev_vec + eps))
            .into_dimensionality::<Ix2>()
            .expect("error shaping output of cmvn with variance normalization")
    } else {
        mean_subtracted
            .into_dimensionality::<Ix2>()
            .expect("error shaping output of cmvn")
    }
}

/// This function is aimed to perform local cepstral mean and
/// variance normalization on a sliding window.
/// The code assumes that there is one observation per row.
/// Args:
///     vec : input feature matrix
///         (size:(num_observation,num_features))
///     win_size : The size of sliding window for local normalization.
///         Default=301 which is around 3s if 100 Hz rate is
///         considered(== 10ms frame stide)
///     variance_normalization : If the variance normilization should
///         be performed or not.
/// Return:
///         array: The mean(or mean+variance) normalized feature vector.
pub fn cmvnw(
    vec: Array2<f32>,
    win_size: usize,              /*=301*/
    variance_normalization: bool, /*=False*/
) -> Array2<f32> {
    //TODO: verify shape of output
    // Get the shapes
    let eps = 2f32.powf(-30.);
    let rows = vec.shape()[0];

    // Windows size must be odd.
    //assert isinstance(win_size, int), "Size must be of type 'int'!"
    assert!(win_size % 2 == 1, "Windows size must be odd!");

    // Padding and initial definitions
    let pad_size = (win_size - 1) / 2;
    //NOTE: see https://github.com/rust-ndarray/ndarray/issues/823#issuecomment-942392888
    let vec_pad = pad(
        &vec,
        vec![[pad_size, pad_size], [0, 0]],
        0.,
        PadType::Symmetric,
    );
    let mut mean_subtracted = ndarray::Array2::<f32>::zeros(vec.raw_dim());

    (0..rows).for_each(|i| {
        let window = vec_pad.slice(s![i..i + win_size, ..]); //NOTE: we have to fix pad before fixing this error
                                                             //TODO: preallocate window mean
        let window_mean = window.mean_axis(Axis(0)).unwrap();
        azip!((a in &mut mean_subtracted.slice_mut(s![i, ..]),&b in &vec.slice(s![i, ..]),c in &window_mean)*a=b-c); //this took way too long to figure out lol
    });

    // Variance normalization
    if variance_normalization {
        // Initial definitions.
        let mut variance_normalized = Array2::<f32>::zeros(vec.raw_dim());
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
