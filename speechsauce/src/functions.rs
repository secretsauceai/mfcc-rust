/// contains necessary functions for calculating the features in the `features` module.
use ndarray::{s, Array, Array1, ArrayD, Dimension, Zip};
use stft::realfft::RealFftPlanner;

use crate::util::PadType;

/// converts a single value representing a frequency in Hz to Mel scale.
pub fn frequency_to_mel(f: f64) -> f64 {
    1127. * (1. + f / 700.).ln()
}
///converts all values in the input array from frequency to mel scale
pub fn frequency_arr_to_mel<D>(freq: Array<f64, D>) -> Array<f64, D>
where
    D: Dimension,
{
    freq.mapv(|v| 1127. * (1. + v / 700.).ln())
}

/// converts a single value from the mel scale to a frequency scale.
pub fn mel_to_frequency<D>(mel: f64) -> f64 {
    700. * ((mel / 1127.0).exp() - 1.)
}

///converts all values in the input array from mel scale to frequency scale
pub fn mel_arr_to_frequency<D>(mel: Array<f64, D>) -> Array<f64, D>
where
    D: Dimension,
{
    mel.mapv(|v| 700. * ((v / 1127.0).exp() - 1.))
}

pub fn triangle(arr: Array1<f64>, left: f64, middle: f64, right: f64) -> Array1<f64> {
    //original function: https://github.com/astorfi/speechpy/blob/master/speechpy/functions.py#L44
    let mut out = ndarray::Array1::<f64>::zeros(arr.len());
    Zip::from(&mut out).and(&arr).for_each(|v, x| {
        if (left..right).contains(x) {
            //TODO: fix range bounds to be exclusive, see https://github.com/rust-lang/rust/issues/37854
            if x <= &middle {
                *v = (x - left) / (middle - left);
            } //NOTE: depending on whether the double <= >= is intended or not, may be simplified to just else
            if &middle <= x {
                *v = (right - x) / (right - middle);
            }
        } else {
            *v = 0.0
        }
    });
    out
}

///    This function handle the issue with zero values if the are exposed
///    to become an argument for any log function.
///    :param x: The vector.
///    :return: The vector with zeros substituted with epsilon values.
pub fn zero_handling<D>(x: Array<f64, D>) -> Array<f64, D>
where
    D: Dimension,
{
    x.mapv(|x| if x == 0.0 { std::f64::EPSILON } else { x })
}

// https://github.com/librosa/librosa/blob/c800e74f6a6ec5c27e0fa978d7355943cce04359/librosa/core/spectrum.py#L46
///Short time Fourier transform (STFT)
/// TODO:
fn stft(
    y: &ArrayD<f64>,
    n_fft: usize,
    hop_length: Option<usize>,
    win_length: Option<usize>,
    window: &ArrayD<f64>,
    center: bool,
    pad_mode: PadType,
) -> ArrayD<f64> {
    // By default, use the entire frame
    let win_length = win_length.unwrap_or(n_fft);

    // Set the default hop, if it's not already specified
    let hop_length = match hop_length {
        Some(hop) if hop > 0 => hop,
        Some(hop) => panic!("hop_length={} must be a positive integer", hop),
        None => win_length / 4,
    };

    // Check audio is valid
    assert!(
        y.ndim() >= 1 && y.ndim() <= 2,
        "y must have 1 or 2 dimensions"
    );

    // Compute the window
    let fft_window = compute_fft_window(window, win_length, n_fft);

    // Compute the padding
    let (padded_y, start, extra) = compute_padding(y, center, pad_mode, n_fft, hop_length);

    // Allocate the STFT matrix
    let n_frames = (y.len() - start) / hop_length + 1 + extra;
    let stft_matrix = if let Some(mut out_array) = out {
        assert!(
            out_array.shape()[..out_array.ndim() - 1]
                == &[1 + n_fft / 2, n_frames.try_into().unwrap()],
            "Shape mismatch for provided output array"
        );
        out_array.view_mut()
    } else {
        Array::zeros((1 + n_fft / 2, n_frames.try_into().unwrap()))
    };

    // Compute the STFT
    let mut planner = RealFftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut input_frame = Array::zeros((n_fft,));
    for t in 0..n_frames {
        let t_offset = start + t * hop_length;
        input_frame.assign(&padded_y.slice(s![t_offset..t_offset + n_fft]));
        input_frame *= &fft_window;
        let fft_output = ndfft_r compute_fft(&mut fft, &input_frame);
        let stft_frame = &mut stft_matrix.slice_mut(s![.., t]);
        copy_stft_frame(fft_output, stft_frame);
    }

    stft_matrix
}

fn compute_fft_window(window: &NdArrayD, win_length: usize, n_fft: usize) -> ArrayD<f32> {
    let fft_window = if window.shape() == &[win_length] {
        window.view()
    } else {
        let fft_window = window
            .into_shape((win_length,))
            .unwrap_or_else(|_| {
                panic!("window must have length equal to or less than win_length={}", win_length)
            })
            .to_owned();
        pad_fft_window(&fft_window, n_fft)
    };
    fft_window.to_owned()
}