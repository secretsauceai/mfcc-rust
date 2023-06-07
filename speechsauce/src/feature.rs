use crate::config::SpeechConfig;
/// This module provides functions for calculating the main speech
/// features that the package is aimed to extract as well as the required elements.
use crate::functions::{frequency_to_mel, mel_arr_to_frequency, triangle, zero_handling};
use crate::processing::stack_frames;
use crate::util::ArrayLog;

use ndarray::{
    concatenate, s, Array, Array1, Array2, Array3, ArrayBase, ArrayView1, ArrayViewMut1, Axis, Dim,
    Dimension, Ix2, NewAxis, Slice,
};
use ndrustfft::{nddct2, DctHandler};

/// Compute the Mel-filterbanks. Each filter will be stored in one rows.
///The columns correspond to fft bins.
/// Args:
///     num_filter : the number of filters in the filterbank, default 20.
///     coefficients : (fftpoints//2 + 1). Default is 257.
///     sampling_freq : the sample rate of the signal we are working with. It affects mel spacing.
///     low_freq : lowest band edge of mel filters, default 0 Hz
///     high_freq : highest band edge of mel filters,
///         default samplerate/2
/// Returns:
///         array: A ndarray of size num_filter x (fftpoints//2 + 1)
///             which are filterbank
pub(crate) fn filterbanks(
    num_filter: usize,
    coefficients: usize,
    sampling_freq: f32,
    low_freq: Option<f32>,
    high_freq: Option<f32>,
) -> Array2<f32> {
    //TODO: compare to https://pytorch.org/audio/main/_modules/torchaudio/functional/functional.html#melscale_fbanks
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
    let mels = Array1::<f32>::linspace(
        frequency_to_mel(low_freq),
        frequency_to_mel(high_freq),
        num_filter + 2,
    );

    // we should convert Mels back to Hertz because the start and end-points
    // should be at the desired frequencies.

    // The frequency resolution required to put filters at the
    // exact points calculated above should be extracted.
    //  So we should round those frequencies to the closest FFT bin.
    let freq_index = mel_arr_to_frequency(mels)
        .map(|x| ((coefficients + 1) as f32 * x / sampling_freq) as usize);

    // Initial definition
    let mut filterbank = Array2::zeros([num_filter, coefficients]);

    // The triangular function for each filter
    for i in 0..num_filter {
        let left = freq_index[i];
        let middle = freq_index[i + 1];
        let right = freq_index[i + 2];

        let z = Array1::<f32>::linspace(left as f32, right as f32, right - left + 1);

        {
            let mut s: ArrayViewMut1<f32> = filterbank.slice_mut(s![i, left..right + 1]);

            s.assign(&triangle(z, left as f32, middle as f32, right as f32));
        }
    }
    filterbank
}

/// Compute MFCC features from an audio signal.
///     Args:
///          signal : the audio signal from which to compute features.
///              Should be an N x 1 array
pub fn mfcc(signal: ArrayView1<f32>, speech_config: &SpeechConfig) -> Array2<f32> {
    let (mut feature, energy) = mfe(signal, &speech_config);

    if feature.is_empty() {
        return Array::<f32, _>::zeros((0_usize, speech_config.num_cepstral));
    }
    feature = feature.log();
    //feature second axis equal to num_filters
    let feature_axis_len = feature.shape()[1];
    let mut transformed_feature = Array2::<f32>::zeros(feature.raw_dim());
    //link to og code:
    //https://github.com/astorfi/speechpy/blob/2.4/speechpy/feature.py#L147
    //function dct docs:
    //https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dct.html
    //param explanations:
    //1. feature: actual matrix being transformed
    //2. type=2: the dct type, in this case 2.
    //3. axis=-1: apply along last axis of feature
    //4. norm="ortho": when used with dct type 2, applies a scaling factor

    //need to switch to rustfft, provide len of last axis specifically
    let mut dct_handler: DctHandler<f32> = DctHandler::new(feature_axis_len);

    //need to check how to specify axis of transformation
    nddct2(&feature, &mut transformed_feature, &mut dct_handler, 1);
    //NOTE: may be able to remove the if/else by processing first element separately
    //https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dct.html#:~:text=2N)-,If%20norm%3D%27ortho%27%2C%20y%5Bk%5D%20is%20multiplied%20by%20a%20scaling%20factor%20f,-f%3D%7B
    let n = transformed_feature.len() as f32;
    //orthonormalized the transformed axis
    transformed_feature[[0, 0]] *= 1. / (4. * n).sqrt();
    transformed_feature
        .slice_axis_mut(Axis(1), Slice::new(1, None, 1))
        .mapv_inplace(|x| x * (1. / (2. * n).sqrt()));

    transformed_feature = transformed_feature.slice_move(s![.., ..speech_config.num_cepstral]);

    // replace first cepstral coefficient with log of frame energy for DC
    // elimination.
    if speech_config.dc_elimination {
        //>>>x = np.array([[1,2,3,4],[5,6,7,8]])
        //>>>x[:,0]
        //array([1, 5])
        //TODO: need to verify the shapes for these two match
        let mut replace_me = transformed_feature.slice_mut(s![.., 0]);
        {
            replace_me.assign(&energy.log())
        }
    }
    transformed_feature
}
//TODO: https://pytorch.org/audio/main/_modules/torchaudio/transforms/_transforms.html#MelSpectrogram
//https://github.com/librosa/librosa/blob/c800e74f6a6ec5c27e0fa978d7355943cce04359/librosa/feature/spectral.py#LL2021C5-L2021C5
fn mel_spectrogram(signal: ArrayView1<f32>, speech_config: &SpeechConfig) {
    //ndarray::einsum
    todo!("implement mel_spectrogram")
}
///a helper function that is passed to stack_frames from mfe
fn _f_it(x: usize) -> Array2<f32> {
    Array2::<f32>::ones((x, 1))
}

/// Compute Mel-filterbank energy features from an audio signal.
///    Args:
///         signal: the audio signal from which to compute features.
///             Should be an N x 1 array
///         sampling_frequency : the sampling frequency of the signal
///             we are working with.
///         frame_length : the length of each frame in seconds.
///             Default is 0.020s
///         frame_stride : the step between successive frames in seconds.
///             Default is 0.02s (means no overlap)
///         num_filters : the number of filters in the filterbank,
///             default 40.
///         fft_length : number of FFT points. Default is 512.
///         low_frequency : lowest band edge of mel filters.
///             In Hz, default is 0.
///         high_frequency : highest band edge of mel filters.
///             In Hz, default is samplerate/2
///    Returns:
///         array: features - the energy of fiterbank of size num_frames x num_filters.
///         The energy of each frame: num_frames x 1
pub fn mfe(signal: ArrayView1<f32>, speech_config: &SpeechConfig) -> (Array2<f32>, Array1<f32>) {
    //
    // Stack frames
    let frames = stack_frames(
        signal,
        speech_config.sample_rate,
        speech_config.frame_length,
        speech_config.frame_stride,
        None,
        false,
    );

    // calculation of the power spectrum
    let power_spectrum = crate::processing::power_spectrum(frames, speech_config.window_size);

    // this stores the total energy in each frame
    let frame_energies = power_spectrum.sum_axis(Axis(1));

    // Handling zero energies.
    let frame_energies = zero_handling(frame_energies);

    // Filterbank energies
    let features = power_spectrum.dot(&speech_config.filter_banks.view().reversed_axes());
    let features = crate::functions::zero_handling(features);

    (features, frame_energies)
}

/// Compute log Mel-filterbank energy features from an audio signal.
///    Args:
///         signal : the audio signal from which to compute features.
///             Should be an N x 1 array
///         speech_config: the configuration for the speech processing functions
///    Returns:
///         array: Features - The log energy of fiterbank of size num_frames x num_filters frame_log_energies. The log energy of each frame num_frames x 1
fn lmfe(signal: ArrayView1<f32>, speech_config: &SpeechConfig) -> Array2<f32> {
    let (feature, _) = mfe(signal, speech_config);
    feature.log()
}

/// extracts temporal derivative features which are first and second derivatives.
/// uses the derivative extraction function from the processing module
/// Args:
///     feature : The feature vector which its size is: N x M
/// Return:
///     array: The feature cube vector which contains the static, first and second derivative features of size: N x M x 3
fn extract_derivative_feature(feature: Array2<f32>) -> Array3<f32> {
    let first_derivative_feature = crate::processing::derivative_extraction(&feature, 2);
    let second_derivative_feature =
        crate::processing::derivative_extraction(&first_derivative_feature, 2);

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
