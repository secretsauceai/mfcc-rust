use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use speechsauce::{config::SpeechConfig, feature, processing};
#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PySpeechConfig {
    pub speech_config: SpeechConfig,
}

#[pymodule]
fn speechsauce(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    /// Compute MFCC features from an audio signal.
    ///     Args:
    ///          signal : the audio signal from which to compute features.
    ///              Should be an N x 1 array
    ///          sampling_frequency : the sampling frequency of the signal
    ///              we are working with.
    ///          frame_length : the length of each frame in seconds.
    ///              Default is 0.020s
    ///          frame_stride : the step between successive frames in seconds.
    ///              Default is 0.02s (means no overlap)
    ///          num_filters : the number of filters in the filterbank,
    ///              default 40.
    ///          fft_length : number of FFT points. Default is 512.
    ///          low_frequency : lowest band edge of mel filters.
    ///              In Hz, default is 0.
    ///          high_frequency (float): highest band edge of mel filters.
    ///              In Hz, default is samplerate/2
    ///          num_cepstral (int): Number of cepstral coefficients.
    ///          dc_elimination (bool): hIf the first dc component should
    ///              be eliminated or not.
    ///     Returns:
    ///         array: A numpy array of size (num_frames x num_cepstral) containing mfcc features.
    #[pyfn(m)]
    fn mfcc<'py>(
        py: Python<'py>,
        signal: PyReadonlyArray1<f64>,
        config: PySpeechConfig,
    ) -> &'py PyArray2<f64> {
        let PySpeechConfig { speech_config } = config;
        feature::mfcc(signal.as_array(), &speech_config).to_pyarray(py)
    }

    //TODO: #14 make signal a mutable borrow (PyReadWriteArray) once the next version of numpy-rust is released
    #[pyfn(m)]
    fn preemphasis<'py>(
        py: Python<'py>,
        signal: PyReadonlyArray1<f64>,
        shift: isize,
        cof: f64,
    ) -> &'py PyArray1<f64> {
        processing::preemphasis(signal.as_array().to_owned(), shift, cof).into_pyarray(py)
    }

    #[pyfn(m)]
    fn cmvn<'py>(
        py: Python<'py>,
        vec: PyReadonlyArray2<f64>,
        variance_normalization: bool,
    ) -> &'py PyArray2<f64> {
        processing::cmvn(vec.as_array(), variance_normalization).into_pyarray(py)
    }

    fn _speech_config<'py>(
        py: Python<'py>,
        sampling_frequency: usize,
        frame_length: f64,           // =0.020,
        frame_stride: f64,           // =0.01,
        num_cepstral: usize,         // =13,
        num_filters: usize,          // =40,
        fft_length: usize,           // =512,
        low_frequency: f64,          // =0,
        high_frequency: Option<f64>, // =None,
        dc_elimination: bool,        //True
    ) -> &'py PySpeechConfig {
        &'py PySpeechConfig {
            speech_config: SpeechConfig::new(
                sampling_frequency,
                fft_length,
                frame_length,
                frame_stride,
                num_cepstral,
                num_filters,
                low_frequency,
                high_frequency.unwrap_or(sampling_frequency as f64 / 2.0),
                dc_elimination,
            ),
        }
    }
    Ok(())
}
