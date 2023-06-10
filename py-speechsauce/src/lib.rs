use std::sync::Arc;

use ::speechsauce::{config::SpeechConfig, feature, processing};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArrayDyn, ToPyArray,
};
use pyo3::{callback::IntoPyCallbackOutput, prelude::*};
#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PySpeechConfig(SpeechConfig);

impl IntoPyCallbackOutput<Self> for PySpeechConfig {
    fn convert(self, py: Python<'_>) -> PyResult<Self> {
        Ok(self)
    }
}

impl IntoPy<SpeechConfig> for PySpeechConfig {
    fn into_py(self, py: Python<'_>) -> SpeechConfig {
        self.0
    }
}

#[pymodule]
fn speechsauce(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySpeechConfig>()?;
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
        signal: PyReadonlyArray1<f32>,
        config: Py<PySpeechConfig>,
    ) -> &'py PyArray2<f32> {
        let cell = config.as_ref(py);
        let obj_ref = cell.borrow();
        let speech_config = &obj_ref.0;
        feature::mfcc(signal.as_array(), &speech_config).to_pyarray(py)
    }

    #[pyfn(m)]
    fn mel_spectrogram<'py>(
        py: Python<'py>,
        signal: PyReadonlyArrayDyn<f32>,
        config: Py<PySpeechConfig>,
    ) -> &'py PyArrayDyn<f32> {
        let cell = config.as_ref(py);
        let obj_ref = cell.borrow();
        let speech_config = obj_ref.0;
        let result = match signal.ndim() {
            1 => feature::mel_spectrogram1(
                signal.as_array().into_dimensionality().unwrap(),
                speech_config,
            )
            .into_dyn(),
            2 => feature::mel_spectrogram2(
                signal.as_array().into_dimensionality().unwrap(),
                speech_config,
            )
            .into_dyn(),
            _ => {
                panic!("Input signal must be 1d or 2d")
            }
        };
        result.to_pyarray(py)
    }

    //TODO: #14 make signal a mutable borrow (PyReadWriteArray) once the next version of numpy-rust is released
    #[pyfn(m)]
    fn preemphasis<'py>(
        py: Python<'py>,
        signal: PyReadonlyArray1<f32>,
        shift: isize,
        cof: f32,
    ) -> &'py PyArray1<f32> {
        processing::preemphasis(signal.as_array().to_owned(), shift, cof).into_pyarray(py)
    }

    #[pyfn(m)]
    fn cmvn<'py>(
        py: Python<'py>,
        vec: PyReadonlyArray2<f32>,
        variance_normalization: bool,
    ) -> &'py PyArray2<f32> {
        processing::cmvn(vec.as_array(), variance_normalization).into_pyarray(py)
    }

    #[pyfn(m)]
    fn _speech_config<'py>(
        py: Python<'py>,
        sampling_frequency: usize,
        frame_length: f32,           // =0.020,
        frame_stride: f32,           // =0.01,
        num_cepstral: usize,         // =13,
        num_filters: usize,          // =40,
        fft_length: usize,           // =512,
        low_frequency: f32,          // =0,
        high_frequency: Option<f32>, // =None,
        dc_elimination: bool,        //True
    ) -> Py<PySpeechConfig> {
        Py::new(
            py,
            PySpeechConfig(SpeechConfig::new(
                sampling_frequency,
                fft_length,
                frame_length,
                frame_stride,
                num_cepstral,
                num_filters,
                low_frequency,
                high_frequency.unwrap_or(sampling_frequency as f32 / 2.0),
                dc_elimination,
            )),
        )
        .unwrap()
    }
    Ok(())
}
