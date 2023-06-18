use ::speechsauce::{config::SpeechConfig, feature, processing};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArrayDyn, ToPyArray,
};
use pyo3::{callback::IntoPyCallbackOutput, prelude::*};
#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PySpeechSauce(SpeechConfig);

impl IntoPyCallbackOutput<Self> for PySpeechSauce {
    fn convert(self, py: Python<'_>) -> PyResult<Self> {
        Ok(self)
    }
}

impl IntoPy<SpeechConfig> for PySpeechSauce {
    fn into_py(self, py: Python<'_>) -> SpeechConfig {
        self.0
    }
}

// #[pymethods]
// impl PySpeechSauce {
//     #[new]
//     fn new(
//         sample_rate: usize,
//         fft_points: usize,
//         frame_length: f32,
//         frame_stride: f32,
//         num_cepstral: usize,
//         num_filters: usize,
//         low_frequency: f32,
//         high_frequency: f32,
//         dc_elimination: bool,
//     ) -> Self {
//         Self(SpeechConfig::new(
//             sample_rate,
//             fft_points,
//             frame_length,
//             frame_stride,
//             num_cepstral,
//             num_filters,
//             low_frequency,
//             high_frequency,
//             dc_elimination,
//         ))
//     }

//     #[getter]
//     fn sample_rate(&self) -> usize {
//         self.0.sample_rate
//     }

//     #[getter]
//     fn fft_points(&self) -> usize {
//         self.0.window_size
//     }
//     #[getter]
//     fn window_size(&self) -> usize {
//         self.0.window_size
//     }

//     #[getter]
//     fn frame_length(&self) -> f32 {
//         self.0.frame_length
//     }

//     #[getter]
//     fn frame_stride(&self) -> f32 {
//         self.0.frame_stride
//     }

//     #[getter]
//     fn num_cepstral(&self) -> usize {
//         self.0.num_cepstral
//     }

//     #[getter]
//     fn num_filters(&self) -> usize {
//         self.0.num_filters
//     }

//     #[getter]
//     fn low_frequency(&self) -> f32 {
//         self.0.low_frequency
//     }

//     #[getter]
//     fn high_frequency(&self) -> f32 {
//         self.0.high_frequency
//     }

//     #[getter]
//     fn dc_elimination(&self) -> bool {
//         self.0.dc_elimination
//     }

//     #[setter]
//     fn set_sample_rate(&mut self, sample_rate: usize) {
//         self.0.sample_rate = sample_rate;
//     }

//     #[setter]
//     fn set_fft_points(&mut self, fft_points: usize) {
//         self.set_window_size(fft_points);
//     }
//     #[setter]
//     fn set_window_size(&mut self, fft_points: usize) {
//         self.0.window_size = fft_points;
//         self.0.window_size_half = fft_points / 2;
//     }

//     #[setter]
//     fn set_frame_length(&mut self, frame_length: f32) {
//         self.0.frame_length = frame_length;
//     }

//     #[setter]
//     fn set_frame_stride(&mut self, frame_stride: f32) {
//         self.0.frame_stride = frame_stride;
//     }

//     #[setter]
//     fn set_num_cepstral(&mut self, num_cepstral: usize) {
//         self.0.num_cepstral = num_cepstral;
//     }

//     fn mel_spectrogram(&mut self, signal: &PyArrayDyn<f32>) -> Py<PyArrayDyn<f32>> {
//         match signal.ndim() {
//             1 => {
//                 let signal = signal.as_array().into_dimensionality::<Ix1>().unwrap();
//                 let mel_spectrogram = feature::mel_spectrogram1(&signal, self.0);
//                 mel_spectrogram.into_pyarray(self.py()).to_owned()
//             }
//         }
//     }
// }

#[pymodule]
#[pyo3(name = "_internal")]
fn speechsauce(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySpeechSauce>()?;
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
        config: Py<PySpeechSauce>,
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
        config: Py<PySpeechSauce>,
    ) -> &'py PyArrayDyn<f32> {
        let cell = config.as_ref(py);
        let obj_ref = cell.borrow();
        let speech_config = &obj_ref.0;
        let result = match signal.ndim() {
            1 => feature::mel_spectrogram1(
                signal.as_array().into_dimensionality().unwrap(),
                &speech_config,
            )
            .into_dyn(),
            2 => feature::mel_spectrogram2(
                signal.as_array().into_dimensionality().unwrap(),
                &speech_config,
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
        low_frequency: f32,          // =0
        dc_elimination: bool,        //True
        high_frequency: Option<f32>, // =None,
    ) -> Py<PySpeechSauce> {
        Py::new(
            py,
            PySpeechSauce(SpeechConfig::new(
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
