use std::{cell::RefCell, sync::Arc};

use ndarray::Array2;
use ndrustfft::{DctHandler, R2cFftHandler};
use num_complex::Complex32;

use crate::feature::filterbanks;
use realfft::{RealFftPlanner, RealToComplex};

#[derive(Default)]
pub struct SpeechConfigBuilder {
    ///sampling frequency of the signal
    sample_rate: usize,
    /// number of FFT points.
    fft_points: usize,
    /// the length of each frame in seconds.
    frame_length: f32, // =0.020,
    /// the step between successive frames in seconds.
    frame_stride: f32, // =0.01,
    /// Number of cepstral coefficients.
    num_cepstral: usize, // =13,
    /// the number of filters in the filterbank
    num_filters: usize, // =40,
    ///lowest band edge of mel filters in Hz
    low_frequency: f32,
    ///highest band edge of mel filters in Hz.
    high_frequency: f32,
    /// If the first dc component should be eliminated or not
    dc_elimination: bool,
    // for mel_spectrogram
    //power: u8,
}

impl SpeechConfigBuilder {
    pub fn new(sample_rate: usize) -> SpeechConfigBuilder {
        SpeechConfigBuilder {
            sample_rate,
            fft_points: 512,
            frame_length: 0.02,
            frame_stride: 0.01,
            num_cepstral: 13,
            num_filters: 40,
            low_frequency: 0.0,
            high_frequency: sample_rate as f32 / 2.0,
            dc_elimination: true,
        }
    }

    pub fn high_freq(mut self, high_frequency: f32) -> SpeechConfigBuilder {
        self.high_frequency = high_frequency;
        self
    }

    pub fn dc_elimination(mut self, dc_elimination: bool) -> SpeechConfigBuilder {
        self.dc_elimination = dc_elimination;
        self
    }

    pub fn low_freq(mut self, low_frequency: f32) -> SpeechConfigBuilder {
        self.low_frequency = low_frequency;
        self
    }

    pub fn num_cepstral(mut self, num_cepstral: usize) -> SpeechConfigBuilder {
        self.num_cepstral = num_cepstral;
        self
    }

    pub fn frame_stride(mut self, frame_stride: f32) -> SpeechConfigBuilder {
        self.frame_stride = frame_stride;
        self
    }

    pub fn frame_length(mut self, frame_length: f32) -> SpeechConfigBuilder {
        self.frame_length = frame_length;
        self
    }

    pub fn fft_points(mut self, fft_points: usize) -> SpeechConfigBuilder {
        self.fft_points = fft_points;
        self
    }

    pub fn build(self) -> SpeechConfig {
        SpeechConfig::new(
            self.sample_rate,
            self.fft_points,
            self.frame_length,
            self.frame_stride,
            self.num_cepstral,
            self.num_filters,
            self.low_frequency,
            self.high_frequency,
            self.dc_elimination,
        )
    }
}

#[derive(Clone)]
pub struct SpeechConfig {
    ///sampling frequency of the signal
    pub sample_rate: usize,
    /// number of FFT points.
    pub window_size: usize,
    pub window_size_half: usize,
    /// the length of each frame in seconds.
    pub frame_length: f32, // =0.020,
    /// the step between successive frames in seconds.
    pub frame_stride: f32, // =0.01,
    /// Number of cepstral coefficients.
    pub num_cepstral: usize, // =13,
    /// the number of filters in the filterbank
    pub num_filters: usize, // =40,
    ///lowest band edge of mel filters in Hz
    pub low_frequency: f32,
    ///highest band edge of mel filters in Hz.
    pub high_frequency: f32,
    /// Number of frequency bins, which is (window_size/2)+1
    pub freq_size: usize,
    /// Number of samples in a frame, which is frame_length * sample_rate
    pub frame_size: usize,
    /// If the first dc component should be eliminated or not
    pub dc_elimination: bool,
    pub wnorm: f32,
    pub window: Vec<f32>,
    pub analysis_mem: RefCell<Vec<f32>>,
    pub dct_handler: DctHandler<f32>,
    pub fft_handler: R2cFftHandler<f32>,
    pub fft_forward: Arc<dyn RealToComplex<f32>>,
    pub analysis_scratch: RefCell<Vec<Complex32>>,
}

impl Default for SpeechConfig {
    fn default() -> Self {
        SpeechConfigBuilder::new(16000).build()
    }
}

impl SpeechConfig {
    pub fn new(
        sample_rate: usize,
        fft_points: usize,
        frame_length: f32,
        frame_stride: f32,
        num_cepstral: usize,
        num_filters: usize,
        low_frequency: f32,
        high_frequency: f32,
        dc_elimination: bool,
    ) -> Self {
        // Initialize the vorbis window: sin(pi/2*sin^2(pi*n/N))
        let pi = std::f64::consts::PI;
        let mut fft = RealFftPlanner::<f32>::new();
        let frame_size = (frame_length * sample_rate as f32) as usize;
        let window_size_h = fft_points / 2;
        let mut window = vec![0.0; fft_points];
        for (i, w) in window.iter_mut().enumerate() {
            let sin = (0.5 * pi * (i as f64 + 0.5) / window_size_h as f64).sin();
            *w = (0.5 * pi * sin * sin).sin() as f32;
        }
        let forward: Arc<dyn RealToComplex<f32>> = fft.plan_fft_forward(fft_points);
        let analysis_mem = RefCell::new(vec![0.; fft_points - frame_size]);
        let analysis_scratch = RefCell::new(forward.make_scratch_vec());
        Self {
            dct_handler: DctHandler::new(num_filters),
            fft_handler: R2cFftHandler::new(fft_points),
            fft_forward: forward,
            window_size: fft_points,
            window_size_half: window_size_h,
            frame_length,
            frame_stride,
            num_cepstral,
            num_filters,
            low_frequency,
            high_frequency,
            dc_elimination,
            sample_rate,
            wnorm: 1.0 / (fft_points.pow(2) as f32 / (2 * frame_size) as f32),
            freq_size: (fft_points / 2) + 1,
            frame_size: frame_size,
            window,
            analysis_mem,
            analysis_scratch: analysis_scratch,
        }
    }

    pub fn builder() -> SpeechConfigBuilder {
        SpeechConfigBuilder::default()
    }
}
