use ndrustfft::{DctHandler, R2cFftHandler};

#[derive(Default)]
pub struct MfccBuilder {
    ///sampling frequency of the signal
    sample_rate: usize,
    /// number of FFT points.
    fft_points: usize,
    /// the length of each frame in seconds.
    frame_length: f64, // =0.020,
    /// the step between successive frames in seconds.
    frame_stride: f64, // =0.01,
    /// Number of cepstral coefficients.
    num_cepstral: usize, // =13,
    /// the number of filters in the filterbank
    num_filters: usize, // =40,
    ///lowest band edge of mel filters in Hz
    low_frequency: f64,
    ///highest band edge of mel filters in Hz.
    high_frequency: f64,
    /// If the first dc component should be eliminated or not
    dc_elimination: bool,
}

impl MfccBuilder {
    fn new(sampling_frequency: usize) -> MfccBuilder {
        MfccBuilder {
            sample_rate,
            fft_points: 512,
            frame_length: 0.02,
            frame_stride: 0.02,
            num_cepstral: 13,
            num_filters: 40,
            low_frequency: 0.0,
            high_frequency: sample_rate as f64 / 2.0,
            dc_elimination: True,
        }
    }

    pub fn high_freq(mut self, high_frequency: f64) -> MfccBuilder {
        self.high_frequency = high_frequency;
        self
    }

    pub fn dc_elimination(mut self, dc_elimination: bool) -> MfccBuilder {
        self.dc_elimination = dc_elimination;
        self
    }

    pub fn low_freq(mut self, low_frequency: f64) -> MfccBuilder {
        self.low_frequency = low_frequency;
        self
    }

    pub fn num_cepstral(mut self, num_cepstral: usize) -> MfccBuilder {
        self.num_cepstral = num_cepstral;
        self
    }

    pub fn frame_stride(mut self, frame_stride: f64) -> MfccBuilder {
        self.frame_stride = frame_stride;
        self
    }

    pub fn frame_length(mut self, frame_length: f64) -> MfccBuilder {
        self.frame_length = frame_length;
        self
    }

    pub fn fft_points(mut self, fft_points: usize) -> MfccBuilder {
        self.fft_points = fft_points;
        self
    }

    pub fn build(self) -> Mfcc {
        Mfcc::new(
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

pub struct Mfcc {
    dct_handler: DctHandler<f64>,
    fft_handler: R2cFftHandler<f64>,
    ///sampling frequency of the signal
    sample_rate: usize,
    /// number of FFT points.
    fft_points: usize,
    /// the length of each frame in seconds.
    frame_length: f64, // =0.020,
    /// the step between successive frames in seconds.
    frame_stride: f64, // =0.01,
    /// Number of cepstral coefficients.
    num_cepstral: usize, // =13,
    /// the number of filters in the filterbank
    num_filters: usize, // =40,
    ///lowest band edge of mel filters in Hz
    low_frequency: f64,
    ///highest band edge of mel filters in Hz.
    high_frequency: f64,
    /// If the first dc component should be eliminated or not
    dc_elimination: bool,
}

impl Mfcc {
    pub fn new(
        sample_rate: usize,
        fft_points: usize,
        frame_length: f64,
        frame_stride: f64,
        num_cepstral: usize,
        num_filters: usize,
        low_frequency: f64,
        high_frequency: f64,
        dc_elimination: bool,
    ) -> Self {
        Self {
            dct_handler: DctHandler::new(num_filters),
            fft_handler: R2cFftHandler::new(fft_points),
            sample_rate,
            fft_points,
            frame_length,
            frame_stride,
            num_cepstral,
            num_filters,
            low_frequency,
            high_frequency,
            dc_elimination,
        }
    }

    pub fn builder() -> MfccBuilder {
        MfccBuiler::default()
    }
}
