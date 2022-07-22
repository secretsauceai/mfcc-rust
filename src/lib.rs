#![recursion_limit = "512"]
pub mod feature;
pub mod functions;
pub mod processing;
pub mod util;

#[cfg(test)]
mod tests {
    use ndarray::{Array, Array1};
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::{Normal,Uniform};
    use crate::feature::mfcc;
    use crate::functions::frequency_to_mel;
    use crate::processing::{cmvn, preemphasis, stack_frames};

    

    fn create_signal() -> ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> {
        let mu=0.; //mean
        let sigma=0.1; //standard deviation
        Array::random(1000000,Normal::new(mu,sigma).unwrap())
    }

    fn sampling_frequency() -> usize {
        16000
    }

    #[test]
    fn test_preprocessing(){
        let signal = create_signal();
        let coeff = 0.98;
        let signal_preemphasized = preemphasis(signal.clone(),1,coeff);
        assert_eq!(signal_preemphasized.ndim(),1);
        assert_eq!(signal_preemphasized.shape(), signal.shape());
    }

    fn test_stack_frames(){
        let signal = create_signal();
        let freq=sampling_frequency();
        let frame_length=0.02;
        let frame_stride=0.02;
        let filter: fn(usize) -> ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> = |x:usize| -> ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> {Array1::<f64>::ones(x)};
        let zero_padding=true;
        let frames=stack_frames(signal.clone(), freq, frame_length, frame_stride, filter, zero_padding);
        let window = (frame_length * freq as f64).round();
        let step = (frame_stride * freq as f64).round();
        let all_frames = ((signal.shape()[0] as f64 - window)/step).ceil() as usize;
        assert_eq!(all_frames,frames.shape()[0])

    } 

    #[test]
    fn test_cmvn() {
        let variance_normalization=true;
        let feature_vector=Array::random((50,100),Uniform::new(0.,1.));
        let normalized_feature=cmvn(feature_vector.clone(),variance_normalization);
        assert_eq!(normalized_feature.shape(),feature_vector.shape());
        
        //check the standard deviation and mean of the output vector
        let output_std=normalized_feature.std_axis(ndarray::Axis(0),0.);
        let output_mean=normalized_feature.mean_axis(ndarray::Axis(0)).unwrap();
        //TODO: verify the shape of cvmn and np.zeroes((1,x))
        //should be comparing two 1d arrays in original code
        assert!(output_std.abs_diff_eq(&ndarray::Array1::zeros((normalized_feature.shape()[1])),1e-8));
        assert!(output_mean.abs_diff_eq(&ndarray::Array1::ones((normalized_feature.shape()[1])),1e-8));
    }

    #[test]
    fn test_mfcc() {
        
        let num_cepstral : usize=13;

        let sampling_frequency=16000;
        let frame_length=0.02;
        let frame_stride = 0.01;
        let num_filters=40;
        let fft_length=512;
        let low_frequency=0.;
        let hi_frequency=None;
        let dc_elination=true;

        let signal=create_signal();
        let mfcc=mfcc(signal,sampling_frequency,frame_length,frame_stride, num_cepstral, num_filters,fft_length,low_frequency,hi_frequency,dc_elination);
        assert_eq!(mfcc.shape()[1], num_cepstral);
    }
}
