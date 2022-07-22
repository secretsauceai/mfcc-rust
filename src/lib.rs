#![recursion_limit = "512"]
pub mod feature;
pub mod functions;
pub mod processing;
pub mod util;

#[cfg(test)]
mod tests {
    use ndarray::Array;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::{Normal,Uniform};
    use crate::feature::mfcc;
    use crate::processing::cmvn;

    

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
        todo!()
    }

    fn test_stack_frames(){
        todo!()
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
