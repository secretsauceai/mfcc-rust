/// contains necessary functions for calculating the features in the `features` module.
use ndarray::{Array, Array1, ArrayViewMut1, Dimension};


/// converts a single value representing a frequency in Hz to Mel scale.
pub fn frequency_to_mel(f: f64) -> f64 {
    1127. * (1. + f / 700.).ln()
}
///converts all values in the input array from frequency to mel scale
pub fn frequency_arr_to_mel<D>(freq: Array<f64,D>) -> Array<f64,D> 
where D: Dimension{
    freq.mapv(|v| 1127. * (1. + v / 700.).ln())
}

/// converts a single value from the mel scale to a frequency scale.
pub fn mel_to_frequency<D>(mel: f64)-> f64 {
    700. * ((mel / 1127.0).exp() - 1.)
}

///converts all values in the input array from mel scale to frequency scale
pub fn mel_arr_to_frequency<D>(mel: Array<f64,D>) -> Array<f64,D> 
where D: Dimension{
    mel.mapv(|v| 700. * ((v / 1127.0).exp() - 1.))
}


pub fn triangle(arr: &mut ArrayViewMut1<f64>, x: Array1<f64>, left: f64, middle: f64, right: f64) {
    //original function: https://github.com/astorfi/speechpy/blob/master/speechpy/functions.py#L44

    //arr[x <= left] = 0;
    //arr[x >= right] = 0;
    arr.indexed_iter_mut().for_each(|(i, v)| {
        if (left..right).contains(v) {
            //TODO: fix range bounds to be exclusive, see https://github.com/rust-lang/rust/issues/37854
            if *v <= middle {
                *v = (x[i] - left) / (middle - left);
            } //NOTE: depending on whether the double <= >= is intended or not, may be simplified to just else
            if middle <= *v {
                *v = (right - x[i]) / (right - middle);
            }
        } else {
            *v = 0.0
        }
    });
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
