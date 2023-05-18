/// contains necessary functions for calculating the features in the `features` module.
use ndarray::{Array, Array1, Dimension, Zip};

/// converts a single value representing a frequency in Hz to Mel scale.
pub fn frequency_to_mel(f: f32) -> f32 {
    1127. * (1. + f / 700.).ln()
}
///converts all values in the input array from frequency to mel scale
pub fn frequency_arr_to_mel<D>(freq: Array<f32, D>) -> Array<f32, D>
where
    D: Dimension,
{
    freq.mapv(|v| 1127. * (1. + v / 700.).ln())
}

/// converts a single value from the mel scale to a frequency scale.
pub fn mel_to_frequency<D>(mel: f32) -> f32 {
    700. * ((mel / 1127.0).exp() - 1.)
}

///converts all values in the input array from mel scale to frequency scale
pub fn mel_arr_to_frequency<D>(mel: Array<f32, D>) -> Array<f32, D>
where
    D: Dimension,
{
    mel.mapv(|v| 700. * ((v / 1127.0).exp() - 1.))
}

pub fn triangle(arr: Array1<f32>, left: f32, middle: f32, right: f32) -> Array1<f32> {
    //original function: https://github.com/astorfi/speechpy/blob/master/speechpy/functions.py#L44
    let mut out = ndarray::Array1::<f32>::zeros(arr.len());
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
pub fn zero_handling<D>(x: Array<f32, D>) -> Array<f32, D>
where
    D: Dimension,
{
    x.mapv(|x| if x == 0.0 { std::f32::EPSILON } else { x })
}
