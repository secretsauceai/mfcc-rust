use std::fmt::format;

use ndarray::{
    azip, Array, Array1, Array2, ArrayBase, ArrayD, ArrayView, ArrayViewMut, Axis, Dim, DimMax,
    Dimension, IntoDimension, IxDyn, IxDynImpl, OwnedRepr, Slice, Zip,
};

//for all the stuff that doesn't exist

/// from numpy docs:
/// Construct an array by repeating A the number of times given by reps.

/// If `reps` has length `d`, the result will have dimension of `max(d, A.ndim)`.

/// If `A.ndim < d`, `A` is promoted to be d-dimensional by prepending new axes. So a shape (3,) array is promoted to (1, 3) for 2-D replication, or shape (1, 1, 3) for 3-D replication. If this is not the desired behavior, promote `A` to d-dimensions manually before calling this function.

/// If `A.ndim > d`, `reps` is promoted to `A`.ndim by pre-pending 1's to it. Thus for an `A` of shape (2, 3, 4, 5), a `reps` of (2, 2) is treated as (1, 1, 2, 2).

/// Note : Although tile may be used for broadcasting, it is strongly recommended to use numpy's broadcasting operations and functions. */
pub enum PadType {
    Constant,
    Symmetric,
    Edge, //may add more
}
enum EdgeRowOp {
    Top,
    Bottom,
}
enum EdgeColOp {
    Left,
    Right,
}
///WARNING: this implementation currently doesn't match the behavior of numpy due
/// to issues with the dimensions
pub(crate) fn tile<A, D>(arr: &Array<A, D>, reps: Vec<usize>) -> Array<A, IxDyn>
where
    A: Clone,
    D: Dimension,
{
    let num_of_reps = reps.len();

    //just clone the array if reps is all ones

    let bail_flag = true;

    for &x in reps.iter() {
        if x != 1 {
            bail_flag = false;
        }
    }

    if bail_flag {
        let mut res_dim = arr.shape().to_owned();

        _new_shape(num_of_reps, arr.ndim(), &mut res_dim);
        let res = arr
            .to_owned()
            .into_shape(Dim(res_dim))
            .expect("something went wrong with the tile function during base case");

        return res;
    }
    //TODO: this may need to be changed, numpy is avoiding allocations unless
    //necessary. This may not be possible in rust
    //see the line: https://github.com/numpy/numpy/blob/v1.22.0/numpy/lib/shape_base.py#L1246
    // and SO post: https://stackoverflow.com/a/27609904/11019565

    //so this seems to pad the number of reps so that the later zip doesn't lose
    //data, but still not getting how this
    _new_shape(num_of_reps, arr.ndim(), &mut reps);
    //NOTE: need to revisit this section, also if we can determine the type of shape out
    //we may be able to reduce the two zips to one.
    //shape_out = tuple(s*t for s, t in zip(c.shape(), tup))
    let mut shape_out = arr.shape().to_owned();
    let mut input_shape = shape_out.clone();
    azip!((a in &mut shape_out, &b in &reps) *a= *a*b);
    //in numpy len(n) returns the length of the outermost axis so shape (1,3,4) -> 1
    let mut n = arr.len_of(Axis(0));
    //the length of the last axis of the results shape at a given step
    let mut last_axis_len: usize = 0;
    //the a copy of the input which will be reshaped
    let mut res: ArrayBase<OwnedRepr<A>, IxDyn> = arr.to_owned().into_dyn();
    if n > 0 {
        //what's going on if reps is larger than shape
        Zip::from(&mut arr.shape().to_owned())
            .and(&reps)
            .for_each(|dim_in, &nrep| {
                if nrep != 1 {
                    //shape (2,4) should return 4
                    last_axis_len = res.shape()[res.shape().len() - 1];
                    res = res
                        .into_shape([last_axis_len, n])
                        .expect(&format!(
                            "error reshaping result into shape {:?} , {:?}",
                            last_axis_len, n,
                        ))
                        .repeat(nrep, 0); //FFFFUUUUUUCKKK!
                                          //TODO: figure out how to repeat elements
                }
                n = n / *dim_in;
            });
    }
    res.into_shape(IxDyn(&shape_out));
    return res;
}

fn _new_shape(num_of_reps: usize, array_dims: usize, reps: &mut Vec<usize>) {
    if num_of_reps < array_dims {
        // in case of fire: https://github.com/numpy/numpy/blob/v1.22.0/numpy/lib/shape_base.py#L1250
        ////tup = (1,)*(res.ndim()-num_of_reps) + tup
        //a tuple multiplied by `n` is that same tuple repeated `n` times
        //c is a "copy" of the input array
        //+ concatenates tuples in python
        let mut tmp: Vec<usize> = vec![1; array_dims - num_of_reps];
        tmp.append(reps);
        *reps = tmp;
    }
}
fn _new_dims(num_of_reps: usize, array_dims: usize, reps: Vec<usize>) -> Dim<IxDynImpl> {
    if num_of_reps < array_dims {
        // in case of fire: https://github.com/numpy/numpy/blob/v1.22.0/numpy/lib/shape_base.py#L1250
        ////tup = (1,)*(res.ndim()-num_of_reps) + tup
        //a tuple multiplied by `n` is that same tuple repeated `n` times
        //c is a "copy" of the input array
        //+ concatenates tuples in python
        let mut tmp: Vec<usize> = vec![1; array_dims - num_of_reps];
        tmp.append(&mut reps);
        Dim(tmp)
    } else {
        Dim(reps)
    }
}
/// Pad the edges of an array with zeros.
///
/// `pad_width` specifies the length of the padding at the beginning
/// and end of each axis.
///
/// **Panics** if `arr.ndim() != pad_width.len()`.
/// TODO: need to replace last arg in call: "edge"
/// see: https://numpy.org/doc/stable/reference/generated/numpy.pad.html?highlight=pad#numpy.pad
/// TODO: currently having issue with the generic arguments, may need to change
/// potentially relevant SO post: https://stackoverflow.com/questions/61758934/how-can-i-write-a-generic-function-that-takes-either-an-ndarray-array-or-arrayvi

pub fn pad<A, D>(
    arr: &Array<A, D>,
    pad_width: Vec<[usize; 2]>,
    const_value: A,
    pad_type: PadType,
) -> Array<A, D>
where
    A: Clone,
    D: Dimension,
{
    assert_eq!(
        arr.ndim(),
        pad_width.len(),
        "Array ndim must match length of `pad_width`."
    );

    // Compute shape of final padded array.
    let mut padded_shape = arr.raw_dim();
    for (ax, (&ax_len, &[pre_ax, post_ax])) in arr.shape().iter().zip(&pad_width).enumerate() {
        padded_shape[ax] = ax_len + pre_ax + post_ax;
    }

    let mut padded = Array::from_elem(padded_shape, const_value);
    let padded_dim = padded.raw_dim();
    {
        // Select portion of padded array that needs to be copied from the
        // original array.
        let mut orig_portion = padded.view_mut();
        // Copy the data from the original array.
        orig_portion.assign(arr);
        match pad_type {
            PadType::Constant => (),
            PadType::Symmetric => {
                /*
                >>>a = [1, 2, 3, 4, 5]
                >>>np.pad(a, (2, 3), 'symmetric')
                array([2, 1, 1, 2, 3, 4, 5, 5, 4, 3])
                */

                symmetric_pad(&mut padded, arr, pad_width, padded_shape);
            }
            PadType::Edge => {
                /*
                >>>a = [1, 2, 3, 4, 5]
                >>>np.pad(a, (2, 3), 'edge')
                [out]: array([1, 1, 1, ..., 5, 5, 5])
                */
                edge_pad(&mut padded, arr, pad_width, padded_shape);
            }
        }
    }
    padded
}

fn symmetric_pad<A, D>(
    padded_array: &mut Array<A, D>,
    input_array: &Array<A, D>,
    pad_width: Vec<[usize; 2]>,
    padded_shape: D,
) where
    A: Clone,
    D: Dimension,
{
    let mut a_inv = input_array.clone();

    let mut sub_len = 0_usize;
    //let mut center_flag = false;
    let mut left_b: isize = 0;
    let mut right_b: isize = 0;
    let mut range_start: isize = 0;
    let mut range_stop: isize = 0;

    //right now this only handles the cross sections
    //since the input is always 0 padded along one axis this
    //works for our use case

    for (ax, (&ax_len, &[pre_ax, post_ax])) in
        input_array.shape().iter().zip(&pad_width).enumerate()
    {
        //the outer loops controls which axis we are tiling along
        if pre_ax == 0 && post_ax == 0 {
            //diag_flag=false;
            continue;
        }

        a_inv.invert_axis(Axis(ax));

        range_start = if pre_ax > 0 {
            0 - (pre_ax / ax_len) as isize
        } else {
            0
        };
        range_stop = if post_ax > 0 {
            (post_ax / ax_len) as isize + 1
        } else {
            1
        };

        //get the length of the leftover that won't fit a complete tile
        sub_len = pre_ax % ax_len;
        if sub_len != 0 {
            let mut portion = padded_array.view_mut();
            //need to slice the input as well as the output
            let mut original_slice = if ((pre_ax / ax_len) + 2) % 2 == 0 {
                input_array.view()
            } else {
                a_inv.view()
            };

            set_edge::<A, D>(
                &mut portion,
                &mut original_slice,
                sub_len as isize,
                padded_shape,
                &pad_width,
                ax as isize,
                Some(EdgeRowOp::Top),
            );
        }
        for i in range_start..range_stop {
            let mut portion = padded_array.view_mut();

            for (axis, &[lo, hi]) in pad_width.iter().enumerate() {
                //draw a box around the ROI

                right_b = if hi > 0 {
                    padded_shape[axis] as isize - hi as isize + (i * ax_len as isize)
                } else {
                    padded_shape[axis] as isize - hi as isize
                };

                left_b = if lo > 0 {
                    lo as isize + (i * ax_len as isize)
                } else {
                    lo as isize
                };

                if right_b - left_b != 0 {
                    portion.slice_axis_inplace(Axis(axis), Slice::from(left_b..right_b));
                }
                if i > 0 {
                    portion.assign(&input_array);
                } else {
                    portion.assign(&a_inv);
                }
            }
        }
        //check to see if there is any leftover
        sub_len = post_ax % ax_len;
        if sub_len != 0 {
            let mut portion = padded_array.view_mut();
            let mut original_slice = if ((post_ax / ax_len) + 1) % 2 != 0 {
                input_array.view()
            } else {
                a_inv.view()
            };
            set_edge::<A, D>(
                &mut portion,
                &mut original_slice,
                sub_len as isize,
                padded_shape,
                &pad_width,
                ax as isize,
                Some(EdgeRowOp::Bottom),
            );
        }

        //flip it back so it works for the next axis
        a_inv.invert_axis(Axis(ax));
    }
}

fn edge_pad<A, D>(
    padded_array: &mut Array<A, D>,
    input_array: &Array<A, D>,
    pad_width: Vec<[usize; 2]>,
    padded_shape: D,
) where
    A: Clone,
    D: Dimension,
{
    let mut sub_len = 0_usize;
    let mut range_start: isize = 0;
    let mut range_stop: isize = 0;
    for (ax, (&ax_len, &[pre_ax, post_ax])) in
        input_array.shape().iter().zip(&pad_width).enumerate()
    {
        //the outer loops controls which axis we are tiling along
        if pre_ax == 0 && post_ax == 0 {
            //diag_flag=false;
            continue;
        }

        range_start = if pre_ax > 0 {
            0 - (pre_ax / ax_len) as isize
        } else {
            0
        };
        range_stop = if post_ax > 0 {
            (post_ax / ax_len) as isize + 1
        } else {
            1
        };
        let mut portion = padded_array.view_mut();
        let mut input_slice = input_array.view_mut();
        for (axis, &[lo, hi]) in pad_width.iter().enumerate() {
            if ax == axis {
                portion.slice_axis_inplace(
                    Axis(axis),
                    Slice::from(lo as isize..(padded_shape[axis] - hi) as isize),
                );

                input_slice.slice_axis_inplace(Axis(axis), Slice::from(0..));
            } else {
                portion.slice_axis_inplace(Axis(axis), Slice::from(0..lo));
                input_slice.slice_axis_inplace(Axis(axis), Slice::from(0..1));
            }
        }
        portion.assign(&input_slice);
        portion = padded_array.view_mut();
        for (axis, &[lo, hi]) in pad_width.iter().enumerate() {
            padded_array.slice_axis_inplace(
                Axis(axis),
                Slice::from(lo as isize..(padded_shape[axis] - hi) as isize),
            );
        }
        portion.assign(&input_array);

        portion = padded_array.view_mut();
        input_slice = input_array.view_mut();
        //start on the last half
        for (axis, &[lo, hi]) in pad_width.iter().enumerate() {
            if ax == axis {
                portion.slice_axis_inplace(
                    Axis(axis),
                    Slice::from(lo as isize..(padded_shape[axis] - hi) as isize),
                );

                input_slice.slice_axis_inplace(Axis(axis), Slice::from(0..));
            } else {
                portion.slice_axis_inplace(Axis(axis), Slice::from(hi..padded_shape[axis]));
                input_slice.slice_axis_inplace(Axis(axis), Slice::from(-1..));
            }
        }
        portion.assign(&input_slice);
    }
}
fn set_edge<A, D>(
    padded_array: &mut ArrayViewMut<A, D>,
    input_array: &mut ArrayView<A, D>, //todo: fix this, view should be mutable not the underlying input
    sub_len: isize,
    padded_shape: D,
    pad_width: &Vec<[usize; 2]>,
    current_axis: isize,
    row_op: Option<EdgeRowOp>,
    //col_op: Option<EdgeColOp>,
) where
    A: Clone,
    D: Dimension,
{
    for (axis, &[lo, hi]) in pad_width.iter().enumerate() {
        if current_axis == axis as isize {
            match row_op {
                Some(EdgeRowOp::Top) => {
                    input_array.slice_axis_inplace(Axis(axis), Slice::from(0..sub_len + 1_isize));
                    padded_array.slice_axis_inplace(Axis(axis), Slice::from(0..sub_len + 1));
                }
                Some(EdgeRowOp::Bottom) => {
                    input_array
                        .slice_axis_inplace(Axis(axis), Slice::from(0 as isize..sub_len + 1_isize));
                    padded_array.slice_axis_inplace(
                        Axis(axis),
                        Slice::from(
                            padded_shape[axis] as isize - (sub_len + 1)
                                ..padded_shape[axis] as isize,
                        ),
                    );
                }
                None => {
                    input_array.slice_axis_inplace(Axis(axis), Slice::from(0_isize..));
                    padded_array.slice_axis_inplace(
                        Axis(axis),
                        Slice::from(lo as isize..(padded_shape[axis] - hi) as isize),
                    );
                }
            }
        } else {
            input_array.slice_axis_inplace(Axis(axis), Slice::from(0_isize..));
            padded_array.slice_axis_inplace(
                Axis(axis),
                Slice::from(lo as isize..(padded_shape[axis] - hi) as isize),
            );
        }
    }
    padded_array.assign(&input_array);
}

/// A simple trait to implement log operation for matrixes
pub trait ArrayLog<A: num_traits::real::Real, I: ndarray::Dimension> {
    fn log(self) -> Array<A, I>;
}

impl<A: num_traits::real::Real, I: ndarray::Dimension> ArrayLog<A, I> for Array<A, I> {
    fn log(self) -> Array<A, I> {
        self.map_inplace(|n| *n = (*n).ln());
        self
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn tile_test() {
        todo!()
    }
}
