/// for all the stuff that doesn't yet exist in rust but is outside the scope of the crate

use std::{fmt::format, iter::zip};

use ndarray::{
    azip, Array, Array1, Array2, ArrayBase, ArrayD, ArrayView, ArrayViewMut, Axis, Dim, DimMax,
    Dimension, IntoDimension, IxDyn, IxDynImpl, OwnedRepr, Slice, Zip, ArrayView2,
};



/// from numpy docs:
/// Construct an array by repeating A the number of times given by reps.

/// If `reps` has length `d`, the result will have dimension of `max(d, A.ndim)`.

/// If `A.ndim < d`, `A` is promoted to be d-dimensional by prepending new axes. So a shape (3,) array is promoted to (1, 3) for 2-D replication, or shape (1, 1, 3) for 3-D replication. If this is not the desired behavior, promote `A` to d-dimensions manually before calling this function.

/// If `A.ndim > d`, `reps` is promoted to `A`.ndim by pre-pending 1's to it. Thus for an `A` of shape (2, 3, 4, 5), a `reps` of (2, 2) is treated as (1, 1, 2, 2).

/// Note : Although tile may be used for broadcasting, it is strongly recommended to use numpy's broadcasting operations and functions. */
pub(crate) enum PadType {
    Constant,
    Symmetric,
    Edge, //may add more
}
pub(crate) enum EdgeRowOp {
    Top,
    Bottom,
}
pub(crate) enum EdgeColOp {
    Left,
    Right,
}
///WARNING: this implementation currently doesn't match the behavior of numpy due
/// to issues with the dimensions
pub(crate) fn tile<A, D>(arr: &Array<A, D>, mut reps: Vec<usize>) -> Array<A, IxDyn>
where
    A: Clone + std::fmt::Display + num_traits::Zero,
    D: Dimension,
{
    let num_of_reps = reps.len();

    //just clone the array if reps is all ones

    let mut bail_flag = true;

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

    //to avoid repeatedly casting to IxDyn, and since the input is a 1D array, and the loop takes a 2d array for all ops
    //the result is going to be stored as a 2d array until it's reshaped at the very last step
    let mut res: Array2<A> = arr.to_owned().into_shape((1,arr.len())).unwrap();
    let mut shape_out = res.shape().to_owned();
    
    
    if num_of_reps > res.ndim(){
        _new_shape(res.ndim(), num_of_reps, &mut shape_out);
        //res=res.into_shape(shape_out.clone()).unwrap();
    }
    
    //NOTE: need to revisit this section, also if we can determine the type of shape out
    //we may be able to reduce the two zips to one.
    //shape_out = tuple(s*t for s, t in zip(c.shape(), tup))
    _new_shape(num_of_reps,arr.ndim(),&mut reps);
    println!("shape_out {:?}/n, reps: {:?}", shape_out, reps);
    azip!((a in &mut shape_out, &b in &reps) *a= *a * b);
    
    
    let mut n = res.len();
    
    //used because negative indexing doesn't work
    let mut first_dim: usize;
    
    //the a copy of the input which will be reshaped
    
    if n > 0 {
        //what's going on if reps is larger than shape
        for (dim_in, &nrep) in zip(&mut res.shape().to_owned(), &reps) {
            if nrep != 1 {
                //shape (2,4) should return 4
                println!("res len: {:?}, n: {:?}, nrep: {:?}",res.len(),n,nrep);
                first_dim = res.len()/n;
                //first_dim = if first_dim>0 {first_dim} else {1};
                res = res
                    .into_shape([first_dim, n])
                    .expect(&format!(
                        "error reshaping result into shape ( {:?} , {:?} )",
                        first_dim, n,
                    ));
                println!("starting repeat");
                res = repeat_axis(res.view(),Axis(0), nrep);
                println!("finished repeat");
            }
            n = n / *dim_in;
        }
    }
    
    //res.into_shape(IxDyn(&shape_out));
    return res
        .into_shape(IxDyn(&shape_out))
        .expect("trouble reshaping output");
}

//this works for how repeat is called in our project
pub(crate) fn repeat_axis<A>(arr: ArrayView2<A>, ax:Axis, nrep: usize) -> Array2<A>
where
    A: Clone + std::fmt::Display + num_traits::Zero,
    
{
    println!("orig axis len: {:?}",arr.shape()[0]);
    let repeat_axis_len = arr.shape()[0] * nrep;
    let res = ndarray::concatenate(ax, &vec![arr;nrep]).unwrap();
    // let mut res = ndarray::Array2::<A>::zeros((repeat_axis_len, arr.shape()[1]));
    // println!("res shape: {:?}",res.shape());
    // let repeated_row=arr.row(0);
    // //this works for how repeat is called in our project
    // for mut current_row in res.axis_iter_mut(Axis(0)) {
    //     current_row.assign(&repeated_row.clone()); 
    // }
    res
}

//prepends 
fn _new_shape(current_ndims: usize, min_ndims: usize, reps: &mut Vec<usize>) {
    if current_ndims < min_ndims {
        // in case of fire: https://github.com/numpy/numpy/blob/v1.22.0/numpy/lib/shape_base.py#L1250
        //tup = (1,)*(res.ndim()-num_of_reps) + tup
        //a tuple multiplied by `n` is that same tuple repeated `n` times
        //c is a "copy" of the input array
        //+ concatenates tuples in python
        let mut tmp: Vec<usize> = vec![1; min_ndims - current_ndims];
        tmp.append(reps);
        *reps = tmp;
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

pub(crate) fn pad<A, D>(
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

    let mut padded = Array::from_elem(padded_shape.clone(), const_value);
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

    let mut sub_len: usize;
    //let mut center_flag = false;
    let mut left_b: isize;
    let mut right_b: isize;
    let mut range_start: isize;
    let mut range_stop: isize;

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
                padded_shape.clone(),
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
                padded_shape.clone(),
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
    //let mut sub_len: usize;
    let mut range_start: isize;
    let mut range_stop: isize;
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
        let mut input_slice = input_array.view();
        for (axis, &[lo, hi]) in pad_width.iter().enumerate() {
            if ax == axis {
                portion.slice_axis_inplace(
                    Axis(axis),
                    Slice::from(lo as isize..(padded_shape[axis] - hi) as isize),
                );

                input_slice.slice_axis_inplace(Axis(axis), Slice::from(0_isize..));
            } else {
                portion.slice_axis_inplace(Axis(axis), Slice::from(0..lo));
                input_slice.slice_axis_inplace(Axis(axis), Slice::from(0_isize..1));
            }
        }
        portion.assign(&input_slice);
        let mut portion = padded_array.view_mut();
        for (axis, &[lo, hi]) in pad_width.iter().enumerate() {
            portion.slice_axis_inplace(
                Axis(axis),
                Slice::from(lo as isize..(padded_shape[axis] - hi) as isize),
            );
        }
        portion.assign(&input_array);

        let mut portion = padded_array.view_mut();
        input_slice = input_array.view();
        //start on the last half
        for (axis, &[lo, hi]) in pad_width.iter().enumerate() {
            if ax == axis {
                portion.slice_axis_inplace(
                    Axis(axis),
                    Slice::from(lo as isize..(padded_shape[axis] - hi) as isize),
                );

                input_slice.slice_axis_inplace(Axis(axis), Slice::from(0_isize..));
            } else {
                portion.slice_axis_inplace(Axis(axis), Slice::from(hi..padded_shape[axis]));
                input_slice.slice_axis_inplace(Axis(axis), Slice::from(-1_isize..));
            }
        }
        portion.assign(&input_slice);
    }
}
fn set_edge<A, D>(
    padded_array: &mut ArrayViewMut<A, D>,
    input_array: &ArrayView<A, D>, //todo: fix this, view should be mutable not the underlying input
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
    let mut input_slice = input_array.view();
    for (axis, &[lo, hi]) in pad_width.iter().enumerate() {
        if current_axis == axis as isize {
            match row_op {
                Some(EdgeRowOp::Top) => {
                    input_slice.slice_axis_inplace(Axis(axis), Slice::from(0..sub_len + 1_isize));
                    padded_array.slice_axis_inplace(Axis(axis), Slice::from(0..sub_len + 1));
                }
                Some(EdgeRowOp::Bottom) => {
                    input_slice
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
                    input_slice.slice_axis_inplace(Axis(axis), Slice::from(0_isize..));
                    padded_array.slice_axis_inplace(
                        Axis(axis),
                        Slice::from(lo as isize..(padded_shape[axis] - hi) as isize),
                    );
                }
            }
        } else {
            input_slice.slice_axis_inplace(Axis(axis), Slice::from(0_isize..));
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
    fn log(mut self) -> Array<A, I> {
        self.map_inplace(|n| *n = (*n).ln());
        self
    }
}

#[cfg(test)]
mod test {
    use ndarray::array;
    use ndarray::concatenate;
    use super::*;

    //#[test]
    // fn tile_test() {
    //     let arr1=array![[0,1,2]];
    //     assert_eq!(tile(&arr1,vec![2,2]),array![[0,1,2,0,1,2],[0,1,2,0,1,2]].into_dyn());
    //     assert_eq!(tile(&array![[1,2],[3,4]],vec![2,1]),array![[1,2],[3,4],[1,2],[3,4]].into_dyn());
    // }

    #[test]
    fn tile_equiv_test(){
        let arr1= array![[1,2],[3,4]];
        let arr2=array![[1,2,3,4]];
        //these two assertions are from the examples in the numpy docs for tiling with reps all one save
        //for first value
        //https://numpy.org/doc/stable/reference/generated/numpy.tile.html
        assert_eq!(concatenate![Axis(0),arr1,arr1], array![[1,2],[3,4],[1,2],[3,4]]);
        assert_eq!(repeat_axis(arr2.view(),Axis(0),4),array![[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]])

    }

    #[test]
    fn repeat_test() {
        let input_arr=array![[1,2,3,4]];
        assert_eq!(repeat_axis(input_arr.view(),Axis(0),2),array![[1,2,3,4],[1,2,3,4]])
    }
}
