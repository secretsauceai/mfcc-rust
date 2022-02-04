use ndarray::{Array, ArrayBase, Axis, Dimension, Slice};

//for all the stuff that doesn't exist

/// from numpy docs:
/// Construct an array by repeating A the number of times given by reps.

/// If `reps` has length `d`, the result will have dimension of `max(d, A.ndim)`.

/// If `A.ndim < d`, `A` is promoted to be d-dimensional by prepending new axes. So a shape (3,) array is promoted to (1, 3) for 2-D replication, or shape (1, 1, 3) for 3-D replication. If this is not the desired behavior, promote `A` to d-dimensions manually before calling this function.

/// If `A.ndim > d`, `reps` is promoted to `A`.ndim by pre-pending 1's to it. Thus for an `A` of shape (2, 3, 4, 5), a `reps` of (2, 2) is treated as (1, 1, 2, 2).

/// Note : Although tile may be used for broadcasting, it is strongly recommended to use numpy's broadcasting operations and functions. */
enum PadType {
    Constant,
    Symmetric,
    Edge, //may add more
}
pub fn tile<A, S, D>(arr: &ArrayBase<S, D>, reps: &ArrayBase<S, D>) -> Array<A, D>
where
    A: Clone,
    S: ndarray::Data<Elem = A>,
    D: Dimension,
{
    unimplemented!()
}

/// Pad the edges of an array with zeros.
///
/// `pad_width` specifies the length of the padding at the beginning
/// and end of each axis.
///
/// **Panics** if `arr.ndim() != pad_width.len()`.
/// TODO: need to replace last arg in call: "edge"
/// see: https://numpy.org/doc/stable/reference/generated/numpy.pad.html?highlight=pad#numpy.pad
pub fn pad<S, A, D>(
    arr: &ArrayBase<S, D>,
    pad_width: Vec<[usize; 2]>,
    const_value: A,
    pad_type: PadType, //Enum?
) -> Array<A, D>
where
    S: ndarray::Data<Elem = A>,
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
        for (axis, &[pre_ax, post_ax]) in pad_width.iter().enumerate() {
            orig_portion.slice_axis_inplace(
                Axis(axis),
                Slice::from(pre_ax as isize..padded_shape[axis] as isize - (post_ax as isize)),
            );
        }
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

                todo!()
            }
            PadType::Edge => {
                /*
                >>>a = [1, 2, 3, 4, 5]
                >>>np.pad(a, (2, 3), 'edge')
                [out]: array([1, 1, 1, ..., 5, 5, 5])
                */
                todo!()
            }
        }
    }
    padded
}

fn symmetric_pad<S, A, D>(
    a: &ArrayBase<S, D>,
    pad_width: Vec<[usize; 2]>,
    const_value: A,
    pad_type: PadType, //Enum?
) -> Array<A, D>
where
    S: ndarray::Data<Elem = A>,
    A: Clone,
    D: Dimension,
{
    let mut a_inv = a.clone();

    //let mut diag_flag = true; //we don't need it
    let mut sub_len = 0_usize;
    let mut center_flag = false;
    let mut left_b: isize = 0;
    let mut right_b: isize = 0;

    let mut padded_shape = a.raw_dim();
    for (ax, (&ax_len, &[pre_ax, post_ax])) in a.shape().iter().zip(&pad_width).enumerate() {
        padded_shape[ax] = ax_len + pre_ax + post_ax;
    }
    let mut padded = Array::from_elem(padded_shape, const_value);
    let padded_dim = padded.raw_dim();
    //right now this only handles the cross sections
    //since the input is always 0 padded along one axis this
    //works for our use case

    for (ax, (&ax_len, &[pre_ax, post_ax])) in a.shape().iter().zip(&pad_width).enumerate() {
        //the outer loops controls which axis we are tiling along
        if pre_ax == 0 && post_ax == 0 {
            //diag_flag=false;
            continue;
        }

        a_inv.invert_axis(Axis(ax));

        if pre_ax > 0 {
            //all "tiles" preceding original along axis

            //get the length of the leftover that won't fit a complete tile
            sub_len = pre_ax % ax_len;
            if sub_len != 0 {
                let mut portion = padded.view_mut();
                for (axis, &[lo, hi]) in pad_width.iter().enumerate() {
                    left_b = if ax == axis { 0 } else { lo as isize };
                    right_b = if ax == axis {
                        sub_len as isize
                    } else {
                        padded_shape[axis] as isize - hi as isize
                    };
                    if right_b - left_b != 0 {
                        portion.slice_axis_inplace(Axis(axis), Slice::from(left_b..right_b));
                    }
                }
                if ((pre_ax / ax_len) + 2) % 2 == 0 {
                    todo!();
                } else {
                    todo!();
                }
            }
            for i in (1..(pre_ax / ax_len) + 1 as usize).rev() {
                let mut portion = padded.view_mut();
                for (axis, &[lo, hi]) in pad_width.iter().enumerate() {
                    //draw a box around the ROI

                    right_b = if hi > 0 {
                        padded_shape[axis] as isize - (hi + i * ax_len) as isize
                    } else {
                        padded_shape[axis] as isize - hi as isize
                    };

                    left_b = if lo > 0 {
                        lo as isize - (i * ax_len) as isize
                    } else {
                        lo as isize
                    };

                    if right_b - left_b != 0 {
                        portion.slice_axis_inplace(Axis(axis), Slice::from(left_b..right_b));
                    }
                    if i % 2 == 0 {
                        portion.assign(&a);
                    } else {
                        portion.assign(&a_inv);
                    }
                }
            }
        }
        if ax == 0 {
            //only once for the actual original axis

            let mut orig_portion = padded.view_mut();
            for (axis, &[lo, hi]) in pad_width.iter().enumerate() {
                orig_portion.slice_axis_inplace(
                    Axis(axis),
                    Slice::from(lo as isize..padded_shape[axis] as isize - (hi as isize)),
                );
            }
            orig_portion.assign(&a);
        }
        if post_ax > 0 {
            //now for the tiles to the right of the original
            for i in 1..(post_ax / ax_len) + 1 as usize {
                let mut orig_portion = padded.view_mut();
                for (axis, &[lo, hi]) in pad_width.iter().enumerate() {
                    right_b = if hi > 0 {
                        padded_shape[axis] as isize - (hi - i * ax_len) as isize
                    } else {
                        padded_shape[axis] as isize - hi as isize
                    };

                    left_b = if lo > 0 {
                        lo as isize + (i * ax_len) as isize
                    } else {
                        lo as isize
                    };

                    orig_portion.slice_axis_inplace(Axis(axis), Slice::from(left_b..right_b));
                }
                if i % 2 == 0 {
                    orig_portion.assign(&a);
                } else {
                    orig_portion.assign(&a_inv);
                }
            }
            //check to see if there is any leftover
            sub_len = post_ax % ax_len;
            if sub_len != 0 {
                for (axis, &[lo, hi]) in pad_width.iter().enumerate() {
                    left_b = if ax == axis {
                        padded_shape[axis] as isize - sub_len as isize
                    } else {
                        lo as isize
                    };
                    right_b = if ax == axis {
                        -1
                    } else {
                        padded_shape[axis] as isize - hi as isize
                    };
                }
                if ((post_ax / ax_len) + 2) % 2 == 0 {
                    todo!();
                } else {
                    todo!();
                }
            }
        }
        //flip it back so it works for the next axis
        a_inv.invert_axis(Axis(ax));
    }
    padded
}
