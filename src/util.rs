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
    let mut b = a.clone();
    let mut inv_flag = false;
    let mut diag_flag = true;
    let mut start = 0_usize;
    let mut center_flag = false;
    let mut left_b: isize = 0;
    let mut right_b: isize = 0;

    let mut padded_shape = a.raw_dim();
    for (ax, (&ax_len, &[pre_ax, post_ax])) in a.shape().iter().zip(&pad_width).enumerate() {
        padded_shape[ax] = ax_len + pre_ax + post_ax;
    }
    let mut padded = Array::from_elem(padded_shape, const_value);
    let padded_dim = padded.raw_dim();

    for (ax, (&ax_len, &[pre_ax, post_ax])) in a.shape().iter().zip(&pad_width).enumerate() {
        if pre_ax == 0 && post_ax == 0 {
            continue;
        }
        println!("loop {}", ax);
        println!("ax_len: {}", ax_len);
        b.invert_axis(Axis(ax));
        start = pre_ax % ax_len;
        //if (start!=0){
        //    todo!();
        //}
        if pre_ax > 0 {
            //all "tiles" upto original axis
            println!("pre_ax started");
            for i in (0..(pre_ax / ax_len) as usize).rev() {
                println!("inside loop{}", i);
                let mut orig_portion = padded.view_mut();
                for (axis, &[lo, hi]) in pad_width.iter().enumerate() {
                    //draw a box around the ROI
                    println!("lo: {}, hi: {}", lo, hi);
                    right_b = if hi > 0 {
                        padded_shape[axis] as isize - (hi + i * ax_len) as isize
                    } else {
                        padded_shape[axis] as isize - hi as isize
                    };
                    println!("right_b: {}", right_b);
                    left_b = if lo > 0 {
                        lo as isize - (i * ax_len) as isize
                    } else {
                        lo as isize
                    };
                    println!("left_b: {}", left_b);
                    if right_b - left_b != 0 {
                        orig_portion.slice_axis_inplace(Axis(axis), Slice::from(left_b..right_b));
                    }
                    orig_portion.assign(&a);
                }
            }
            println!("pre_ax ended");
        } else if ax == 0 {
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
            println!("post_ax started");
            for i in 1..(post_ax / ax_len) as usize {
                let mut orig_portion = padded.view_mut();
                for (axis, &[lo, hi]) in pad_width.iter().enumerate() {
                    println!("lo: {}, hi: {}", lo, hi);
                    right_b = if hi > 0 {
                        padded_shape[axis] as isize - (hi - i * ax_len) as isize
                    } else {
                        padded_shape[axis] as isize - hi as isize
                    };
                    println!("right_b: {}", right_b);
                    left_b = if lo > 0 {
                        lo as isize + (i * ax_len) as isize
                    } else {
                        lo as isize
                    };
                    println!("left_b: {}", left_b);
                    orig_portion.slice_axis_inplace(Axis(axis), Slice::from(left_b..right_b));
                }
                orig_portion.assign(&a);
            }
        }
        b.invert_axis(Axis(ax));
    }
    padded
}
