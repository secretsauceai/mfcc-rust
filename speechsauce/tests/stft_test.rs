use ndarray_npy::read_npy;
use num_complex::Complex32;
use serde::Deserialize;
use speechsauce::functions::{stft1, stft2};
use std::path::PathBuf;
// take in a path to a test data and return a list of directories containing the npy files
#[derive(Deserialize, Debug)]
struct StftArgs {
    #[serde(rename = "n_fft")]
    fft_points: usize,
    #[serde(rename = "hop_length")]
    frame_stride: usize,
}

// take in a path to a test data and return a list of directories containing the npy files
fn get_test_data_paths(fn_test_dir: PathBuf) -> (PathBuf, Vec<PathBuf>) {
    let mut test_dirs: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(fn_test_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            test_dirs.push(path);
        }
    }
    todo!()
}

#[test]
fn test_stft2() {
    let (args_file, test_dirs) = get_test_data_paths(PathBuf::from("tests/data/stft2"));
    todo!()

    //let expected_out: ndarray::Array3<Complex32> = read_npy("tests/data/stft2/out.npy").unwrap();
}
