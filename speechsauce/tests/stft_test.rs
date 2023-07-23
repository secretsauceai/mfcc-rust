use ndarray_npy::read_npy;
use num_complex::Complex32;
use serde::Deserialize;
use speechsauce::{
    config::AnalysisState,
    functions::{stft1, stft2},
};
/* Test data is currently organized in the following hierarchy
fn_name dir
|- hash_of_args dir
|   |- args.json
|   |- uuid dir
|   |  |- in.npy
|   |  |- out.npy
|   |- ...
|- ...
test are currently grouped by non-array args*/
use std::path::PathBuf;
// take in a path to a test data and return a list of directories containing the npy files
#[derive(Deserialize, Debug)]
struct StftArgs {
    #[serde(rename = "n_fft")]
    fft_points: usize,
    #[serde(rename = "hop_length")]
    frame_stride: usize,
    center: bool,
}

fn config_from_args_file(args_file: PathBuf) -> AnalysisState {
    let args: StftArgs =
        serde_json::from_str(&std::fs::read_to_string(args_file).unwrap()).unwrap();
    let state = AnalysisState::new(args.fft_points, args.frame_stride);
    state
}

// take in a path to a test data and return a list of directories containing the npy files
fn get_test_data_paths(fn_test_args_dir: PathBuf) -> (PathBuf, Vec<PathBuf>) {
    let mut test_dirs: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(fn_test_args_dir).unwrap() {
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
    for test_args_dir in PathBuf::from("tests/data/stft2").read_dir().unwrap() {
        let test_args_dir = test_args_dir.unwrap().path();
        if test_args_dir.is_dir() {
            let (args_file, test_dirs) = get_test_data_paths(test_args_dir);
            let mut state = config_from_args_file(args_file);
            for test_dir in test_dirs {
                let in_: ndarray::Array2<f32> = read_npy(test_dir.join("in.npy")).unwrap();
                let expected_out: ndarray::Array3<Complex32> =
                    read_npy(test_dir.join("out.npy")).unwrap();
                let out_ = stft2(in_.view(), &mut state);
                assert_eq!(out_, expected_out);
            }
        }
        let (args_file, test_dirs) = get_test_data_paths(PathBuf::from("tests/data/stft2"));

        //let expected_out: ndarray::Array3<Complex32> = read_npy("tests/data/stft2/out.npy").unwrap();
    }
}
