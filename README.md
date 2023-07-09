# SpeechSauce

A rust lib providing a mel spectrogram and mfcc function for use with ndarray and tract. The goal is to enable you to train your (speech centric) models in python, and run them in rust with a certain degree of trust that preprocessing is the same. While this started as a port of [speechpy](https://github.com/astorfi/speechpy), it's aiming at being completely compatible with the same functions in librosa. Why? they are faster, more well maintained, and sort of the defacto standard. 

we're still getting there. 

## Usage

Supported features:

- Mel Frequency Cepstral Coefficients (based off speechpy's version currently, will change)
- Mel Spectrogram (based off librosa's)
- Filterbank Energies (speechpy's)
- Log Filterbank Energies (speechpy's)
- Spectral Subband Centroids (speechpy's)

all functions currenlty are hardcoded to use `f32`, we have plans for making functions generic over floating point types, but It's a bit down the list of things that need to be done. 

## Building it

the rust component (currently the `speechsauce` directory) can be built,tested and ran with cargo

the python component (currently the `py-speechsauce` directory) requires [maturin](https://github.com/PyO3/maturin), if you want to test it out in a python session, here's recommended way of doing so:

```sh
cd py-speechsauce
python -m venv .venv
source .venv/bin/activate # if using bash/zsh
#source .venv/bin/activate.fish #if using fish
pip install maturin
maturin develop
```
also we have a "benchmark" python program that's more useful for comparing the different mfcc/ mel spectrogram implementations. you can find it [here](https://github.com/skewballfox/mfcc-benchmark)

```sh
git clone https://github.com/skewballfox/mfcc-benchmark
cd mfcc-benchmark
poetry install
poetry shell
python -m mfcc-bench
```

## What's left to do?
because there is no standard definition of what is correct for an mfcc or mel spectrogram implementation, the best you can probably manage is to match the behavior of a heavily used implementation and have sane defaults when that isn't possible. librosa, and their tensor based clone in pytorch, seem to be the most popular library for audio preprocessing
- [ ] use npy files to get input and output for librosa's mel spec (and the stack of function calls under it), and mfcc
- [ ] write a series of test for each function to confirm the behavior matches rust side

at that point if everything passes, it's ready to use rust side (for f32). for it's use in python, there is an extra step of reworking the python api to be more ergonomic.

## Acknowledgements

This work originally started as a rewrite of speechpy in rust, which involved quite a bit of reverse engineering of the [speechpy codebase](https://github.com/astorfi/speechpy). At the time of writing, the behavior of the library and functionality offered *should be* one-to-one. This may, and likely will, change as this project matures. But this likely wouldn't exists without their prior work.

the 2d version of the stft came from another developer rikorose in his project [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet/blob/ebfec786ce37951365c38b8e793a7b99e66fd7b3/libDF/src/transforms.rs#L134), which I copied with his permission. That may wind up being packaged into a separate crate, eventually.
