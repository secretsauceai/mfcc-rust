# MFCC-rust

A rust port of [speechpy](https://github.com/astorfi/speechpy) built using [ndarray](https://github.com/rust-ndarray/ndarray) for matrix operations. All documentation, including this readme, is currently a WIP.

## Usage

Supported features:

- Mel Frequency Cepstral Coefficients
- Filterbank Energies
- Log Filterbank Energies
- Spectral Subband Centroids

## Acknowledgements

This work originally started as a rewrite of speechpy in rust, which involved quite a bit of reverse engineering of the [speechpy codebase](https://github.com/astorfi/speechpy). At the time of writing, the behavior of the library and functionality offered *should be* one-to-one. This may, and likely will, change as this project matures. But this likely wouldn't exists without their prior work.
