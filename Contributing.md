# Contributing

This project is still in the early days so contributions are both welcome and probably very needed. It's meant to be used for preprocessing audio for feeding into ML models that do stuff like detect wakewords or do speech-to-text transcription. It started as a copy of speechpy, and is now moving to being fully compatible with librosa. We're still trying to get to the MVP stage, and I recommend looking at our open issues. When trying to decide what to work on or how you might make the most impactful contribution I think it's worth reflecting on our goals for this library:

we want speechsauce(name still pending) to be:

- compatible with librosa's mel spectrogram and mfcc function
- compatible to the point that models can be trained in python, and then ran in rust without the preprocessing step being the thing that skews the results.
- verifiably correct, having test cases that ensure the output is correct. it doesn't matter if the code runs if the math is wrong
- usable in rust. if you are porting a model to rust that was built in python, you can make it so that speech preprocessing is the same on both ends.

given those goals, and the state of the project, there are a lot of low hanging fruits:

- writing benchmarks for the existing functions
- if you understand the math, or have done DSP, writing test (or just telling us what to test for so we can write the test)
- improving the documentation for the existing code
- creating issues as you encounter them.

## Suggestions related to issue tags

### librosa_compat

if an issue is tagged with `librosa_compat` open the permalinked source code above the function in your browser, our implementation doesn't need to be line for line the same, but it does need behave the exact same way. If that sounds incredibly stupid, please [see this issue](https://github.com/librosa/librosa/issues/1093) there is no reference standard, and when I compared the results of the mfcc functions from speechpy, sonopy, and librosa to each other, none of them were the same. given that, the best way to make sure that a model trained in python will be just as accurate when ran in rust, is to make sure the output of preprocessing is damn near identical. 

### python
These are issues which either require some python coding, either to make sure that our code works/runs in python, or to generate outputs of python functions for API compatiblity testing. 

### hard

this will probably be broken down into different types at some point. Some task are hard because they require specific domain knowledge (that I don't really have), some are hard because they require a good bit of knowledge of generics in rust but are mainly altering the content of functions rather than changing them. but 

## Currently Out of scope but welcome

with these, I'm open to PRs can make suggestions but these would be far enough out of scope for the projects intended usecase that I won't be able to provide much assistant (other than minor code changes)

### adding support for other array implementations in rust (`arrayfire`, `nalgebra`, or `rapl`).
right now this is just focused on using ndarray, and getting the result to be runnable in tract, but the rust ML ecosystem is growing, and this functionality *may* be generalizable enough to use with other matrix or tensor data types. 

the only restriction is that it cannot be done in a way that makes it sufficiently more difficult to support ndarray/tract


### adding functions which may be useful for the purpose of speech processing.

I'm not going to lie, I have no idea what I'm doing, and I'm only interested in implementing specific sets of functions as it related to preprocessing audio for feeding into ML models for wakewords and STT engines, but this library might eveventually have other uses where speech processing needs to happen rust side. If you want to add functions that are useful  for whatever falls into the umbrella of "speech preprocessing" and don't break the intended use case, I have no problem adding them. Just please sufficiently documenent them, because otherwise they may as well be a magic box of voodoo if that task falls to me. 

Don't be a stranger, feel free to contact us if you have any questions or need help figuring out where to start.
