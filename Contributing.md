# Contributing

This project is still in the early days so contributions are both welcome and probably very needed. We're still trying to get to the MVP stage, and I recommend looking at our open issues. When trying to decide what to work on or how you might make the most impactful contribution I think it's worth reflecting on our goals for this library:

we want speechsauce(name still pending) to be:

- faster than speechpy, or the other alternatives in python
- familiar to users of speechpy
- verifiably correct, having test cases that ensure the output is correct. it doesn't matter if the code runs if the math is wrong
- usable in rust. if you are porting a model to rust that was built in python, you can make it so that speech preprocessing is the same on both ends.

given those goals, and the state of the project, there are a lot of low hanging fruits:

- writing benchmarks for the existing functions
- if you understand the math, or have done DSP, writing test (or just telling us what to test for so we can write the test)
- improving the documentation for the existing code
- creating issues as you encounter them.

if you don't mind reaching further up the tree:

- adding support for other array implementations in rust (`arrayfire` or `nalgebra`).
- making the existing functions faster
- adding functions which may be useful for the purpose of speech processing.

Don't be a stranger, feel free to contact us if you have any questions or need help figuring out where to start.