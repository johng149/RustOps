`cross test --target aarch64-unknown-linux-gnu`

Other targets we want are:
`aarch64-linux-android`
`arm-linux-androideabi`
`armv7-linux-androideabi`

Time to beat:

Optimize completed in: 430.294153ms
Generated sensory input for predict with shape: [32, 400, 256]
Running predict...
Predict completed in: 172.951446ms


Batch size for the input tensor [6]: 32
Number of chunks (fields/nodes in outer layer) [8]: 400
Dimension of each chunk (feature dimension) [7]: 256
Use release mode (optimized build)? [y/N]: y
Run with samply profiler? [y/N]: y

Did some tests, the PyTorch version (also on CPU) with the same network size and input size
does the optimization step in about 100ms.