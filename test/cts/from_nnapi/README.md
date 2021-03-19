# Converted CTS from NNAPI CTS

## Test Generator Tool
This [test_generator](./test_generator) tool is for converting existed native 
[NNAPI CTS](https://android.googlesource.com/platform/frameworks/ml/+/refs/tags/android-cts-10.0_r5/nn/runtime/test/specs/) of V1_0, V1_1 and V1_2 versions to these tests for WebNN API.

### NNAPI Operations being map to WebNN API Operations Tables
* Part I 

The `test_generator` tool has supported to convert those tests of following
NNAPI operations to the tests for such WebNN API operations of first wave.

| NNAPI                         | WebNN API (first wave ops)    |
|:------------------------------|:------------------------------|
| RELU1                         | clamp                         |
| RELU6                         | clamp                         |
| CONCATENATION                 | concat                        |
| ADD                           | add [+ relu/clamp]            |
| SUB                           | sub [+ relu/clamp]            |
| MUL                           | mul [+ relu/clamp]            |
| DIV                           | div [+ relu/clamp]            |
| MAXIMUM                       | max                           |
| MINIMUM                       | min                           |
| EXP                           | exp                           |
| LOGISTIC                      | sigmoid                       |
| SQRT                          | pow                           |
| TANH                          | tanh                          |
| FULLY_CONNECTED               | matmul [+ add+ relu/clamp]    |
| AVERAGE_POOL_2D               | averagePool2d [+ relu/clamp]  |
| MAX_POOL_2D                   | maxPool2d  [+ relu/clamp]     |
| CONV_2D                       | conv2d [+ add + relu/clamp]   |
| DEPTHWISE_CONV_2D             | conv2d [+ add + relu/clamp]   |
| RELU                          | relu                          |
| RESHAPE                       | reshape                       |
| SLICE                         | slice                         |
| SOFTMAX                       | softmax                       |
| SPLIT                         | split                         |
| SQUEEZE                       | squeeze                       |
| TRANSPOSE                     | transpose                     |

* Part II

And there're these following NNAPI operations which could be map to others
WebNN API operations of next waves.

| NNAPI                         | WebNN API (next waves ops)     |
|:------------------------------|:------------------------------|
| ABS                           | abs                           |
| FLOOR                         | floor                         |
| LOG                           | log                           |
| NEG                           | neg                           |
| SIN                           | sin                           |
| L2_POOL_2D                    | l2Pool2d  [+ relu/clamp]      |

* Note: 

1. Current WebNN Polyfill API supports Float32 and Int32 two types, so these
NNAPI CTS using Float32 and Int32 types were able to be converted, while those
NNAPI CTS using Uint8 and Int8 types would be converted until WebNN Polyfill API
supports Uint8 and Int8 types.

2. Native NNAPI supports Float16, while there's lack of Float16 in JavaScript
environments, so such NNAPI CTS with Float16 wouldn't be convertted.

### Usage
* Prerequisites
  * Python3
  * Numpy
  * Download and unzip NNAPI CTS Specs tarball files locally by below commands

    ```shell
    cd test_generator
    ./ready_nnapi_cts_specs.sh
    ```
* Generate Tests

  ```shell
  npm start
  ```

 Generated tests would be in following three folders  
[./tests/V1_0](./test/V1_0)  
[./tests/V1_1](./test/V1_1)  
[./tests/V1_2](./test/V1_2)  
and these tests could also be in all-in-one 
[./tests/cts.js](./tests/cts.js) file.


### Accuracy for Generated Tests
The converted tests follow these [reference accuracy](https://android.googlesource.com/platform/frameworks/ml/+/refs/tags/android-cts-10.0_r5/nn/runtime/test/TestGenerated.cpp#117):
```cpp
  float fpAtol = 1e-5f;
  float fpRtol = 5.0f * 1.1920928955078125e-7f;
```
And for relaxed tests
```cpp
  // If in relaxed mode, set the absolute tolerance to be 5ULP of FP16.
  fpAtol = 5.0f * 0.0009765625f;
  // Set the relative tolerance to be 5ULP of the corresponding FP precision.
  fpRtol = 5.0f * 0.0009765625f;
```