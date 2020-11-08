This is basically `tf.scan`, but for any computation graph.

```python
import tensorflow as tf
import numpy as np

from super_op import SuperOp

N = 3 # 100

x = tf.ones((N,))
a = tf.ones((N,)) * 81

super_op = SuperOp()

@super_op
def f(x):
    return (x + a / x) * 0.5

y = x
for i in range(10):
    y = f(y)

super_op.compute(results=y)
```

In the code above, `@super_op` decorator creates a function, which returns lazy values. Using these registered functions, `SuperOp` remembers all operations.

`SuperOp.compute()` method takes desired result, and creates a single `tf.while_loop()`. Each iteration picks inputs and a function (using `tf.switch_case()`), and stores the outputs to `tf.TensorArray`.

Inputs, outputs and results can be nested structures of tuples, list, dicts, SparseTensors - whatever tensorflow supports in `tf.nest.*`.

# Known issues
The combination of `tf.switch_case`, `tf.while_loop`, gradients and XLA together make tensorflow 2.3.1 fail. tf 2.5.0 nightly didn't have this issue.
