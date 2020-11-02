import functools
import typing

import tensorflow as tf
import numpy as np


class TensorHandle:
    def __init__(self, tensor):
        assert isinstance(tensor, tf.Tensor), tensor
        self.tensor = tensor

    def __repr__(self):
        return f"{self.tensor}({id(self)})"


class ValueHandle:
    def __init__(self, tf_value):
        # For ValueHandle, uniqueness of id(self) is important.
        # There could be many distinct ValueHandle's, referring to the same tf_value object.
        assert not isinstance(tf_value, ValueHandle), tf_value
        self.tf_value = tf_value
        self.tensors = [
            TensorHandle(t) for t in tf.nest.flatten(tf_value, expand_composites=True)
        ]

    def __repr__(self):
        return f"{self.tf_value}({id(self)})"


class CallRecord(typing.NamedTuple):
    cf: tf.python.eager.function.ConcreteFunction
    inputs: typing.List[ValueHandle]
    outputs: typing.List[ValueHandle]


class SuperOp:
    def __init__(self):
        """
        This class cannot deal with capturing results of operations from itself.
        """
        self.calls = []
        self._value_handles = {}  # ValueHandle cache

    def get_input_value_handle(self, x) -> ValueHandle:
        # NB: outputs cannot and should not be cached
        if isinstance(x, ValueHandle):
            return x

        x_id = id(x)
        try:
            return self._value_handles[x_id]
        except KeyError:
            value_handle = self._value_handles[x_id] = ValueHandle(x)
            return value_handle

    def get_tf_value(self, value_id: ValueHandle):
        assert isinstance(value_id, ValueHandle), value_id
        return value_id.tf_value

    def __call__(self, func):
        func = tf.function(func)

        @functools.wraps(func)
        def f(*args, **kwargs):
            flat_input_value_handles = [
                self.get_input_value_handle(v) for v in tf.nest.flatten((args, kwargs))
            ]
            flat_input_values = [
                self.get_tf_value(vh) for vh in flat_input_value_handles
            ]
            args, kwargs = tf.nest.pack_sequence_as((args, kwargs), flat_input_values)

            cf = func.get_concrete_function(*args, **kwargs)
            output_values_handles = tf.nest.map_structure(
                ValueHandle, cf.structured_outputs
            )

            self.calls.append(
                CallRecord(
                    cf,
                    flat_input_value_handles,
                    tf.nest.flatten(output_values_handles),
                )
            )
            return output_values_handles

        return f
