import collections
import functools
import itertools
import typing

from more_itertools import zip_equal

import tensorflow as tf
from tensorflow.python.eager.function import ConcreteFunction
import numpy as np


class ValueHandle:
    def __init__(self, tf_value):
        # For ValueHandle, uniqueness of id(self) is important.
        # There could be many distinct ValueHandle's, referring to the same tf_value object.
        assert not isinstance(tf_value, ValueHandle), tf_value
        self.tf_value = tf_value

    def __repr__(self):
        return f"{self.tf_value}({id(self)})"


class TensorId(typing.NamedTuple):
    value_handle: ValueHandle
    index: int


class CallRecord(typing.NamedTuple):
    cf: ConcreteFunction
    inputs: typing.List[ValueHandle]
    outputs: typing.List[ValueHandle]


class Indexer:
    def __init__(self):
        self.elements = []
        self.counts = {}
        self.index = {}

    def add(self, value):
        count = self.counts.get(value, 0)
        if not count:
            self.elements.append(value)
            index = self.index[value] = len(self.index)
            assert len(self.index) == len(self.elements)
        else:
            index = self.index[value]
        self.counts[value] = count + 1
        return index

    def get(self, index):
        return self.elements[index]

    def __iter__(self):
        return iter(self.elements)

    def __len__(self):
        return len(self.elements)


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
        func = tf.function(autograph=False, experimental_compile=True)(func)

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

    def compute(self, results):
        # NB must be a constant method (not change `self`)
        builder = SuperOpBuilder()
        result_info_flat = [
            (value, builder.allocate_input(value) if isinstance(value, ValueHandle) else None)
            for value in tf.nest.flatten(results)
        ]

        for call in reversed(self.calls):
            builder.add_call(call)

        tas = builder.create_tensor_arrays()

        encoded_script = []
        for record in reversed(builder.script):
            encoded_script.extend(
                [
                    record.extended_cf_index,
                    *record.input_value_locations,
                    *record.output_value_locations,
                ]
            )
        encoded_script_tensor = tf.constant(encoded_script, tf.int32)

        def body(ip, tas):
            branch_fns = []

            for extended_cf_index, extended_cf in enumerate(builder.extended_cfs):

                def fn(extended_cf_index=extended_cf_index, extended_cf=extended_cf):
                    local_tas = dict(tas)

                    shift_counter = itertools.count(1)
                    cf, output_counts = extended_cf
                    input_specs = tf.nest.flatten(
                        cf.structured_input_signature, expand_composites=True
                    )
                    input_tensors = [
                        local_tas[get_ta_key(spec)].read(
                            encoded_script_tensor[next(shift_counter) + ip]
                        )
                        for spec in input_specs
                    ]
                    args, kwargs = tf.nest.pack_sequence_as(
                        cf.structured_input_signature,
                        input_tensors,
                        expand_composites=True,
                    )
                    outputs = cf(*args, **kwargs)
                    output_values = tf.nest.flatten(outputs)
                    for output_value, count in zip_equal(output_values, output_counts):
                        for tensor in tf.nest.flatten(
                            output_value, expand_composites=True
                        ):
                            ta_key = get_ta_key(tf.TensorSpec.from_tensor(tensor))
                            ta = local_tas[ta_key]
                            for _ in range(count):
                                ta = ta.write(
                                    encoded_script_tensor[next(shift_counter) + ip],
                                    tensor,
                                )
                            local_tas[ta_key] = ta

                    return next(shift_counter) + ip, local_tas

                branch_fns.append(fn)

            return switch_case(encoded_script_tensor[ip], branch_fns)

        index, tas = tf.while_loop(
            cond=lambda ip, tas: ip < len(encoded_script),
            body=body,
            loop_vars=(tf.constant(0, tf.int32), tas),
            maximum_iterations=len(builder.script),
            parallel_iterations=1,
        )

        result_values_flat = [
            vh if locations is None else tf.nest.pack_sequence_as(
                vh.tf_value,
                [
                    tas[get_ta_key(tf.TensorSpec.from_tensor(tensor))].read(loc)
                    for tensor, loc in zip_equal(
                        tf.nest.flatten(vh.tf_value, expand_composites=True), locations
                    )
                ],
                expand_composites=True,
            )
            for vh, locations in result_info_flat
        ]
        return tf.nest.pack_sequence_as(results, result_values_flat)


def switch_case(index, branch_fns):
    return tf.switch_case(index, branch_fns)

    # if len(branch_fns) == 1:
    #     return branch_fns[0]()

    # mid = len(branch_fns) // 2

    # return tf.cond(
    #     index < mid,
    #     lambda: switch_case(index, branch_fns[:mid]),
    #     lambda: switch_case(index - mid, branch_fns[mid:]),
    # )



def get_ta_key(tensor_spec):
    return tensor_spec.dtype


def make_ta(ta_key, **kwargs):
    return tf.TensorArray(dtype=ta_key, **kwargs)


class ScriptRecord(typing.NamedTuple):
    # the concrete operator
    extended_cf_index: int
    # tensor_pos -> (index_in_ta: int) - flattened over input values
    input_value_locations: typing.List[int]
    # tensor_pos -> index_in_ta: int - flattened over masked values and over their duplicates
    output_value_locations: typing.List[int]


class SuperOpBuilder:
    def __init__(self):
        # TensorId = Tuple[ValueHandle, int]
        # For now, only len(list) is used
        self.ta_key_to_tensor_ids = collections.defaultdict(list)  # content of each TA

        # value_handle -> tensor_pos -> (indices_in_ta: List[int])
        self.vh_locations: typing.Dict[
            ValueHandle, typing.List[typing.List[int]]
        ] = {}  # constitution of each value
        self.all_vh_locations = {}  # for DEBUG only

        self.extended_cfs = Indexer()  # [(ConcreteFunction, output_value_counts)]
        self.script = []  # [ScriptRecord]

    def allocate_input(self, value_handle: ValueHandle) -> typing.List[int]:
        tensors = tf.nest.flatten(value_handle.tf_value, expand_composites=True)
        try:
            value_locations = self.vh_locations[value_handle]
        except KeyError:
            if value_handle in self.all_vh_locations:
                raise RuntimeError(
                    f"{value_handle} was already commited to computations"
                )

            value_locations = self.vh_locations[value_handle] = self.all_vh_locations[
                value_handle
            ] = [[] for _ in tensors]

        result: typing.List[int] = []
        for pos, (tensor, tensor_locations) in enumerate(zip(tensors, value_locations)):
            ta_key = get_ta_key(tf.TensorSpec.from_tensor(tensor))
            tensor_ids = self.ta_key_to_tensor_ids[ta_key]
            tensor_location = len(tensor_ids)
            tensor_locations.append(tensor_location)
            result.append(tensor_location)
            tensor_ids.append((value_handle, pos))

        return result

    def add_call(self, call: CallRecord) -> None:
        def output_uses_count(vh):
            try:
                (count,) = {
                    len(indices_in_ta) for indices_in_ta in self.vh_locations[vh]
                }
                return count
            except KeyError:
                return 0
            except ValueError:
                raise RuntimeError(
                    f"{self.vh_locations[vh]} has different lengths, which should be impossible"
                )

        output_value_counts = tuple([output_uses_count(vh) for vh in call.outputs])
        if sum(output_value_counts) == 0:
            # Unused
            return

        extended_cf_index: int = self.extended_cfs.add((call.cf, output_value_counts))

        output_value_locations: typing.List[int] = [
            index_in_ta
            for count, vh in zip(output_value_counts, call.outputs)
            if count
            for indices_in_ta in self.vh_locations.pop(vh)
            for index_in_ta in indices_in_ta
        ]
        input_value_locations: typing.List[int] = [
            loc for vh in call.inputs for loc in self.allocate_input(vh)
        ]
        self.script.append(
            ScriptRecord(
                extended_cf_index, input_value_locations, output_value_locations
            )
        )

    def create_tensor_arrays(self):
        tas = {
            ta_key: make_ta(ta_key, size=len(tensor_ids), infer_shape=False)
            for ta_key, tensor_ids in self.ta_key_to_tensor_ids.items()
        }
        for value_handle, value_locations in self.vh_locations.items():
            tensors = tf.nest.flatten(value_handle.tf_value, expand_composites=True)

            for tensor, tensor_locations in zip_equal(tensors, value_locations):
                ta_key = get_ta_key(tf.TensorSpec.from_tensor(tensor))
                ta = tas[ta_key]
                for loc in tensor_locations:
                    ta = ta.write(loc, tensor)
                tas[ta_key] = ta
        return tas