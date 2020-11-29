
import tensorflow as tf


class MyClass:
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32), tf.TensorSpec(shape=None, dtype=tf.float32)])
    def my_method(self, my_input, my_other_input):
        print(my_other_input)
        return my_input + tf.constant(10.0)


if __name__ == "__main__":
    my_class = MyClass()
    print(my_class.my_method(10.0, 20.0))
