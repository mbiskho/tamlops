import tensorflow as tf
import os

# Check if a GPU is available and set it as the default device
if tf.test.is_gpu_available():
    print("GPU available. Using GPU.")
    # devices = tf.config.experimental.list_physical_devices("GPU")
    # tf.config.experimental.set_memory_growth(devices[0], True)
else:
    print("No GPU available. Using CPU.")

# # Create two random matrices
# matrix_a = tf.random.normal([1000, 1000])
# matrix_b = tf.random.normal([1000, 1000])

# # Perform matrix multiplication on the GPU (or CPU if no GPU is available)
# result = tf.matmul(matrix_a, matrix_b)

# # Print the result
# print("Matrix multiplication result:")
# print(result)



print(os.environ['CUDA_VISIBLE_DEVICE'])
