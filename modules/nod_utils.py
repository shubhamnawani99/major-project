import numpy as np


# module to compute nodding of a participant
def is_nodding(vector_buffer):
    vector_buffer = np.asarray(vector_buffer)
    vector_buffer_in_y_direction = vector_buffer[:, 1]
    variance_y = np.var(vector_buffer_in_y_direction)
    return variance_y > 15
