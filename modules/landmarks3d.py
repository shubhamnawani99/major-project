import face_alignment
import numpy as np

# face_alignment library for 3d landmark detection
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)


# get the facial landmarks
def get_face_keypoints(frame):
    return fa.get_landmarks(frame)


# calculate the attention
def calculate_attention(vector):
    return vector[2] / np.linalg.norm(vector)


# calculate the orientation vector
def calculate_vector(key_points):
    # calculate the mean position of all the landmarks (x,y,z) co-ordinates
    mean_pos = np.mean(key_points, 0)
    nose_pos = (key_points[31] + key_points[34]) / 2
    orientation_diff = nose_pos - mean_pos
    return orientation_diff
