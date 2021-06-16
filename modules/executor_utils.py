from extract_names import extract_name_from_frame
from landmarks3d import get_face_keypoints, calculate_attention, calculate_vector
from lip_variance_utils import get_lip_dist, get_lip_variance
from nod_utils import is_nodding
from yawn import is_yawning
from eye import is_eye_open


# process the images and upload in the buffer
def process_and_upload(frame, buffer):

    # get the person name from the frame
    name = buffer.match_name(extract_name_from_frame(frame))

    # add the person to the buffer, to both the current set and total set
    buffer.announce(name)

    # get facial landmarks for the frame
    landmarks = get_face_keypoints(frame)[0]

    # if the person is detected and landmarks are formed
    if landmarks is not None:

        # Primary Calculations
        cur_vector = calculate_vector(landmarks)
        cur_orientation = calculate_attention(cur_vector)
        dist = get_lip_dist(landmarks)
        # returns true if mouth is open (yawning)
        is_yawn = is_yawning(landmarks)
        # return true if eye is open
        is_eye = is_eye_open(landmarks)

        # Initialize the buffer to enable computing of running mean
        if len(buffer.eyes[name]) < 7:
            buffer.add_initial_values(name)

        # Add Primary Values to the Buffer
        buffer.add_lip_dist(name, dist)
        buffer.add_orientation_vector(name, cur_vector)
        buffer.add_orientation_score(name, cur_orientation)
        buffer.add_yawn(name, is_yawn)
        buffer.add_eye(name, is_eye)

        # Secondary Calculations
        variance = get_lip_variance(buffer.lip_distances[name])
        nod = is_nodding(buffer.orientation_vectors[name])

        # Add Secondary Values to the Buffer
        buffer.add_variance(name, variance)
        buffer.add_nod(name, nod)