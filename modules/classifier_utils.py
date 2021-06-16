from collections import defaultdict
from enum import Enum

import numpy as np


# the different attention class values
class AttentionClass(Enum):
    UNKNOWN = 0
    DROWSY = 1
    INATTENTIVE = 2
    ATTENTIVE = 3
    INTERACTIVE = 4


# MAX QUEUE SIZE IS 7
# Threshold values
NOD_THRESHOLD = 3
YAWN_THRESHOLD = 2
EYE_THRESHOLD = 3


# module to classify the person
def classify(buffer):
    classes = defaultdict(lambda: AttentionClass.UNKNOWN)
    scores = defaultdict(lambda: 0)

    # the people in the current frame
    print(buffer.this_frame_people)

    # classification for all names in the current buffer frame
    for name in buffer.this_frame_people:

        if sum(buffer.presences[name]) < len(buffer.presences[name]) / 2:
            classes[name] = AttentionClass.UNKNOWN
            scores[name] = -1
            continue

        # set the mean variance and orientation
        mean_var = np.mean(buffer.lip_variances[name]) if len(buffer.lip_variances[name]) != 0 else 0
        mean_orientation_score = np.mean(buffer.orientation_scores[name]) if len(
            buffer.orientation_scores[name]) != 0 else 0.5

        # running sum for queue of size 7
        sum_eyes_open = sum(buffer.eyes[name])
        sum_yawns = sum(buffer.yawns[name])
        sum_nods = sum(buffer.nods[name])

        # mean orientation is greater than 0.7
        if mean_orientation_score >= 0.7:

            if sum_eyes_open <= EYE_THRESHOLD:
                classes[name] = AttentionClass.INATTENTIVE
                print("{} : INATTENTIVE".format(name))

            else:
                # a person is INTERACTIVE if they nod more than the given threshold
                # or have active lip movement
                # mean_var high range 200, since any value more than
                # this would indicate the complete face is not in frame
                if 200 > mean_var > 90 or sum_nods >= NOD_THRESHOLD:
                    classes[name] = AttentionClass.INTERACTIVE
                    print("{} : INTERACTIVE".format(name))

                # a person is ATTENTIVE if they have a high mean orientation score
                else:
                    classes[name] = AttentionClass.ATTENTIVE
                    print("{} : ATTENTIVE".format(name))

        else:
            # a person is DROWSY if they have yawns more than the threshold
            # or if there eye is closed for a certain duration
            if sum_yawns > YAWN_THRESHOLD or sum_eyes_open <= EYE_THRESHOLD or mean_var < 20:
                classes[name] = AttentionClass.DROWSY
                print("{} : DROWSY".format(name))

            else:
                classes[name] = AttentionClass.INATTENTIVE
                print("{} : INATTENTIVE".format(name))

        var_bin = (mean_var > 100)
        nod_bin = (sum(buffer.nods[name]) >= NOD_THRESHOLD)
        yawn_bin = (sum(buffer.yawns[name]) >= YAWN_THRESHOLD)

        scores[name] = (var_bin * 0.5 + mean_orientation_score * 1 + nod_bin * 0.5 - yawn_bin * 2 + 2) * 25

        # add the attention score and class to the person buffer
        buffer.add_attention_score(name, scores[name])
        buffer.add_attention_class(name, classes[name])

        # cleanup for people left out of the current frame
        for name in buffer.all_people:
            if name not in buffer.this_frame_people:
                if sum(buffer.presences[name]) < len(buffer.presences[name]) / 2:
                    classes[name] = AttentionClass.UNKNOWN
                    scores[name] = -1
                    continue
                else:
                    classes[name] = buffer.attention_classes[name][-1]
                    scores[name] = buffer.attention_scores[name][-1]

    # return thr classes and the scores
    return classes, scores


def print_info(buffer, name):
    print('\nEye buffer length: ', len(buffer.eyes[name]), 'Eye buffer sum:', sum(buffer.eyes[name]))
    print('Lips Buffer Length: ', len(buffer.yawns[name]), 'yawns buffer sum:', sum(buffer.yawns[name]))
