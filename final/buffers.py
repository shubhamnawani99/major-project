from collections import defaultdict, deque
from threading import RLock
from functools import partial

import Levenshtein

# Buffer class to store all the attribute of users
class Buffer:
    def __init__(self, MAX_FRAMES=20):
        self.MAX_FRAMES = MAX_FRAMES
        self.lip_distances = defaultdict(partial(deque, maxlen=MAX_FRAMES))
        self.orientation_vectors = defaultdict(partial(deque, maxlen=MAX_FRAMES))
        self.orientation_scores = defaultdict(partial(deque, maxlen=MAX_FRAMES))
        self.lip_variances = defaultdict(partial(deque, maxlen=MAX_FRAMES))
        self.yawns = defaultdict(partial(deque, maxlen=MAX_FRAMES))
        self.nods = defaultdict(partial(deque, maxlen=MAX_FRAMES))
        self.presences = defaultdict(partial(deque, maxlen=MAX_FRAMES))
        self.attention_scores = defaultdict(partial(deque, maxlen=MAX_FRAMES))
        self.attention_classes = defaultdict(partial(deque, maxlen=MAX_FRAMES))
        self.num_people = -1
        self.all_people = set()
        self.this_frame_people = set()
        self.lock = RLock()

    # add the various attributes
    def add_lip_dist(self, name, dist):
        self.lip_distances[name].append(dist)

    def add_orientation_vector(self, name, vector):
        self.orientation_vectors[name].append(vector)

    # add the orientation score corresponding to the user
    def add_orientation_score(self, name, score):
        self.orientation_scores[name].append(score)

    def add_variance(self, name, variance):
        self.lip_variances[name].append(variance)

    def add_yawn(self, name, yawn):
        self.yawns[name].append(yawn)

    def add_nod(self, name, nod):
        self.nods[name].append(nod)

    # set the total number of people
    def set_num_people(self, num_people):
        self.num_people = num_people

    # add the person to the sets
    # 1. this frame people
    # 2. all people
    def announce(self, name):
        self.this_frame_people.add(name)
        self.all_people.add(name)

    # reset the set of current people frame
    def reset_people(self):
        self.this_frame_people = set()

    # add the attention score corresponding to the user
    def add_attention_score(self, name, score):
        self.attention_scores[name].append(score)

    # add the attention class corresponding to the user
    def add_attention_class(self, name, class_name):
        self.attention_classes[name].append(class_name)

    # set the presence of a person in the frame
    # if person not in the current frame,
    # assign false in the presence dict
    def set_presences(self):
        for name in self.all_people:
            if name not in self.this_frame_people:
                self.presences[name].append(False)
            else:
                self.presences[name].append(True)

    # match the name from the list of all people
    def match_name(self, name):
        parts = name.split()
        for n in self.all_people:
            n_split = n.split()
            if Levenshtein.distance(n, name) < 5 or Levenshtein.distance(parts[0], n_split[0]) < 2:
                return n
        return name
