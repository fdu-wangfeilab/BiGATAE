from utils.PASTE import pairwise_align

def get_pi(slice1, slice2):
    return pairwise_align(slice2, slice1)