def sequence_equal(seq, other_seq):
    return len(seq) == len(other_seq) and sorted(seq) == sorted(other_seq)
