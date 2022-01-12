import numpy as np

SOURCE_FILE = ""
TARGET_FILE = ""
VOCAB_SIZE = 173267
WORD_DIMENSION = 256

lines = [line for line in open(SOURCE_FILE,"rt")]
lines = lines[1:]

data = np.zeros([VOCAB_SIZE, 256], dtype=np.float32)
for l in lines:
    word_id, *word_vec = l.strip().split(",")
    data[int(word_id)] = [float(x) for x in word_vec]

np.savez(TARGET_FILE, embedding=data)
