from keras.models import load_model
import collections
import numpy as np

def read_words(filename):
    with open(filename, "r") as f:
        text = f.read()
        text = text.replace("\n", " ")
        text = text.replace("<unk>", " ")
        return [w for w in text.split() if len(w) > 4]


def build_vocab(filename):
    data = read_words(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id


def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

word_to_id = build_vocab("ptb.train.txt")
id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))
test_data = file_to_word_ids("ptb.test.txt", word_to_id)
model = load_model("test.h5")
#print(model.summary())

k = 150
inp = np.array(test_data[k:k+20])
print([id_to_word[v] for v in inp])
for j in range(0, 10):
    one_hot_result = model.predict(np.reshape(inp,(1,20)))
    result = np.argmax(one_hot_result)
    word = id_to_word[result]
    print(word)

    inp = np.append(inp[1:], result)