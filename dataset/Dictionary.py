import pickle


class Dictionary(object):
    def __init__(self, file_path=None):
        """Vocabulary comprising a word2idx dictionary and idx2word
        list. By default, the vocabulary contains a special token [MASK]
        whose index is 0.

        Args:
            file_path (string, optional): Path to the file to load.
            Defaults to None.
        """
        if file_path is None:
            self.word2idx = {
                "[MASK]": 0,
                "[UNKNOWN]": 1,
                "[START]": 2,
                "[END]": 3,
                "[TRUNCATE]": 4,
            }
            self.idx2word = [
                "[MASK]",
                "[UNKNOWN]",
                "[START]",
                "[END]",
                "[TRUNCATE]",
            ]
        else:
            with open(file_path, "rb") as f:
                self.word2idx, self.idx2word = pickle.load(f)

    def get_idx(self, word):
        """Return the index of the word given in argument or [UNKNOWN]
        (1) if not in the dictionary.

        Args:
            value (str): Word.

        Returns:
            int: Index.
        """
        return self.word2idx[word] if word in self.word2idx else 1

    def add_word(self, word):
        """Add a word to the dictionary.

        Args:
            word (str): Word to add.

        Returns:
            int: Index of the word.
        """
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        """Number of word in the vocabulary.

        Returns:
            int: Number of word in the vocabulary.
        """
        return len(self.idx2word)

    def save(self, file_path):
        """Save the dictionnary using Pickle.

        Args:
            file_path (string): Path to the file.
        """
        with open(file_path, "wb") as f:
            pickle.dump((self.word2idx, self.idx2word), f)
