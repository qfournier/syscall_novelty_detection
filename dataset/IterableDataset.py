from itertools import islice
from torch.utils import data


class IterableDataset(data.IterableDataset):
    def __init__(self, file_path, max_sample, max_token):
        """An iterable dataset that reads data from a file without
        storing it in memory. The first max_sample samples are read, and
        each sample is truncated to max_token if necessary.

        Args:
            file_path (string): Path to the file to load.
            max_sample (int): Number of samples to read before cycling.
            max_token (int): Maximum sequence length.
        """
        self.file_path = file_path
        self.max_sample = max_sample
        self.max_token = max_token

        # For DistributedDataParallel (by default, only a single GPU)
        self.rank = 0
        self.world_size = 1

    def parse_file(self, file_path):
        """Parse the dataset file and yield a sample (i, call, entry,
        time, prod, pid, tid, ret, req_duration).

        Args:
            file_path (string): Path to the file to load.

        Yields:
            tuple: (i, call, entry, time, prod, pid, tid, ret,
            req_duration).
        """
        worker_info = data.get_worker_info()
        start = (
            self.rank
            if worker_info is None
            else self.rank * worker_info.num_workers + worker_info.id
        )
        step = (
            self.world_size
            if worker_info is None
            else self.world_size * worker_info.num_workers
        )

        with open(file_path, "r") as f:
            for sample in islice(f, start, self.max_sample, step):
                # Get a list of argument sequences
                sample = sample.split(";")
                # Remove the request duration from the line
                req_duration = float(sample.pop())
                # Convert the sequences to integer
                sample = [list(map(int, x.split(","))) for x in sample]
                # If the request is longer than max_token
                if (
                    self.max_token is not None
                    and len(sample[0]) > self.max_token
                ):
                    sample = [x[: self.max_token] for x in sample]
                    for i, x in enumerate(sample):
                        # Append [TRUNCATE] to the syscall and process names
                        if i == 0 or i == 3:
                            x.append(4)
                        # Append [MASK] to the other arguments
                        else:
                            x.append(0)
                # Yield the request
                yield (
                    *[x[:-1] for x in sample],
                    sample[0][1:],
                    req_duration,
                )

    def __iter__(self):
        """Iterator that yields samples.

        Returns:
            iterator: Samples.
        """
        return self.parse_file(self.file_path)
