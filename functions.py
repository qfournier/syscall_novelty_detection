import os
import sys
import math
import torch
import shutil
import sklearn
import warnings
import argparse
import itertools
import numpy as np
from time import time
import seaborn as sns
from datetime import timedelta
from sklearn.manifold import TSNE

from nltk import ngrams
from nltk.lm import NgramCounter

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from ranger import Ranger
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from torch.nn import CrossEntropyLoss
from models import LabelSmoothingCrossEntropy
from models import LSTM
from models import Transformer

from models import MyLongformer

mpl.use("Agg")

###############################################################################
# Arguments
###############################################################################


def get_arguments():
    """Parse the arguments and check their values.

    Returns:
        argparse.ArgumentParser: Arguments.
    """
    # Create parser
    parser = argparse.ArgumentParser()

    # Misc
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--log_folder",
        type=str,
        default=None,
        help="name of the log folder (optional)",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="id of GPUs to use seperated by commas",
    )

    # Data
    parser.add_argument(
        "--data_path",
        type=str,
        help="path to the folder that contains the datasets",
    )
    parser.add_argument(
        "--train_folder",
        type=str,
        help="name of the folder that contains the training set "
        "(format: 'Name to display:folder')",
    )
    parser.add_argument(
        "--valid_id_folder",
        type=str,
        help="name of the folder that contains the in-distribution "
        "validation set (format: 'Name to display:folder')",
    )
    parser.add_argument(
        "--valid_ood_folders",
        type=str,
        help="name of the folders that contains the out-of-distribution "
        "validation sets (format: 'Name to display:folder1,"
        "Name to display:folder2,')",
    )
    parser.add_argument(
        "--test_id_folder",
        type=str,
        help="name of the folder that contains the in-distribution "
        "test set (format: 'Name to display:folder1,"
        "Name to display:folder2,')",
    )
    parser.add_argument(
        "--test_ood_folders",
        type=str,
        help="name of the folders that contains the out-of-distribution "
        "test sets (format: 'Name to display:folder1,"
        "Name to display:folder2,')",
    )
    parser.add_argument(
        "--generate_dataset",
        action="store_true",
        help="generate the dataset in the data folder",
    )
    parser.add_argument(
        "--max_sample",
        type=int,
        default=None,
        help="maximum number of sequences to load",
    )
    parser.add_argument(
        "--max_token", type=int, default=None, help="maximum sequence lengths"
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="transformer",
        choices=["ngram", "lstm", "transformer", "longformer"],
        help="model to use",
    )
    parser.add_argument(
        "--load_model",
        type=int,
        default=None,
        help="load the model from the load_model log folder",
    )
    parser.add_argument(
        "--order", type=int, default=None, help="ngram order (value of n)"
    )
    parser.add_argument(
        "--dim_sys",
        type=int,
        default=None,
        help="embedding dimension of system call name",
    )
    parser.add_argument(
        "--dim_entry",
        type=int,
        default=None,
        help="embedding dimension of the entry or exit",
    )
    parser.add_argument(
        "--dim_ret",
        type=int,
        default=None,
        help="embedding dimension of the return value",
    )
    parser.add_argument(
        "--dim_proc",
        type=int,
        default=None,
        help="embedding dimension of process names",
    )
    parser.add_argument(
        "--dim_pid",
        type=int,
        default=None,
        help="embedding dimension of the process id",
    )
    parser.add_argument(
        "--dim_tid",
        type=int,
        default=None,
        help="embedding dimension of the thread id",
    )
    parser.add_argument(
        "--dim_time",
        type=int,
        default=None,
        help="embedding dimension of the elapsed time between events",
    )
    parser.add_argument(
        "--dim_order",
        type=int,
        default=None,
        help="embedding dimension of the ordering",
    )
    parser.add_argument(
        "--n_head",
        type=int,
        default=None,
        help="number of attention heads (d_k = d/h)",
    )
    parser.add_argument(
        "--n_hidden",
        type=int,
        default=None,
        help="number of hidden units of each encoder MLP",
    )
    parser.add_argument(
        "--n_layer", type=int, default=None, help="number of layers"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="model dropout rate (embedding & encoder)",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=None,
        choices=["relu", "gelu", "swiglu"],
        help="activation function",
    )
    parser.add_argument(
        "--tfixup",
        action="store_true",
        help="uses T-fixup initialization and removes the layer normalization",
    )
    parser.add_argument(
        "--window",
        type=str,
        default=None,
        help="LongSelfAttention window size for each layer in the encoder "
        "(format: '3,3,5')",
    )
    parser.add_argument(
        "--dilatation",
        type=str,
        default=None,
        help="Dilatation for each LongSelfAttention layer in the encoder, "
        "with 1 meaning no dilatation (format: '1, 1, 2')",
    )
    parser.add_argument(
        "--global_att",
        type=str,
        default=None,
        help="Token indexes to apply global attention in all "
        "LongSelfAttention layer in the encoder (format: '0, 1').",
    )

    # Training
    parser.add_argument(
        "--batch", type=int, default=None, help="batch size per GPU"
    )
    parser.add_argument(
        "--n_update", type=int, default=None, help="number of updates"
    )
    parser.add_argument(
        "--eval",
        type=int,
        default=None,
        help="number of updates before evaluating the model "
        "(impact early stopping)",
    )
    parser.add_argument("--lr", type=float, default=None, help="learning rate")
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="increase the learning rate linearly for the first warmup_steps "
        "training steps, and decrease it thereafter proportionally to the "
        "inverse square root of the step number",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=["adam", "ranger"],
        help="Optimizer algorithm used for training the chosen model",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=None,
        help="maximum norm of the gradients",
    )
    parser.add_argument(
        "--ls", type=float, default=None, help="label smoothing [0,1]"
    )
    parser.add_argument(
        "--reduce_lr_patience",
        type=int,
        default=None,
        help="number of iterations before dividing the learning rate by 10 "
        "if the validation loss did not improve in the last (args.patience/2) "
        "evaluations by at least 0.001",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help="number of iterations before early stopping",
    )
    parser.add_argument(
        "--chk", action="store_true", help="use gradient checkpointing"
    )
    parser.add_argument(
        "--amp", action="store_true", help="use automatic mixed-precision"
    )

    # Analysis
    parser.add_argument(
        "--dataset_stat",
        action="store_true",
        help="display data information and plot distributions",
    )
    parser.add_argument(
        "--analysis", action="store_true", help="analyze the model"
    )

    args = parser.parse_args()

    # Assertions
    assert os.path.exists(
        os.path.join(args.data_path, args.train_folder.split(":")[1])
    ), f"{os.path.join(args.data_path, args.train_folder.split(':')[1])} "
    "does not exist"
    assert os.path.exists(
        os.path.join(args.data_path, args.valid_id_folder.split(":")[1])
    ), f"{os.path.join(args.data_path, args.valid_id_folder.split(':')[1])} "
    "does not exist"
    assert os.path.exists(
        os.path.join(args.data_path, args.test_id_folder.split(":")[1])
    ), f"{os.path.join(args.data_path, args.test_id_folder.split(':')[1])} "
    "does not exist"
    for f in args.valid_ood_folders.split(","):
        assert os.path.exists(
            os.path.join(args.data_path, f.split(":")[1])
        ), f"{os.path.join(args.data_path, f.split(':')[1])} does not exist"
    for f in args.test_ood_folders.split(","):
        assert os.path.exists(
            os.path.join(args.data_path, f.split(":")[1])
        ), f"{os.path.join(args.data_path, f.split(':')[1])} does not exist"

    assert (
        args.max_sample is None or args.max_sample > 0
    ), "The number of samples must be greater than 0"
    assert (
        args.max_token is None or args.max_token > 0
    ), "The number of samples must be greater than 0"
    assert (
        args.order is None or args.order > 1
    ), "The n-gram order must be greater than 1"
    assert (
        args.dim_sys is None or args.dim_sys > 0
    ), "The embedding dimensions must be greater than or equal to 0"
    assert (
        args.dim_entry is None or args.dim_entry > 0
    ), "The embedding dimensions must be greater than or equal to 0"
    assert (
        args.dim_ret is None or args.dim_ret > 0
    ), "The embedding dimensions must be greater than or equal to 0"
    assert (
        args.dim_proc is None or args.dim_proc > 0
    ), "The embedding dimensions must be greater than or equal to 0"
    assert (
        args.dim_pid is None or args.dim_pid > 0
    ), "The embedding dimensions must be greater than or equal to 0"
    assert (
        args.dim_tid is None or args.dim_tid > 0
    ), "The embedding dimensions must be greater than or equal to 0"
    assert (
        args.dim_time is None or args.dim_time > 0
    ), "The embedding dimensions must be greater than or equal to 0"
    assert (
        args.dim_order is None or args.dim_order > 0
    ), "The embedding dimensions must be greater than or equal to 0"
    assert (
        args.n_head is None or args.n_head > 0
    ), "The number of heads must be greater than 0"
    assert (
        args.n_hidden is None or args.n_hidden > 0
    ), "The number of units must be greater than 0"
    assert (
        args.n_layer is None or args.n_layer > 0
    ), "The number of layers must be greater than 0"
    assert (
        args.dropout is None or args.dropout >= 0 and args.dropout <= 1
    ), "The dropout probability must be greater than 0 and lower than 1"
    assert args.window is None or all(
        int(x) > 0 for x in args.window.split(",")
    ), "The window size must be greater than 0"
    assert args.dilatation is None or all(
        int(x) > 0 for x in args.dilatation.split(",")
    ), "The dilatation must be greater than 0 (1 = no dilatation)"
    assert args.global_att is None or all(
        int(x) >= 0 for x in args.global_att.split(",")
    ), "The global attention(s) must be greater or equal to 0"
    assert (
        args.batch is None or args.batch > 0
    ), "The number of sample per batch must be greater than 0"
    assert (
        args.n_update is None or args.n_update > 0
    ), "The number of updates must be greater than 0"
    assert (
        args.eval is None or args.eval > 0
    ), "The number of updates before evaluating the model must be greater "
    "than 0"
    assert (
        args.n_layer is None or args.n_layer > 0
    ), "The learning rate must be greater than 0"
    assert (
        args.warmup_steps is None or args.warmup_steps >= 0
    ), "The number of warmup steps must be greater than 0"
    assert (
        args.clip is None or args.clip > 0
    ), "The gradients' maximum norm must be greater than 0"
    assert args.ls is None or (
        args.ls >= 0 and args.ls <= 1
    ), "The label smoothing coefficient must be greater than 0 and lower "
    "than 1"
    assert (
        args.early_stopping_patience is None
        or args.early_stopping_patience > 0
    ), "The number of updates before early stopping must be greater than 0"
    assert (
        args.reduce_lr_patience is None or args.reduce_lr_patience > 0
    ), "The number of updates before reducing the learning rate must be "
    "greater than 0"

    if args.model != "ngram":
        assert args.gpu is not None, "Neural networks require a GPU"
        assert (
            len(args.gpu.split(",")) <= torch.cuda.device_count()
        ), f"Only {torch.cuda.device_count()} GPU available"

    if args.model == "lstm":
        if args.chk:
            args.chk = False
            print("Checkpoint is not implemented for the LSTM")
        if args.tfixup:
            args.tfixup = False
            print("T-fixup is not defined for the LSTM")
        if args.activation is not None:
            args.activation = None
            print("The activation functions are fixed for the LSTM")

    if args.model != "longformer":
        if args.window is not None:
            args.window = None
            print("The window attention is only defined for the Longformer")
        if args.dilatation is not None:
            args.dilatation = None
            print("The dilation is only defined for the Longformer")
        if args.global_att is not None:
            args.global_att = None
            print("The global attention is only defined for the Longformer")

    if os.path.exists(args.log_folder):
        print(f"{args.log_folder} already exists")

    return args


###############################################################################
# Data
###############################################################################


def load_trace(file_path):
    """Load the trace located in path.

    Args:
        file_path (str): Path to the LTTng trace folder.

    Returns:
        babeltrace.TraceCollection: A collection of one trace.
    """
    # Load babeltrace in the function to remove the import if the dataset
    # has already been generated (babeltrace not available on Compute Canada)
    try:
        import bt2
    except ImportError:
        raise ImportError(
            "Library bt2 is not available (https://babeltrace.org)"
        )

    return bt2.TraceCollectionMessageIterator(file_path)


def get_events(trace_collection, keys=None):
    """Return a generator of events. An event is a dict with the key the
    arguement's name.

    Args:
        trace_collection (babeltrace.TraceCollection): Trace from which
            to read the events.
        keys (dict, optional): Dictionary of the multiple ways of the arguments
            to consider in addition to name and elapsed time between events.

    Returns:
        generator: A generator of events.
    """
    # Load babeltrace in the function to remove the import if the dataset
    # has already been generated (babeltrace not available on Compute Canada)
    try:
        import bt2
    except ImportError:
        raise ImportError(
            "Library bt2 is not available " "(https://babeltrace.org)"
        )

    for msg in trace_collection:
        if type(msg) is not bt2._EventMessageConst:
            continue

        if (
            "syscall" not in msg.event.name
            and "event_handler" not in msg.event.name
        ):
            continue

        event = dict()
        event["name"] = msg.event.name
        event["timestamp"] = msg.default_clock_snapshot.ns_from_origin

        if (
            event["name"] == "httpd:enter_event_handler"
            or event["name"] == "httpd:exit_event_handler"
        ):
            if msg.event.payload_field["connection_state"] is None:
                event["connection_state"] = -1
            else:
                event["connection_state"] = int(
                    msg.event.payload_field["connection_state"]
                )
        else:
            event["connection_state"] = -1

        for k, v in keys.items():
            try:
                event[v] = msg.event[k]
            except KeyError:
                continue

        yield event


def get_requests(events):
    """Split individual requests from Apache. Note that this implementation
    is not the fastest, but requires very little memory.

    Args:
        events (generator): Generator of event.

    Yields:
        list: A list of events corresponding to a request.
    """
    # Dictionary of active threads
    threads = {}

    for event in events:

        # Start the request for a specific thread
        if event["name"] == "httpd:enter_event_handler":
            # Filter connections that lingers (not real requests)
            if event["connection_state"] not in [6, 7]:
                threads[event["tid"]] = []

        # End the request for a specific thread
        elif event["name"] == "httpd:exit_event_handler":
            if event["tid"] in threads:
                if threads[event["tid"]]:
                    yield threads[event["tid"]]
                del threads[event["tid"]]

        # Add the system calls in all currently recording thread
        else:
            for request in threads.values():
                request.append(event)


def generate_dataset(file_path, dict_sys, dict_proc, train=False):
    """Generate the dataset and write it iteratively into a file
    that will be iteratively read by the Dataloader.

    Args:
        file_path (str): Path to the file to load.
        dict_sys (dataset.Dictionary): Vocabulary of system call names.
        dict_proc (dataset.Dictionary): Vocabulary of process names.
        train (bool): Whether to update the dictionaries.
    """
    # Open the trace
    trace = load_trace(file_path)

    # Open the file
    f = open(f"{file_path}/data.txt", "w")

    # Mapping to consider the multiple way of denoting each argument
    # (e.g., the tid may be stored as 'tid' or 'vtid')
    keys = {
        "vtid": "tid",
        "tid": "tid",
        "vpid": "pid",
        "pid": "pid",
        "procname": "procname",
        "ret": "ret",
    }

    start = time()

    # Skip the first 1,000 requests as issues may occur when tracing starts.
    # Note that 1,000 requests per second were sent, so it amounts to skipping
    # the first second (which is on the cautious side)
    # MUST BE CHANGED TO 100 FOR THE TOY DATASETS (#TODO)
    for i, request in enumerate(
        itertools.islice(get_requests(get_events(trace, keys)), 1000, None)
    ):
        print(
            f"\rReading {file_path:40s}: {i:9d} "
            f"({timedelta(seconds=round(time() - start))})",
            file=sys.stderr,
            end="",
        )
        # Start a sequence with the token [START] with no argument (0s)
        call = [dict_sys.get_idx("[START]")]
        proc = [dict_proc.get_idx("[START]")]
        entry, duration, pid, tid, ret = [0], [0], [0], [0], [0]
        prev_timestp = None

        for event in request:
            # Get system call and process names
            sysname = event["name"].replace("syscall_", "")
            sysname = sysname.replace("entry_", "").replace("exit_", "")
            procname = str(event["procname"])

            # If it is the train set
            if train:
                # Add system call name to dictionary
                dict_sys.add_word(sysname)
                # Add process name to dicitonary
                dict_proc.add_word(procname)

            # Append system call name
            call.append(dict_sys.get_idx(sysname))
            # Append entry (1), exit (2), or none (0)
            if "entry" in event["name"]:
                entry.append(1)
            elif "exit" in event["name"]:
                entry.append(2)
            else:
                entry.append(0)
            # Append elapsed time between events
            if prev_timestp is not None:
                duration.append(event["timestamp"] - prev_timestp)
            else:
                duration.append(0)
            prev_timestp = event["timestamp"]
            # Append process name
            proc.append(dict_proc.get_idx(procname))
            # Append pid
            pid.append(event["pid"])
            # Append tid
            tid.append(event["tid"])
            # Append return value
            if "entry" in event["name"]:
                ret.append(0)  # start event (no return value)
            elif event["ret"] >= 0:
                ret.append(1)  # success
            else:
                ret.append(2)  # failure

        # End the sequence with the token [END] with no argument (0s)
        call.append(dict_sys.get_idx("[END]"))
        proc.append(dict_proc.get_idx("[END]"))
        entry.append(0)
        duration.append(0)
        pid.append(0)
        tid.append(0)
        ret.append(0)

        f.write(",".join(map(str, call)) + ";")
        f.write(",".join(map(str, entry)) + ";")
        f.write(",".join(map(str, duration)) + ";")
        f.write(",".join(map(str, proc)) + ";")
        f.write(",".join(map(str, pid)) + ";")
        f.write(",".join(map(str, tid)) + ";")
        f.write(",".join(map(str, ret)) + ";")
        # Add the duration in ms
        f.write(str(sum(duration) / 1e6) + "\n")

    # Close the file
    f.close()

    print(
        f"\rReading {file_path:40s}: {i:9d} "
        f"({timedelta(seconds=round(time() - start))})",
        file=sys.stderr,
    )


def dataset_stat(file_path, dict_sys, dict_proc, name, log_folder):
    """Load a dataset from the file given in argument and print its
    statistics.

    Args:
        file_path (str): Path to the file to load.
        dict_sys (dataset.Dictionary): Vocabulary of system call names.
        dict_proc (dataset.Dictionary): Vocabulary of process names.
        name (str): Name of the dataset.
        it (int): Iteration number (log folder).
    """
    # Create directories if necessary
    if not os.path.exists(f"{log_folder}/datasets/{name}"):
        os.makedirs(f"{log_folder}/datasets/{name}")

    # Request length
    length, duration, call, proc = [], [], [], []
    with open(file_path, "r") as f:
        for line in f:
            line = line.split(";")
            length.append(len(line[0].split(",")))
            duration.append(float(line[-1]))
            call.append(list(map(int, (line[0].split(",")))))
            proc.append(list(map(int, (line[3].split(",")))))

    print("=" * 100 + f"\n{f'{name} Set':^100s}\n" + "=" * 100)
    print(f"{'Number of requests':30}: {len(length):68,}")
    print(f"{'Min requests length':30}: {min(length):68,}")
    print(
        f"{'Mean requests length':30}: "
        f"{np.mean(length):57.1f} ± {np.std(length):8.1f}"
    )
    print(f"{'Max requests length':30}: {max(length):68,}")
    print(f"{'Min request duration':30}: {min(duration):66.2f}ms")
    print(
        f"{'Mean request duration':30}: "
        f"{np.mean(duration):57.2f} ± {np.std(duration):6.2f}ms"
    )
    print(f"{'Max request duration':30}: {max(duration):66.2f}ms")

    # Plot the duration distribution and syscall/process names histograms
    plot_duration(duration, name, log_folder)
    plot_length(length, name, log_folder)
    plot_hist(call, dict_sys.idx2word, name, "System Call", log_folder)
    plot_hist(proc, dict_proc.idx2word, name, "Process", log_folder)


def collate_fn(data):
    """Construct a batch by padding the sequence.
    Args:
        data (tuple): Tensors to pad.
    Returns:
        tuple: Padded tensors.
    """
    data = list(zip(*data))
    data, req_duration = data[:-1], data[-1]
    size = list(map(len, data[0]))
    pad_data = [
        torch.zeros(len(size), max(size), dtype=torch.int64) for _ in data
    ]
    for i, args in enumerate(data):
        for j, sample in enumerate(args):
            pad_data[i][j][: size[j]] = torch.tensor(sample)
    pad_data = [args.type(torch.int64) for args in pad_data]
    pad_mask = (pad_data[0] == 0).type(torch.bool)
    return pad_data, pad_mask, req_duration


###############################################################################
# n-gram
###############################################################################


# Add n-1 padding at the start and end. Since there is already one START and
# END token, only add n-2 padding (2 = token START) and end (3 = token END)
def nltk_ngram(file_path, n, max_sample):
    """Extract n-grams from the data in the file given in parameter.

    Args:
        file_path (str): File path of the dataset
        n (int): Order of the n-gram.

    Returns:
        nltk.lm.NgramCounter: NLTK n-gram counter.
    """
    start = time()
    with open(file_path, "r") as f:
        counter = NgramCounter(
            (
                ngrams(
                    [2] * (n - 2)
                    + list(map(int, line.split(";")[0].split(",")))[:-1],
                    n,
                )
                for line in itertools.islice(f, max_sample)
            )
        )
    print(
        f"{n}-grams extraction done in "
        f"{timedelta(seconds=round(time() - start))}"
    )
    return counter


def ngram_eval(file_path, counter, n, name, max_sample):
    """Evaluate the n-gram.

    Args:
        file_path (str): File path of the dataset
        counter (nltk.lm.NgramCounter): NLTK n-gram counter
        n (int): Order of the n-gram
        name (string, optional): Name of the dataset. Defaults to None.
    """
    correct, total = 0, 0
    with open(file_path, "r") as f:
        for line in itertools.islice(f, max_sample):
            seq = [2] * (n - 2) + list(
                map(int, line.split(";")[0].split(","))
            )[:-1]
            pred = [
                seq[i + n - 1]
                == counter[tuple(seq[i : i + n - 1])].most_common()[0][0]
                if counter[tuple(seq[i : i + n - 1])]
                else False
                for i in range(len(seq) - n + 1)
            ]
            total += len(pred)
            correct += sum(pred)
    print(f"{name:30}: {f'acc {correct / total:.1%}':>68}")


def ood_detection_ngram(
    counter,
    n,
    epsilon,
    path_valid_id,
    paths_valid_ood,
    val_ood_to_test,
    path_test_id,
    paths_test_ood,
    log_folder,
    max_sample,
):

    # Create directories if necessary
    if not os.path.exists(f"{log_folder}/evaluation/ood"):
        os.makedirs(f"{log_folder}/evaluation/ood")

    # Validation ID
    ppl_id = []
    with open(path_valid_id[1], "r") as f:
        for line in itertools.islice(f, max_sample):
            seq = [2] * (n - 2) + list(
                map(int, line.split(";")[0].split(","))
            )[:-1]
            log_likelihood = sum(
                math.log(
                    counter[tuple(seq[i : i + n - 1])][seq[i + n - 1]]
                    / counter[tuple(seq[i : i + n - 1])].N()
                    if (
                        counter[tuple(seq[i : i + n - 1])]
                        and counter[tuple(seq[i : i + n - 1])][seq[i + n - 1]]
                        > 0
                    )
                    else epsilon
                )
                for i in range(len(seq) - n + 1)
            )
            ppl_id.append(math.exp(-log_likelihood / (len(seq) - n + 1)))

    # Test ID
    ppl_id_test = []
    with open(path_test_id[1], "r") as f:
        for line in itertools.islice(f, max_sample):
            seq = [2] * (n - 2) + list(
                map(int, line.split(";")[0].split(","))
            )[:-1]
            log_likelihood = sum(
                math.log(
                    counter[tuple(seq[i : i + n - 1])][seq[i + n - 1]]
                    / counter[tuple(seq[i : i + n - 1])].N()
                    if (
                        counter[tuple(seq[i : i + n - 1])]
                        and counter[tuple(seq[i : i + n - 1])][seq[i + n - 1]]
                        > 0
                    )
                    else epsilon
                )
                for i in range(len(seq) - n + 1)
            )
            ppl_id_test.append(math.exp(-log_likelihood / (len(seq) - n + 1)))

    for name, path in paths_valid_ood.items():

        # FOR VALIDATION SET:
        ppl_ood = []
        with open(path, "r") as f:
            for line in itertools.islice(f, max_sample):
                seq = [2] * (n - 2) + list(
                    map(int, line.split(";")[0].split(","))
                )[:-1]
                log_likelihood = sum(
                    math.log(
                        counter[tuple(seq[i : i + n - 1])][seq[i + n - 1]]
                        / counter[tuple(seq[i : i + n - 1])].N()
                        if (
                            counter[tuple(seq[i : i + n - 1])]
                            and counter[tuple(seq[i : i + n - 1])][
                                seq[i + n - 1]
                            ]
                            > 0
                        )
                        else epsilon
                    )
                    for i in range(len(seq) - n + 1)
                )
                ppl_ood.append(math.exp(-log_likelihood / (len(seq) - n + 1)))

        ppl = ppl_id + ppl_ood
        y_true = [0] * len(ppl_id) + [1] * len(ppl_ood)

        accuracy, precision, recall, fscore = [], [], [], []
        thresholds = np.arange(
            min(ppl), max(ppl), step=(max(ppl) - min(ppl)) / 100,
        )

        for t in thresholds:
            y_pred = [1 if p > t else 0 for p in ppl]
            precision.append(sklearn.metrics.precision_score(y_true, y_pred))
            recall.append(sklearn.metrics.recall_score(y_true, y_pred))
            accuracy.append(sklearn.metrics.accuracy_score(y_true, y_pred))
            fscore.append(sklearn.metrics.f1_score(y_true, y_pred))

        # Get best score based on the validation set
        # and use it to evaluate the test set
        id_best_threshold = np.argmax(fscore)
        best_threhold_value = thresholds[id_best_threshold]

        ############
        # Log
        ############
        auroc = sklearn.metrics.roc_auc_score(y_true, ppl)
        print(f"{name}:")
        print(f"{'    AUROC':30}: {auroc:68.2%}")
        print(f"{'    Recall':30}: {recall[id_best_threshold]:68.2%}")
        print(f"{'    Precision':30}: {precision[id_best_threshold]:68.2%}")
        print(f"{'    F-score':30}: {np.max(fscore):68.2%}")
        print(f"{'    Accuracy':30}: {accuracy[id_best_threshold]:68.2%}")

        ############
        # ROC curve
        ############
        # Create figure
        fig = plt.figure(figsize=(10, 6), tight_layout=True)
        ax = fig.add_subplot(111)
        # Set colors
        dark_gray = "#808080"
        # Hide the right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        # Change axes and tick color
        ax.spines["bottom"].set_color(dark_gray)
        ax.tick_params(axis="x", colors=dark_gray)
        ax.spines["left"].set_color(dark_gray)
        ax.tick_params(axis="y", colors=dark_gray)
        ax.xaxis.label.set_color(dark_gray)
        ax.yaxis.label.set_color(dark_gray)
        # Plot
        fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, ppl)
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], color=dark_gray, lw=1)
        # Labels
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        # Make title the length of the graph
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="11%", pad=0)
        cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)
        for x in cax.spines:
            cax.spines[x].set_visible(False)
        cax.spines["top"].set_visible(False)
        cax.set_facecolor(dark_gray)
        at = AnchoredText(
            f"Receiver Operating Characteristic (AUROC: {auroc:.2f})",
            loc=6,
            pad=0,
            prop=dict(backgroundcolor=dark_gray, size=18, color="w"),
        )
        at.patch.set_edgecolor("none")
        cax.add_artist(at)
        # Save figure
        plt.savefig(f"{log_folder}/evaluation/ood/ROC_{name}.png", dpi=300)
        plt.close("all")

        # FOR TEST SET:
        # Get respective test set and its path
        name_test = val_ood_to_test[name]
        path_test = paths_test_ood[val_ood_to_test[name]]

        ppl_ood_test = []
        with open(path_test, "r") as f:
            for line in itertools.islice(f, max_sample):
                seq = [2] * (n - 2) + list(
                    map(int, line.split(";")[0].split(","))
                )[:-1]
                log_likelihood = sum(
                    math.log(
                        counter[tuple(seq[i : i + n - 1])][seq[i + n - 1]]
                        / counter[tuple(seq[i : i + n - 1])].N()
                        if (
                            counter[tuple(seq[i : i + n - 1])]
                            and counter[tuple(seq[i : i + n - 1])][
                                seq[i + n - 1]
                            ]
                            > 0
                        )
                        else epsilon
                    )
                    for i in range(len(seq) - n + 1)
                )
                ppl_ood_test.append(
                    math.exp(-log_likelihood / (len(seq) - n + 1))
                )

        ppl_test = ppl_id_test + ppl_ood_test
        y_true = [0] * len(ppl_id_test) + [1] * len(ppl_ood_test)
        y_pred = [1 if p > best_threhold_value else 0 for p in ppl_test]

        ############
        # Log
        ############
        precision = sklearn.metrics.precision_score(y_true, y_pred)
        recall = sklearn.metrics.recall_score(y_true, y_pred)
        accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
        fscore = sklearn.metrics.f1_score(y_true, y_pred)
        auroc = sklearn.metrics.roc_auc_score(y_true, ppl_test)

        print(f"{name_test}:")
        print(f"{'    AUROC':30}: {auroc:68.2%}")
        print(f"{'    Recall':30}: {recall:68.2%}")
        print(f"{'    Precision':30}: {precision:68.2%}")
        print(f"{'    F-score':30}: {fscore:68.2%}")
        print(f"{'    Accuracy':30}: {accuracy:68.2%}")

        ############
        # ROC curve
        ############
        # Create figure
        fig = plt.figure(figsize=(10, 6), tight_layout=True)
        ax = fig.add_subplot(111)
        # Set colors
        dark_gray = "#808080"
        # Hide the right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        # Change axes and tick color
        ax.spines["bottom"].set_color(dark_gray)
        ax.tick_params(axis="x", colors=dark_gray)
        ax.spines["left"].set_color(dark_gray)
        ax.tick_params(axis="y", colors=dark_gray)
        ax.xaxis.label.set_color(dark_gray)
        ax.yaxis.label.set_color(dark_gray)
        # Plot
        fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, ppl_test)
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], color=dark_gray, lw=1)
        # Labels
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        # Make title the length of the graph
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="11%", pad=0)
        cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)
        for x in cax.spines:
            cax.spines[x].set_visible(False)
        cax.spines["top"].set_visible(False)
        cax.set_facecolor(dark_gray)
        at = AnchoredText(
            f"Receiver Operating Characteristic (AUROC: {auroc:.2f})",
            loc=6,
            pad=0,
            prop=dict(backgroundcolor=dark_gray, size=18, color="w"),
        )
        at.patch.set_edgecolor("none")
        cax.add_artist(at)
        # Save figure
        plt.savefig(
            f"{log_folder}/evaluation/ood/ROC_{name_test}.png", dpi=300
        )
        plt.close("all")


def plot_perplexity_ngram(
    counter,
    n,
    epsilon,
    path_id,
    paths_ood,
    log_folder,
    max_sample=None,
    n_sample=10,
):

    # Create directories if necessary
    if not os.path.exists(f"{log_folder}/evaluation/ppl"):
        os.makedirs(f"{log_folder}/evaluation/ppl")

    # In distribution
    ppl_id, length_id, duration_id = [], [], []

    with open(path_id[1], "r") as f:
        for line in itertools.islice(f, max_sample):
            seq = [2] * (n - 2) + list(
                map(int, line.split(";")[0].split(","))
            )[:-1]
            ll_id = sum(
                math.log(
                    counter[tuple(seq[i : i + n - 1])][seq[i + n - 1]]
                    / counter[tuple(seq[i : i + n - 1])].N()
                    if (
                        counter[tuple(seq[i : i + n - 1])]
                        and counter[tuple(seq[i : i + n - 1])][seq[i + n - 1]]
                        > 0
                    )
                    else epsilon
                )
                for i in range(len(seq) - n + 1)
            )

            # Concatenate the computed values for the batch
            length_id.append(len(seq) - n + 1)
            duration_id.append(float(line.split(";")[-1]))
            ppl_id.append(math.exp(-ll_id / (len(seq) - n + 1)))

        # Out of distribution
        for name_ood, path_ood in paths_ood.items():
            ppl_ood, length_ood, duration_ood = [], [], []
            with open(path_ood, "r") as f:
                for line in itertools.islice(f, max_sample):
                    seq = [2] * (n - 2) + list(
                        map(int, line.split(";")[0].split(","))
                    )[:-1]
                    ll_ood = sum(
                        math.log(
                            counter[tuple(seq[i : i + n - 1])][seq[i + n - 1]]
                            / counter[tuple(seq[i : i + n - 1])].N()
                            if (
                                counter[tuple(seq[i : i + n - 1])]
                                and counter[tuple(seq[i : i + n - 1])][
                                    seq[i + n - 1]
                                ]
                                > 0
                            )
                            else epsilon
                        )
                        for i in range(len(seq) - n + 1)
                    )

                    # Concatenate the computed values for the batch
                    length_ood.append(len(seq) - n + 1)
                    duration_ood.append(float(line.split(";")[-1]))
                    ppl_ood.append(math.exp(-ll_ood / (len(seq) - n + 1)))

                _names = [path_id[0], name_ood]
                _duration = [duration_id, duration_ood]
                _length = [length_id, length_ood]
                _ppl = [ppl_id, ppl_ood]
                _plot_duration_ppl(
                    _names, _duration, _ppl, log_folder, name_ood
                )
                _plot_length_ppl(_names, _length, _ppl, log_folder, name_ood)
                _plot_dist_ppl(_names, _ppl, log_folder, name_ood)


###############################################################################
# Train & evaluate the model
###############################################################################


def train(
    rank,
    model,
    n_syscall,
    n_process,
    n_head,
    n_hidden,
    n_layer,
    dropout,
    dim_sys,
    dim_entry,
    dim_ret,
    dim_proc,
    dim_pid,
    dim_tid,
    dim_order,
    dim_time,
    activation,
    tfixup,
    train_dataset,
    valid_dataset,
    n_update,
    reduce_lr_patience,
    early_stopping_patience,
    warmup_steps,
    optimizer_alg,
    lr,
    ls,
    clip,
    eval,
    batch,
    gpu,
    chk,
    amp,
    log_folder,
    window,
    dilatation,
    global_att,
):
    """Create the dataloaders, build the model, and train it using
    DistributedDataParallel.

    Args:
        rank (int): Index of the GPU (mp.spawn).
        model (str): Model type.
        n_syscall (int): number of distinct system call names.
        n_process (int): number of distinct process names.
        n_head (int): Number of heads.
        n_hidden (int): Number of hidden units.
        n_layer ([type]): Number of layers.
        dropout ([type]): Probability of dropout.
        dim_sys (int): Dimension of the system call name embedding.
        dim_entry (int): Dimension of the entry/exit embedding.
        dim_ret (int): Dimension of the return value embedding.
        dim_proc (int): Dimension of the process name embedding.
        dim_pid (int): Dimension of the PID encoding.
        dim_tid (int): Dimension of the TID encoding.
        dim_order (int): Dimension of the order encoding.
        dim_time (int): Dimension of the encoding of the elapsed time
        between events encoding.
        activation (str): Activation function
        train_dataset (torch.utils.data.IterableDataset): Training set
        valid_dataset (torch.utils.data.IterableDataset): validation set
        n_update (int): Maximum number of updates.
        reduce_lr_patience (int): Number of iterations without improvements
        before decreasing the learning rate.
        early_stopping_patience (int): Number of iterations without
        improvements before stopping the training.
        warmup_steps (int): Number of updates before the learning reaches
        its peak.
        optimizer_alg (str): Which optimizer to use during training
        lr (float): Learning rate.
        ls (float): Label smoothing coefficient.
        clip (float): Maximum gradient norm
        eval (int): Number of updates between two evaluations.
        batch (int): Batch size.
        gpu (int): List of GPU id
        chk (bool): Gradient checkpointing.
        it (int): Iteration number (log folder).
        window (List[int]): LongSelfAttention window size for each encoder
        layer.
        dilatation (List[int]) Dilatation for LongSelfAttention each encoder
        layer. 1 = no dilatation.
        global_att (List[int]) Indexes to greant global attention.
    """
    world_size = len(gpu)
    device = gpu[rank]

    # Initialize the process
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank,
    )

    # Log into a file
    sys.stdout = open(f"{log_folder}/log.txt", "a")

    # Create model
    if model == "lstm":
        model = LSTM(
            n_syscall,
            n_process,
            n_hidden,
            n_layer,
            dropout,
            dim_sys,
            dim_entry,
            dim_ret,
            dim_proc,
            dim_pid,
            dim_tid,
            dim_order,
            dim_time,
        ).to(device)

    if model == "transformer":
        model = Transformer(
            n_syscall,
            n_process,
            n_head,
            n_hidden,
            n_layer,
            dropout,
            dim_sys,
            dim_entry,
            dim_ret,
            dim_proc,
            dim_pid,
            dim_tid,
            dim_order,
            dim_time,
            activation,
            tfixup,
        ).to(device)

    if model == "longformer":
        model = MyLongformer(
            n_syscall,
            n_process,
            n_head,
            n_hidden,
            n_layer,
            dropout,
            dim_sys,
            dim_entry,
            dim_ret,
            dim_proc,
            dim_pid,
            dim_tid,
            dim_order,
            dim_time,
            activation,
            tfixup,
            window,
            dilatation,
            global_att,
        ).to(device)

    # Move the model to GPU with id rank
    model = DDP(model, device_ids=[device], find_unused_parameters=False)

    # Loss
    criterion = LabelSmoothingCrossEntropy(label_smoothing=ls)

    # Optimizer
    if optimizer_alg == "ranger":
        optimizer = Ranger(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initialize variables
    update_since_best, _val_loss = 0, 0
    epoch, best_val_loss, update = 0, sys.maxsize, 1
    train_loss, val_loss, train_acc, val_acc, grad_norm = [], [], [], [], []
    total_train_loss, total_train_pred, total_train_correct = 0, 0, 0

    # Dataloader
    train_dataset.rank = rank
    train_dataset.world_size = world_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
    )

    # Gradient Scaler for mixed-precision
    scaler = GradScaler(enabled=amp)

    if rank == 0:
        # Log the model's information
        print("=" * 100 + f"\n{'Model':^100s}\n" + "=" * 100)
        params = filter(lambda p: p.requires_grad, model.parameters())
        n_params = sum([np.prod(p.size()) for p in params])
        print(f"{'Number of parameters':30}: {n_params:68,}")

        # Log the device(s) and gradient checkpointing
        for i in gpu:
            dname = torch.cuda.get_device_name(f"cuda:{i}")
            print(f"{'Device':30}: {dname:>68}" if i == 0 else f"{dname:>100}")

        print(
            f"{'Gradient Checkpointing':30}: "
            f"{'Enabled' if chk else 'Disabled':>68}"
        )
        print(
            f"{'Mixed-Precision':30}: "
            f"{'Enabled' if amp else 'Disabled':>68}"
        )

        print("=" * 100 + f"\n{'Training':^100s}\n" + "=" * 100)

    start, train_time = time(), time()

    while update < n_update:

        # Increment the epoch counter
        epoch += 1

        for data, pad_mask, _ in train_loader:

            # Increment the update counter
            update += 1

            # Stop training after n_update
            if update > n_update:
                break

            # Compute and update the learning rate
            if warmup_steps is not None and warmup_steps > 0:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr * min(
                        math.pow(update, -0.5) * math.pow(warmup_steps, 0.5),
                        update * math.pow(warmup_steps, -1),
                    )

            # Reset gradient
            optimizer.zero_grad()

            # Send tensors to device
            X = [x.to(device) for x in data[:-1]]
            y = data[-1].to(device)
            pad_mask = pad_mask.type(torch.bool).to(device)

            with autocast(enabled=amp):
                # Get prediction
                out = model(*X, pad_mask, chk)

                # Mask the padding for the LabelSmoothingCrossEntropy
                y = torch.masked_select(y, ~pad_mask).reshape(-1)
                out = torch.masked_select(
                    out, ~pad_mask.unsqueeze(-1)
                ).reshape(-1, n_syscall)

                # Compute loss
                loss = criterion(out, y)

            # Scales loss. Calls backward() on scaled loss to create
            # scaled gradients.
            scaler.scale(loss).backward()

            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)

            if clip is not None:
                # Since the gradients of optimizer's assigned params are
                # unscaled, clips as usual
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            # Optimizer's gradients are already unscaled, so scaler.step does
            # not unscale them, although it still skips optimizer.step() if
            # the gradients contain infs or NaNs.
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()

            # Collect the metrics
            total_train_loss += float(loss.item())
            total_train_pred += float(y.size(0))
            total_train_correct += float(
                (torch.max(out, dim=-1)[1] == y).sum().item()
            )

            # Collect the gradient magnitude
            if rank == 0:
                grad_norm.append(
                    sum(
                        p.grad.data.norm(2).item() ** 2
                        for p in model.parameters()
                    )
                    ** 0.5
                )

            # Every eval updates, evaluate and collect metrics
            if update % eval == 0 and rank == 0:

                # Get average duration per batch in ms
                avg_d = (time() - start) * 1000 / eval

                # Evaluate model
                _val_loss, _val_acc, _time_eval = evaluate(
                    model, valid_dataset, batch, criterion, n_syscall, gpu[0],
                )

                # Append metric
                train_loss.append(total_train_loss / eval)
                train_acc.append(total_train_correct / total_train_pred)
                val_loss.append(_val_loss)
                val_acc.append(_val_acc)

                # Save the metric for later comparison
                np.savez(
                    f"{log_folder}/training/eval.npz",
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_acc=train_acc,
                    val_acc=val_acc,
                )

                # Save the model if the validation loss is the best so far
                if _val_loss < best_val_loss - 0.001:
                    with open(f"{log_folder}/model", "wb") as f:
                        torch.save(model.module.state_dict(), f)
                        best_val_loss = _val_loss
                        update_since_best = 0
                else:
                    update_since_best += 1

                # Display summary of the update
                print(
                    f"Updates {update:8d} "
                    f"epoch {epoch:2d} "
                    f"loss {train_loss[-1]:5.3f} "
                    f"val_loss {val_loss[-1]:5.3f} "
                    f"acc {train_acc[-1]:5.1%} "
                    f"val_acc {val_acc[-1]:5.1%} "
                    f"optimization @ {avg_d:3.0f}ms/batch "
                    f"inference @ {_time_eval:3.0f}ms/batch "
                    f"lr {optimizer.param_groups[0]['lr']:3.2e} "
                    f"peak_mem {torch.cuda.max_memory_allocated(0)/1e6:5.0f}Mo"
                )
                torch.cuda.reset_peak_memory_stats(device)

                # Plot the accuracy, the loss, and the gradient norm
                plot_accuracy(train_acc, val_acc, eval, log_folder)
                plot_loss(train_loss, val_loss, eval, log_folder)
                plot_grad_norm(grad_norm, clip, log_folder)

                # Prepare to resume training
                model.train()
                total_train_loss = 0
                total_train_pred = 0
                total_train_correct = 0
                start = time()

            if update % eval == 0:
                # Only Tensors on GPU can be broadcasted
                update_since_best = torch.Tensor([update_since_best]).to(
                    device
                )
                _val_loss = torch.Tensor([_val_loss]).to(device)

                # Broadcast from GPU
                dist.broadcast(update_since_best, gpu[0])
                dist.broadcast(_val_loss, gpu[0])

                # Divide the learning rate by 10 if no improvements for
                # at least reduce_lr_patience evaluation steps
                if (
                    reduce_lr_patience is not None
                    and update_since_best > reduce_lr_patience
                ):
                    lr *= 0.1

            # Early stopping
            if (
                early_stopping_patience is not None
                and update_since_best == early_stopping_patience
            ):
                break
        if (
            early_stopping_patience is not None
            and update_since_best == early_stopping_patience
        ):
            break

    if rank == 0:
        print(
            f"Training done in {timedelta(seconds=round(time() - train_time))}"
        )

    # Destroy the process group
    dist.destroy_process_group(dist.group.WORLD)


def evaluate(model, dataset, batch, criterion, n_syscall, device):
    """Evaluate the model on the loader using the criterion.

    Args:
        model (torch.nn.Module): Network to evaluate.
        dataset (torch.utils.data.IterableDataset): Iterable Dataset.
        batch (int): Batch size.
        criterion (torch.nn): Loss function.
        n_syscall (int): Vocabulary size.

    Returns:
        tuple: The evaluation loss and accuracy.
    """
    # Dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
    )

    # Evaluate model
    model.eval()
    total_val_loss, total_val_pred, total_val_correct, time_per_batch = (
        0,
        0,
        0,
        0,
    )
    with torch.no_grad():
        for i, (data, pad_mask, _) in enumerate(dataloader, 1):
            # Start timer
            start = time()

            # Send tensors to device
            X = [x.to(device) for x in data[:-1]]
            y = data[-1].to(device)
            pad_mask = pad_mask.to(device)

            # Get prediction
            out = model(*X, pad_mask, chk=False)

            # Mask the padding for the LabelSmoothingCrossEntropy
            y = torch.masked_select(y, ~pad_mask).reshape(-1)
            out = torch.masked_select(out, ~pad_mask.unsqueeze(-1)).reshape(
                -1, n_syscall
            )

            # Compute loss
            loss = criterion(out, y)

            # Stop timer
            time_per_batch += time() - start

            # Collect metric
            total_val_loss += float(loss.item())
            total_val_pred += float(y.size(0))
            total_val_correct += float(
                (torch.max(out, dim=-1)[1] == y).sum().item()
            )

    return (
        total_val_loss / i,
        total_val_correct / total_val_pred,
        time_per_batch * 1000 / i,
    )


def ood_detection(
    model,
    dataset_id,
    datasets_ood,
    val_ood_to_test,
    dataset_id_test,
    datasets_ood_test,
    batch,
    n_syscall,
    device,
    log_folder,
):
    # Create directories if necessary
    if not os.path.exists(f"{log_folder}/evaluation/ood"):
        os.makedirs(f"{log_folder}/evaluation/ood")

    # Initialize the Cross-Entropy loss
    criterion = CrossEntropyLoss(ignore_index=0, reduction="none")

    # Send the model and the loss on the GPU
    torch.cuda.set_device(device)
    model.cuda(device)
    criterion.cuda(device)

    # Evaluate model
    model.eval()
    ppl_id_valid = []
    ppl_id_test = []

    with torch.no_grad():
        # Create the dataloader for VALIDATION set
        dataloader_id = DataLoader(
            dataset_id[1],
            batch_size=batch,
            collate_fn=collate_fn,
            num_workers=8,
            pin_memory=True,
        )

        for data, pad_mask, _ in dataloader_id:
            # Send tensors to device
            X = [x.to(device) for x in data[:-1]]
            y = data[-1].to(device)
            pad_mask = pad_mask.to(device)

            # Get prediction
            out = model(*X, pad_mask, chk=False)

            # Compute the perplexity
            loss = criterion(out.reshape(-1, n_syscall), y.reshape(-1))
            loss = loss.reshape(y.shape)
            loss = torch.sum(loss, 1) / torch.sum(~pad_mask, 1)

            # Concatenate the computed values for the batch
            ppl_id_valid.extend(torch.exp(loss).cpu().detach().tolist())

        # Create the dataloader for TEST set
        dataloader_id_test = DataLoader(
            dataset_id_test[1],
            batch_size=batch,
            collate_fn=collate_fn,
            num_workers=8,
            pin_memory=True,
        )
        for data, pad_mask, _ in dataloader_id_test:
            # Send tensors to device
            X = [x.to(device) for x in data[:-1]]
            y = data[-1].to(device)
            pad_mask = pad_mask.to(device)

            # Get prediction
            out = model(*X, pad_mask, chk=False)

            # Compute the perplexity
            loss = criterion(out.reshape(-1, n_syscall), y.reshape(-1))
            loss = loss.reshape(y.shape)
            loss = torch.sum(loss, 1) / torch.sum(~pad_mask, 1)

            # Concatenate the computed values for the batch
            ppl_id_test.extend(torch.exp(loss).cpu().detach().tolist())

        for name, dataset_ood in datasets_ood.items():
            # For VALIDATION set:
            ppl_ood_valid = []

            # Create the dataloader
            dataloader_ood = DataLoader(
                dataset_ood,
                batch_size=batch,
                collate_fn=collate_fn,
                num_workers=8,
                pin_memory=True,
            )

            for data, pad_mask, _ in dataloader_ood:
                # Send tensors to device
                X = [x.to(device) for x in data[:-1]]
                y = data[-1].to(device)
                pad_mask = pad_mask.to(device)

                # Get prediction
                out = model(*X, pad_mask, chk=False)

                # Compute the perplexity
                loss = criterion(out.reshape(-1, n_syscall), y.reshape(-1))
                loss = loss.reshape(y.shape)
                loss = torch.sum(loss, 1) / torch.sum(~pad_mask, 1)

                # Concatenate the computed values for the batch
                ppl_ood_valid.extend(torch.exp(loss).cpu().detach().tolist())

            ppl = ppl_id_valid + ppl_ood_valid
            y_true = [0] * len(ppl_id_valid) + [1] * len(ppl_ood_valid)

            accuracy, precision, recall, fscore = [], [], [], []
            thresholds = np.arange(
                min(ppl), max(ppl), step=(max(ppl) - min(ppl)) / 100,
            )
            for t in thresholds:
                y_pred = [1 if p > t else 0 for p in ppl]

                precision.append(
                    sklearn.metrics.precision_score(y_true, y_pred)
                )
                recall.append(sklearn.metrics.recall_score(y_true, y_pred))
                accuracy.append(sklearn.metrics.accuracy_score(y_true, y_pred))
                fscore.append(sklearn.metrics.f1_score(y_true, y_pred))

            id_best_threshold = np.argmax(fscore)
            best_threshold = thresholds[id_best_threshold]

            ############
            # Log
            ############
            auroc = sklearn.metrics.roc_auc_score(y_true, ppl)
            print(f"{name}:")
            print(f"{'    AUROC':30}: {auroc:68.2%}")
            print(f"{'    Recall':30}: {recall[id_best_threshold]:68.2%}")
            print(
                f"{'    Precision':30}: {precision[id_best_threshold]:68.2%}"
            )
            print(f"{'    F-score':30}: {np.max(fscore):68.2%}")
            print(f"{'    Accuracy':30}: {accuracy[id_best_threshold]:68.2%}")

            ############
            # ROC curve
            ############
            # Create figure
            fig = plt.figure(figsize=(10, 6), tight_layout=True)
            ax = fig.add_subplot(111)
            # Set colors
            dark_gray = "#808080"
            # Hide the right and top spines
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            # Change axes and tick color
            ax.spines["bottom"].set_color(dark_gray)
            ax.tick_params(axis="x", colors=dark_gray)
            ax.spines["left"].set_color(dark_gray)
            ax.tick_params(axis="y", colors=dark_gray)
            ax.xaxis.label.set_color(dark_gray)
            ax.yaxis.label.set_color(dark_gray)
            # Plot
            fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, ppl)
            plt.plot(fpr, tpr)
            plt.plot([0, 1], [0, 1], color=dark_gray, lw=1)
            # Labels
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # Make title the length of the graph
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("top", size="11%", pad=0)
            cax.get_xaxis().set_visible(False)
            cax.get_yaxis().set_visible(False)
            for x in cax.spines:
                cax.spines[x].set_visible(False)
            cax.spines["top"].set_visible(False)
            cax.set_facecolor(dark_gray)
            at = AnchoredText(
                f"Receiver Operating Characteristic (AUROC: {auroc:.2f})",
                loc=6,
                pad=0,
                prop=dict(backgroundcolor=dark_gray, size=18, color="w"),
            )
            at.patch.set_edgecolor("none")
            cax.add_artist(at)
            # Save figure
            plt.savefig(f"{log_folder}/evaluation/ood/ROC_{name}.png", dpi=300)
            plt.close("all")

            # For TEST set:
            ppl_ood_test = []

            # Get respective test data set
            name_ood_test = val_ood_to_test[name]
            dataset_ood_test = datasets_ood_test[name_ood_test]

            # Create the dataloader
            dataloader_ood = DataLoader(
                dataset_ood_test,
                batch_size=batch,
                collate_fn=collate_fn,
                num_workers=8,
                pin_memory=True,
            )

            for data, pad_mask, _ in dataloader_ood:
                # Send tensors to device
                X = [x.to(device) for x in data[:-1]]
                y = data[-1].to(device)
                pad_mask = pad_mask.to(device)

                # Get prediction
                out = model(*X, pad_mask, chk=False)

                # Compute the perplexity
                loss = criterion(out.reshape(-1, n_syscall), y.reshape(-1))
                loss = loss.reshape(y.shape)
                loss = torch.sum(loss, 1) / torch.sum(~pad_mask, 1)

                # Concatenate the computed values for the batch
                ppl_ood_test.extend(torch.exp(loss).cpu().detach().tolist())

            ppl_test = ppl_id_test + ppl_ood_test

            y_true = [0] * len(ppl_id_test) + [1] * len(ppl_ood_test)
            y_pred = [1 if p > best_threshold else 0 for p in ppl_test]

            precision = sklearn.metrics.precision_score(y_true, y_pred)
            recall = sklearn.metrics.recall_score(y_true, y_pred)
            accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
            fscore = sklearn.metrics.f1_score(y_true, y_pred)

            ############
            # Log
            ############
            auroc = sklearn.metrics.roc_auc_score(y_true, ppl_test)
            print(f"{name_ood_test}:")
            print(f"{'    AUROC':30}: {auroc:68.2%}")
            print(f"{'    Recall':30}: {recall:68.2%}")
            print(f"{'    Precision':30}: {precision:68.2%}")
            print(f"{'    F-score':30}: {np.max(fscore):68.2%}")
            print(f"{'    Accuracy':30}: {accuracy:68.2%}")

            ############
            # ROC curve
            ############
            # Create figure
            fig = plt.figure(figsize=(10, 6), tight_layout=True)
            ax = fig.add_subplot(111)
            # Set colors
            dark_gray = "#808080"
            # Hide the right and top spines
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            # Change axes and tick color
            ax.spines["bottom"].set_color(dark_gray)
            ax.tick_params(axis="x", colors=dark_gray)
            ax.spines["left"].set_color(dark_gray)
            ax.tick_params(axis="y", colors=dark_gray)
            ax.xaxis.label.set_color(dark_gray)
            ax.yaxis.label.set_color(dark_gray)
            # Plot
            fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, ppl_test)
            plt.plot(fpr, tpr)
            plt.plot([0, 1], [0, 1], color=dark_gray, lw=1)
            # Labels
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # Make title the length of the graph
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("top", size="11%", pad=0)
            cax.get_xaxis().set_visible(False)
            cax.get_yaxis().set_visible(False)
            for x in cax.spines:
                cax.spines[x].set_visible(False)
            cax.spines["top"].set_visible(False)
            cax.set_facecolor(dark_gray)
            at = AnchoredText(
                f"Receiver Operating Characteristic (AUROC: {auroc:.2f})",
                loc=6,
                pad=0,
                prop=dict(backgroundcolor=dark_gray, size=18, color="w"),
            )
            at.patch.set_edgecolor("none")
            cax.add_artist(at)
            # Save figure
            plt.savefig(
                f"{log_folder}/evaluation/ood/ROC_{name_ood_test}.png", dpi=300
            )
            plt.close("all")


def plot_grad_norm(grad_norm, clip, log_folder):
    # Create figure
    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    ax = fig.add_subplot(111)
    # Set colors
    dark_gray = "#808080"
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Change axes and tick color
    ax.spines["bottom"].set_color(dark_gray)
    ax.tick_params(axis="x", colors=dark_gray)
    ax.spines["left"].set_color(dark_gray)
    ax.tick_params(axis="y", colors=dark_gray)
    ax.xaxis.label.set_color(dark_gray)
    ax.yaxis.label.set_color(dark_gray)
    # Plot
    plt.plot(range(1, len(grad_norm) + 1), grad_norm, lw=1)
    if clip is not None:
        plt.axhline(clip, lw=1, c=dark_gray)
    # Labels
    plt.xlabel("Update")
    plt.ylabel("Gradient Norm")
    # Make title the length of the graph
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="11%", pad=0)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    for x in cax.spines:
        cax.spines[x].set_visible(False)
    cax.spines["top"].set_visible(False)
    cax.set_facecolor(dark_gray)
    at = AnchoredText(
        "Gradient L2-Norm",
        loc=6,
        pad=0,
        prop=dict(backgroundcolor=dark_gray, size=18, color="w"),
    )
    at.patch.set_edgecolor("none")
    cax.add_artist(at)
    # Save figure
    plt.savefig(f"{log_folder}/training/grad_norm.png", dpi=300)
    plt.close("all")


###############################################################################
# Visualization
###############################################################################


def plot_hist(x, mapping, dir, name, log_folder):
    """Plot the histogram of system call or process names

    Args:
        x (list): Dataset of system call or process names.
        mapping (dict): Mapping from int to string.
        dir (str): Name of the dataset.
        name (str): Name (syscall/process).
        it (int): Iteration number (log folder).
    """
    # Count the number of occurence of each word
    count = [0 for _ in mapping]
    for _x in x:
        for w in _x:
            count[int(w)] += 1
    # Convert to probability
    count = [c / sum(count) for c in count]
    # Sort and keep the 10 most probable
    count, mapping = map(list, zip(*sorted(zip(count, mapping))))
    count = count[-9:]
    mapping = mapping[-9:]
    # Add 'other'
    count.insert(0, 1 - sum(count))
    mapping.insert(0, "other")
    # Create figure
    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    ax = fig.add_subplot(111)
    # Set colors
    dark_gray = "#808080"
    light_gray = "#D3D3D3"
    # Plot
    bins = [x - 0.5 for x in range(len(mapping) + 1)]
    n, bins, patches = plt.hist(
        mapping, bins=bins, weights=count, rwidth=0.8, orientation="horizontal"
    )
    # Hide the bottom, right and top spines and ticks
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tick_params(
        axis="y", which="both", left=False, right=False, labelleft=True
    )
    # Change color of other
    patches[0].set_fc(light_gray)
    # For each bar: Place a label
    for i, (c, p) in enumerate(zip(count, patches)):
        x_value = p.get_width()
        y_value = p.get_y() + p.get_height() / 2
        if x_value > 0.01:
            plt.annotate(
                "{:.0%}".format(c),
                (x_value, y_value),
                color="w" if i != 0 else "k",
                xytext=(-2, 0),
                textcoords="offset points",
                va="center",
                ha="right",
            )
    # Change colors and labels of Y axis
    ax.spines["left"].set_color(dark_gray)
    # Add the name to the y-axis
    ax.tick_params(axis="y", colors=dark_gray, labelsize=14)
    ax.set_yticks(range(len(mapping)))
    ax.set_yticklabels(mapping)
    ax.tick_params(axis="x", colors="w")
    # Make title the length of the graph
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="11%", pad=0)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    for x in cax.spines:
        cax.spines[x].set_visible(False)
    cax.spines["top"].set_visible(False)
    cax.set_facecolor(dark_gray)
    at = AnchoredText(
        f"{name} Names in the {dir}",
        loc=6,
        pad=0,
        prop=dict(backgroundcolor=dark_gray, size=18, color="w"),
    )
    at.patch.set_edgecolor("none")
    cax.add_artist(at)
    # Save figurex
    name = name.replace(" ", "_").lower()
    plt.savefig(f"{log_folder}/datasets/{dir}/hist_{name}.png", dpi=300)
    plt.close("all")


def plot_loss(train, val, eval, log_folder):
    """Plot the loss as a function of the model updates.

    Args:
        train (list): Losses on the training set.
        val (list): Losses on the validation set.
        eval (int): Number of updates between two evaluations.
        it (int): Iteration number (log folder).
    """
    # Create figure
    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    ax = fig.add_subplot(111)
    # Set colors
    dark_gray = "#808080"
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Change axes and tick color
    ax.spines["bottom"].set_color(dark_gray)
    ax.tick_params(axis="x", colors=dark_gray)
    ax.spines["left"].set_color(dark_gray)
    ax.tick_params(axis="y", colors=dark_gray)
    ax.xaxis.label.set_color(dark_gray)
    ax.yaxis.label.set_color(dark_gray)
    # Plot
    x = range(eval, (len(train) + 1) * eval, eval)
    ax.plot(x, train, color="C0")
    ax.annotate(
        "Train {:6.3f}".format(train[-1]),
        xy=(x[-1], train[-1]),
        xytext=(5, -5 if train[-1] < val[-1] else 5),
        size=12,
        textcoords="offset points",
        va="center",
        color="C0",
    )
    ax.plot(x, val, color="C1")
    ax.annotate(
        "Valid {:6.3f}".format(val[-1]),
        xy=(x[-1], val[-1]),
        xytext=(5, 5 if train[-1] < val[-1] else -5),
        size=12,
        textcoords="offset points",
        va="center",
        color="C1",
    )
    # Increase left margin
    lim = ax.get_xlim()
    right = lim[1] + (lim[1] - lim[0]) * 0.1
    ax.set_xlim(lim[0], right)
    # Labels
    plt.xlabel("Updates")
    plt.ylabel("Cross Entropy")
    # Make title the length of the graph
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="11%", pad=0)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    for x in cax.spines:
        cax.spines[x].set_visible(False)
    cax.spines["top"].set_visible(False)
    cax.set_facecolor(dark_gray)
    at = AnchoredText(
        "Cross-entropy During Training",
        loc=6,
        pad=0,
        prop=dict(backgroundcolor=dark_gray, size=18, color="w"),
    )
    at.patch.set_edgecolor("none")
    cax.add_artist(at)
    # Save figure
    plt.savefig(f"{log_folder}/training/loss.png", dpi=300)
    plt.close("all")


def plot_accuracy(train, val, eval, log_folder):
    """Plot the accuracy as a function of the model updates.

    Args:
        train (list): Accuracies on the training set.
        val (list): Accuracies on the validation set.
        eval (int): Number of updates between two evaluations.
        it (int): Iteration number (log folder).
    """
    # Create figure
    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    ax = fig.add_subplot(111)
    # Set colors
    dark_gray = "#808080"
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Change axes and tick color
    ax.spines["bottom"].set_color(dark_gray)
    ax.tick_params(axis="x", colors=dark_gray)
    ax.spines["left"].set_color(dark_gray)
    ax.tick_params(axis="y", colors=dark_gray)
    ax.xaxis.label.set_color(dark_gray)
    ax.yaxis.label.set_color(dark_gray)
    # Plot
    x = range(eval, (len(train) + 1) * eval, eval)
    ax.plot(x, train, color="C0")
    ax.annotate(
        "Train {:6.1%}".format(train[-1]),
        xy=(x[-1], train[-1]),
        xytext=(5, -5 if train[-1] < val[-1] else 5),
        size=12,
        textcoords="offset points",
        va="center",
        color="C0",
    )
    ax.plot(x, val, color="C1")
    ax.annotate(
        "Valid {:6.1%}".format(val[-1]),
        xy=(x[-1], val[-1]),
        xytext=(5, 5 if train[-1] < val[-1] else -5),
        size=12,
        textcoords="offset points",
        va="center",
        color="C1",
    )
    # Increase left margin
    lim = ax.get_xlim()
    right = lim[1] + (lim[1] - lim[0]) * 0.1
    ax.set_xlim(lim[0], right)
    # Labels
    plt.xlabel("Updates")
    plt.ylabel("Accuracy")
    # Make title the length of the graph
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="11%", pad=0)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    for x in cax.spines:
        cax.spines[x].set_visible(False)
    cax.spines["top"].set_visible(False)
    cax.set_facecolor(dark_gray)
    at = AnchoredText(
        "Accuracy During Training",
        loc=6,
        pad=0,
        prop=dict(backgroundcolor=dark_gray, size=18, color="w"),
    )
    at.patch.set_edgecolor("none")
    cax.add_artist(at)
    # Save figure
    plt.savefig(f"{log_folder}/training/accuracy.png", dpi=300)
    plt.close("all")


def _plot_duration_ppl(names, duration, ppl, log_folder, suffix):
    # Create figure
    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    ax = fig.add_subplot(111)
    # Set colors
    dark_gray = "#808080"
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Change axes and tick color
    ax.spines["bottom"].set_color(dark_gray)
    ax.tick_params(axis="x", colors=dark_gray)
    ax.spines["left"].set_color(dark_gray)
    ax.tick_params(axis="y", colors=dark_gray)
    ax.xaxis.label.set_color(dark_gray)
    ax.yaxis.label.set_color(dark_gray)
    y_min = min([np.mean(p) - 4 * np.std(p) for p in ppl])
    y_max = max([np.mean(p) + 4 * np.std(p) for p in ppl])
    x_min = min([np.mean(d) - 4 * np.std(d) for d in duration])
    x_max = max([np.mean(d) + 4 * np.std(d) for d in duration])
    # Plot
    for i, (n, d, p) in enumerate(zip(names, duration, ppl)):
        sns.kdeplot(
            data={"Duration": d, "Perplexity": p},
            x="Duration",
            y="Perplexity",
            color=f"C{i}",
            gridsize=500,
            alpha=0.5,
            fill=True,
            levels=10,
            thresh=0.05,
            label=n,
            clip=((max(0, x_min), x_max), (max(0, y_min), y_max)),
        )
    # Change the limits
    plt.xlim((max(0, x_min), x_max))
    plt.ylim((max(0, y_min), y_max))
    # Legend
    legend_elements = [
        Patch(facecolor=f"C{i}", alpha=0.5, label=n)
        for i, n in enumerate(names)
    ]
    legend = plt.legend(handles=legend_elements, frameon=False)
    plt.setp(legend.get_texts(), color=dark_gray)
    # Labels
    plt.xlabel("Duration (ms)")
    plt.ylabel("Perplexity")
    # Make title the length of the graph
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="11%", pad=0)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    for x in cax.spines:
        cax.spines[x].set_visible(False)
    cax.spines["top"].set_visible(False)
    cax.set_facecolor(dark_gray)
    at = AnchoredText(
        "Perplexity in Function of the Request Duration",
        loc=6,
        pad=0,
        prop=dict(backgroundcolor=dark_gray, size=18, color="w"),
    )
    at.patch.set_edgecolor("none")
    cax.add_artist(at)
    # Save figure
    plt.savefig(
        f"{log_folder}/evaluation/ppl/duration_perplexity_{suffix}.png",
        dpi=300,
    )
    plt.close("all")


def _plot_length_ppl(names, length, ppl, log_folder, suffix):
    # Create figure
    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    ax = fig.add_subplot(111)
    # Set colors
    dark_gray = "#808080"
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Change axes and tick color
    ax.spines["bottom"].set_color(dark_gray)
    ax.tick_params(axis="x", colors=dark_gray)
    ax.spines["left"].set_color(dark_gray)
    ax.tick_params(axis="y", colors=dark_gray)
    ax.xaxis.label.set_color(dark_gray)
    ax.yaxis.label.set_color(dark_gray)
    # Compute the limit of the plot
    y_min = min([np.mean(p) - 4 * np.std(p) for p in ppl])
    y_max = max([np.mean(p) + 4 * np.std(p) for p in ppl])
    x_min = min([np.mean(lgt) - 4 * np.std(lgt) for lgt in length])
    x_max = max([np.mean(lgt) + 4 * np.std(lgt) for lgt in length])
    # Plot
    for i, (n, l, p) in enumerate(zip(names, length, ppl)):
        sns.kdeplot(
            data={"Duration": l, "Perplexity": p},
            x="Duration",
            y="Perplexity",
            color=f"C{i}",
            gridsize=500,
            alpha=0.5,
            fill=True,
            levels=10,
            thresh=0.05,
            label=n,
            clip=((max(0, x_min), x_max), (max(0, y_min), y_max)),
        )
    # Change the limits
    plt.xlim((max(0, x_min), x_max))
    plt.ylim((max(0, y_min), y_max))
    # Legend
    legend_elements = [
        Patch(facecolor=f"C{i}", alpha=0.5, label=n)
        for i, n in enumerate(names)
    ]
    legend = plt.legend(handles=legend_elements, frameon=False)
    plt.setp(legend.get_texts(), color=dark_gray)
    # Labels
    plt.xlabel("Length (# system call)")
    plt.ylabel("Perplexity")
    # Make title the length of the graph
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="11%", pad=0)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    for x in cax.spines:
        cax.spines[x].set_visible(False)
    cax.spines["top"].set_visible(False)
    cax.set_facecolor(dark_gray)
    at = AnchoredText(
        "Perplexity in Function of the Sequence Length",
        loc=6,
        pad=0,
        prop=dict(backgroundcolor=dark_gray, size=18, color="w"),
    )
    at.patch.set_edgecolor("none")
    cax.add_artist(at)
    # Save figure
    plt.savefig(
        f"{log_folder}/evaluation/ppl/length_perplexity_{suffix}.png", dpi=300,
    )
    plt.close("all")


def _plot_dist_ppl(names, ppl, log_folder, suffix):
    # Create figure
    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    ax = fig.add_subplot(111)
    # Set colors
    dark_gray = "#808080"
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Change axes and tick color
    ax.spines["bottom"].set_color(dark_gray)
    ax.tick_params(axis="x", colors=dark_gray)
    ax.spines["left"].set_color(dark_gray)
    ax.tick_params(axis="y", colors=dark_gray)
    ax.xaxis.label.set_color(dark_gray)
    ax.yaxis.label.set_color(dark_gray)
    # Plot
    for i, (p, n) in enumerate(zip(ppl, names)):
        sns.kdeplot(
            p,
            shade=True,
            gridsize=1000,
            log_scale=True,
            color=f"C{i}",
            label=n,
        )
    legend = plt.legend(frameon=False)
    plt.setp(legend.get_texts(), color=dark_gray)
    # Labels
    plt.xlabel("Perplexity")
    # Make title the length of the graph
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="11%", pad=0)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    for x in cax.spines:
        cax.spines[x].set_visible(False)
    cax.spines["top"].set_visible(False)
    cax.set_facecolor(dark_gray)
    at = AnchoredText(
        "Perplexity Distribution",
        loc=6,
        pad=0,
        prop=dict(backgroundcolor=dark_gray, size=18, color="w"),
    )
    at.patch.set_edgecolor("none")
    cax.add_artist(at)
    # Save figure
    plt.savefig(
        f"{log_folder}/evaluation/ppl/dist_perplexity_{suffix}.png", dpi=300,
    )
    plt.close("all")


def _plot_sample_ppl(ppl, ll, log_folder, suffix, n_sample):
    if not os.path.exists(f"{log_folder}/evaluation/samples/{suffix}"):
        os.makedirs(f"{log_folder}/evaluation/samples/{suffix}")

    ppl, ll = zip(*sorted(zip(ppl, ll)))

    for i in range(-n_sample, n_sample):
        _ll = [x for x in ll[i] if x != 0]
        # Create figure
        fig = plt.figure(figsize=(10, 6), tight_layout=True)
        ax = fig.add_subplot(111)
        # Set colors
        dark_gray = "#808080"
        # Hide the right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        # Change axes and tick color
        ax.spines["bottom"].set_color(dark_gray)
        ax.tick_params(axis="x", colors=dark_gray)
        ax.spines["left"].set_color(dark_gray)
        ax.tick_params(axis="y", colors=dark_gray)
        ax.xaxis.label.set_color(dark_gray)
        ax.yaxis.label.set_color(dark_gray)
        # Plot
        plt.bar(range(len(_ll)), _ll, align="center")
        # Increase left margin
        lim = ax.get_xlim()
        right = lim[1] + (lim[1] - lim[0]) * 0.1
        ax.set_xlim(lim[0], right)
        plt.xlim((-1, len(_ll)))
        # Labels
        plt.xlabel("System Call Index")
        plt.ylabel("Negative Log-Likelihood")
        # Make title the length of the graph
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="11%", pad=0)
        cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)
        for x in cax.spines:
            cax.spines[x].set_visible(False)
        cax.spines["top"].set_visible(False)
        cax.set_facecolor(dark_gray)
        at = AnchoredText(
            "Event Likelihood of a Sample With a Perplexity "
            f"of {ppl[i]:.3f}",
            loc=6,
            pad=0,
            prop=dict(backgroundcolor=dark_gray, size=18, color="w"),
        )
        at.patch.set_edgecolor("none")
        cax.add_artist(at)
        # Save figure
        if i >= 0:
            filename = f"{log_folder}/evaluation/samples/{suffix}/good_{i + 1}"
        else:
            filename = f"{log_folder}/evaluation/samples/{suffix}/bad_{-i}"
        plt.savefig(f"{filename}.png", dpi=300)
        plt.close("all")


def plot_perplexity(
    model,
    dataset_id,
    datasets_ood,
    batch,
    n_syscall,
    device,
    log_folder,
    n_sample=10,
):
    """Plot the distribution of the perplexity of sequences in the
    dataloader, and the correlation between perplexity and sequence
    lengths.

    Args:
        model (torch.nn.Module): Trained model.
        datasets (dict): Dictionary of Iterable Datasets.
        batch (int): Batch size.
        n_syscall (int): Vocabulary size.
        it (int): Iteration number (log folder).
        n_sample (int, optional): Number of sample to consider. Default to 10.
    """
    # Create directories if necessary
    if not os.path.exists(f"{log_folder}/evaluation/ppl"):
        os.makedirs(f"{log_folder}/evaluation/ppl")

    # Initialize the Cross-Entropy loss
    criterion = CrossEntropyLoss(ignore_index=0, reduction="none")

    # Send the model and the loss on the GPU
    torch.cuda.set_device(device)
    model.cuda(device)
    model.eval()

    with torch.no_grad():

        # In distribution
        ppl_id, ll_id, length_id, duration_id = [], [], [], []

        # Create the dataloader
        dataloader_id = DataLoader(
            dataset_id[1],
            batch_size=batch,
            collate_fn=collate_fn,
            num_workers=8,
            pin_memory=True,
        )

        for data, pad_mask, req_duration in dataloader_id:
            # Send tensors to device
            X = [x.to(device) for x in data[:-1]]
            y = data[-1].to(device)
            pad_mask = pad_mask.to(device)

            # Get prediction
            out = model(*X, pad_mask, chk=False)

            # Compute the perplexity
            loss = criterion(out.reshape(-1, n_syscall), y.reshape(-1))
            loss = loss.reshape(y.shape)

            # Concatenate the computed values for the batch
            length_id.extend(torch.sum(~pad_mask, 1).cpu().detach().tolist())
            duration_id.extend(list(map(float, req_duration)))
            ppl_id.extend(
                torch.exp(torch.sum(loss, 1) / torch.sum(~pad_mask, 1))
                .cpu()
                .detach()
                .tolist()
            )
            ll_id.extend(loss.cpu().detach().tolist())

        # Out of distribution
        for name_ood, dataset in datasets_ood.items():
            ppl_ood, ll_ood, length_ood, duration_ood = [], [], [], []

            # Create the dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=batch,
                collate_fn=collate_fn,
                num_workers=8,
                pin_memory=True,
            )

            for data, pad_mask, req_duration in dataloader:
                # Send tensors to device
                X = [x.to(device) for x in data[:-1]]
                y = data[-1].to(device)
                pad_mask = pad_mask.to(device)

                # Get prediction
                out = model(*X, pad_mask, chk=False)

                # Compute the perplexity
                loss = criterion(out.reshape(-1, n_syscall), y.reshape(-1))
                loss = loss.reshape(y.shape)

                # Concatenate the computed values for the batch
                length_ood.extend(
                    torch.sum(~pad_mask, 1).cpu().detach().tolist()
                )
                duration_ood.extend(list(map(float, req_duration)))
                ppl_ood.extend(
                    torch.exp(torch.sum(loss, 1) / torch.sum(~pad_mask, 1))
                    .cpu()
                    .detach()
                    .tolist()
                )
                ll_ood.extend(loss.cpu().detach().tolist())

            _names = [dataset_id[0], name_ood]
            _duration = [duration_id, duration_ood]
            _length = [length_id, length_ood]
            _ppl = [ppl_id, ppl_ood]
            _plot_duration_ppl(_names, _duration, _ppl, log_folder, name_ood)
            _plot_length_ppl(_names, _length, _ppl, log_folder, name_ood)
            _plot_dist_ppl(_names, _ppl, log_folder, name_ood)
            _plot_sample_ppl(ppl_ood, ll_ood, log_folder, name_ood, n_sample)


def plot_duration(duration, name, log_folder):
    """Plot the distribution of requests' duration.

    Args:
        duration (list): Requests' duration.
        name (str): Name of the dataset.
        it (int): Iteration number (log folder).
    """
    # Create figure
    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    ax = fig.add_subplot(111)
    # Set colors
    dark_gray = "#808080"
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Change axes and tick color
    ax.spines["bottom"].set_color(dark_gray)
    ax.tick_params(axis="x", colors=dark_gray)
    ax.spines["left"].set_color(dark_gray)
    ax.tick_params(axis="y", colors=dark_gray)
    ax.xaxis.label.set_color(dark_gray)
    ax.yaxis.label.set_color(dark_gray)
    # Plot the histogram
    plt.hist(duration, bins=50, rwidth=0.8, range=(0, 10))
    # Remove frame
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Axis labels
    plt.xlabel("Request duration (ms)")
    plt.ylabel("Count")
    # Make title the length of the graph
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="11%", pad=0)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    for x in cax.spines:
        cax.spines[x].set_visible(False)
    cax.spines["top"].set_visible(False)
    cax.set_facecolor(dark_gray)
    at = AnchoredText(
        f"Request Duration in the {name} Set",
        loc=6,
        pad=0,
        prop=dict(backgroundcolor=dark_gray, size=18, color="w"),
    )
    at.patch.set_edgecolor("none")
    cax.add_artist(at)
    # Save and close the figure
    plt.savefig(f"{log_folder}/datasets/{name}/hist_duration.png", dpi=300)
    plt.close("all")


def plot_length(length, name, log_folder):
    """Plot the distribution of requests' length.

    Args:
        length (list): Requests' length.
        name (str): Name of the dataset.
        it (int): Iteration number (log folder).
    """
    # Create figure
    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    ax = fig.add_subplot(111)
    # Set colors
    dark_gray = "#808080"
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Change axes and tick color
    ax.spines["bottom"].set_color(dark_gray)
    ax.tick_params(axis="x", colors=dark_gray)
    ax.spines["left"].set_color(dark_gray)
    ax.tick_params(axis="y", colors=dark_gray)
    ax.xaxis.label.set_color(dark_gray)
    ax.yaxis.label.set_color(dark_gray)
    # Plot the histogram
    plt.hist(length, bins=50, rwidth=0.8)
    # Remove frame
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Axis labels
    plt.xlabel("Number of events")
    plt.ylabel("Count")
    # Make title the length of the graph
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="11%", pad=0)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    for x in cax.spines:
        cax.spines[x].set_visible(False)
    cax.spines["top"].set_visible(False)
    cax.set_facecolor(dark_gray)
    at = AnchoredText(
        f"Request Length in the {name} Set",
        loc=6,
        pad=0,
        prop=dict(backgroundcolor=dark_gray, size=18, color="w"),
    )
    at.patch.set_edgecolor("none")
    cax.add_artist(at)
    # Save and close the figure
    plt.savefig(f"{log_folder}/datasets/{name}/hist_length.png", dpi=300)
    plt.close("all")


def plot_attention(
    model, dataset, batch, n_layer, device, log_folder, n_sample=100
):
    """Plot the distribution of attention activations for each layer.

    Args:
        model (torch.nn.Module): Trained model.
        dataset (torch.utils.data.IterableDataset): Iterable Dataset.
        batch (int): Batch size
        n_layer (int): Number of layers.
        it (int): Iteration number (log folder).
        n_sample (int, optional): Number of batches. Defaults to 100.
    """

    # Create directories if necessary
    if not os.path.exists(f"{log_folder}/evaluation/attention"):
        os.makedirs(f"{log_folder}/evaluation/attention")

    # Create an empty temporary folder attn_output_weights
    if os.path.exists("/tmp/attn_output_weights"):
        shutil.rmtree("/tmp/attn_output_weights")
    os.makedirs("/tmp/attn_output_weights")

    # Send the model and the loss on the GPU
    torch.cuda.set_device(device)
    model.cuda(device)

    dataloader = DataLoader(
        dataset[1],
        batch_size=batch,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
    )

    model.eval()

    attn_layers = [None for _ in range(n_layer)]
    attn_counter = [0 for _ in range(n_layer)]

    with torch.no_grad():
        for data, pad_mask, _ in itertools.islice(
            dataloader, int(n_sample / batch)
        ):

            # Send tensors to device
            X = [x.to(device) for x in data[:-1]]
            pad_mask = pad_mask.to(device)

            # Get prediction
            model(*X, pad_mask, chk=False, save_attn=True)

            for file in os.listdir("/tmp/attn_output_weights/"):

                # Get the layer
                layer = int(file.split("_")[0])

                # Get the attention
                X = torch.load(
                    f"/tmp/attn_output_weights/{file}", map_location="cpu"
                ).numpy()

                # Increment counter of samples
                attn_counter[layer] += X.shape[0]

                # Sum the attention accross the batch dimension
                X = np.sum(X, axis=0)

                # Add the attention to the list with padding if necessary
                if attn_layers[layer] is None:
                    attn_layers[layer] = X
                else:
                    Y = attn_layers[layer]
                    d = max(X.shape[0], Y.shape[0])
                    X = np.pad(X, (0, d - X.shape[0]))
                    Y = np.pad(Y, (0, d - Y.shape[0]))
                    attn_layers[layer] = X + Y

                # Remove file to save space
                os.remove(f"/tmp/attn_output_weights/{file}")

    for i, attn in enumerate(attn_layers):

        # Compute the mean attention
        attn = attn / attn_counter[i]

        for size in [32, 64, 128, 256, 512, 1024, None]:
            attn_cropped = attn[:size, :size]

            # Create figure
            plt.figure(figsize=(10, 6), tight_layout=True)
            # Construct 2D histogram from data using the 'plasma' colormap
            sns.heatmap(
                attn_cropped,
                vmin=0.0,
                vmax=np.nanpercentile(
                    attn_cropped[attn_cropped.nonzero()], 98
                ),
                square=True,
            )
            # Title and labels
            plt.title(f"Attention Activation Patterns at Layer {i+1}")
            # Remove the axis
            plt.axis("off")
            # Save and close the figure
            plt.savefig(
                f"{log_folder}/evaluation/attention/layer_{i + 1}_{size}.png",
                dpi=300,
            )
            plt.close("all")

            # Multiply each row by the number of non-zero elements
            nonzero = np.count_nonzero(attn_cropped, axis=1, keepdims=True)
            attn_cropped_cmp = attn_cropped * nonzero

            # Create figure
            plt.figure(figsize=(10, 6), tight_layout=True)
            # Construct 2D histogram from data using the 'plasma' colormap
            sns.heatmap(
                attn_cropped_cmp,
                vmin=0.0,
                vmax=np.nanpercentile(
                    attn_cropped_cmp[attn_cropped_cmp.nonzero()], 98
                ),
                square=True,
            )
            # Title and labels
            plt.title(f"Attention Activation Patterns at Layer {i+1}")
            # Remove the axis
            plt.axis("off")
            # Save and close the figure
            plt.savefig(
                f"{log_folder}/evaluation/attention/layer_{i + 1}_{size}"
                "_compensated.png",
                dpi=300,
            )
            plt.close("all")

    # Remove the temporary folder
    shutil.rmtree("/tmp/attn_output_weights")


def plot_delay(
    model,
    dataset,
    batch,
    n_syscall,
    device,
    log_folder,
    n_delay=100,
    n_pos=100,
    n_sample=5,
):
    """Plot the average impact of multiple delays on the perplexity.
    Each delay is applied at every position, and the perplexity average
    for each delay. The plot show the perplexity in function of the
    delay.

    Args:
        model (torch.nn.Module): Trained model.
        dataset (torch.utils.data.IterableDataset): Iterable Dataset.
        batch (int): Batch size
        n_syscall (int): Vocabulary size.
        it (int): Iteration number (log folder).
        device (int): Device to be used
        delay (list): List of delays.
        n_sample (int, optional): Number of sample to consider. Default to 5.
    """

    # Initialize the Cross-Entropy loss
    criterion = CrossEntropyLoss(ignore_index=0, reduction="none")

    # Send the model and the loss on the GPU
    torch.cuda.set_device(device)
    model.cuda(device)
    criterion.cuda(device)

    # Dataloaders
    loader = DataLoader(
        dataset[1],
        batch_size=1,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
    )

    # Create directories if necessary
    if not os.path.exists(f"{log_folder}/evaluation/delay"):
        os.makedirs(f"{log_folder}/evaluation/delay")

    model.eval()

    with torch.no_grad():
        for i, (data, pad_mask, _) in enumerate(
            itertools.islice(loader, n_sample)
        ):

            # Compute the baseline
            # Send tensors to device
            X = [x.to(device) for x in data[:-1]]
            y = data[-1].to(device)
            pad_mask = pad_mask.to(device)

            # Get prediction
            out = model(*X, pad_mask, chk=False)

            # Compute the perplexity
            loss = criterion(out.reshape(-1, n_syscall), y.reshape(-1))
            loss = loss.reshape(y.shape)
            loss = torch.sum(loss, 1) / torch.sum(~pad_mask, 1)
            baseline_ppl = torch.exp(loss).cpu().detach().tolist()[0]

            # Compute the impact of delays on the perplexity
            pos = np.sort(
                np.random.choice(
                    range(2, len(data[0][0]) - 2),
                    min(len(data[0][0]) - 4, n_pos),
                    replace=False,
                )
            )
            delay = np.logspace(3, 6, num=n_delay, base=10, dtype=np.int64,)

            # Repeats the tensors along the specified dimensions (copies the
            # tensor’s data).
            data_delay = [d.repeat(n_pos * n_delay, 1) for d in data]

            # Shift the elapsed time between events
            for j, (p, d) in enumerate(itertools.product(pos, delay)):
                data_delay[2][j][p] += d

            cycle_delay = itertools.cycle(delay)
            ppl = {d: [] for d in delay}

            # Create the batch
            for ndx in range(0, n_pos * n_delay, batch):
                samples = [
                    d[ndx : min(ndx + batch, n_pos * n_delay)]
                    for d in data_delay
                ]
                pad_mask_repeat = pad_mask.repeat(len(samples[0]), 1)

                # Send tensors to device
                X = [x.to(device) for x in samples[:-1]]
                y = samples[-1].to(device)
                pad_mask_repeat = pad_mask_repeat.to(device)

                # Get prediction
                out = model(*X, pad_mask_repeat, chk=False)

                # Compute the perplexity
                loss = criterion(out.reshape(-1, n_syscall), y.reshape(-1))
                loss = loss.reshape(y.shape)
                loss = torch.sum(loss, 1) / torch.sum(~pad_mask, 1)
                _ppl = torch.exp(loss).cpu().detach().tolist()

                # Concatenate the computed values for the batch
                for p, d in zip(_ppl, cycle_delay):
                    ppl[d].append(p)

            x, ppl = zip(*ppl.items())
            mean = [np.mean(p) for p in ppl]
            lower = [np.mean(p) - np.std(p) for p in ppl]
            upper = [np.mean(p) + np.std(p) for p in ppl]

            # Create figure
            fig = plt.figure(figsize=(10, 6), tight_layout=True)
            ax = fig.add_subplot(111)
            # Set colors
            dark_gray = "#808080"
            # Hide the right and top spines
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            # Change axes and tick color
            ax.spines["bottom"].set_color(dark_gray)
            ax.tick_params(axis="x", colors=dark_gray)
            ax.spines["left"].set_color(dark_gray)
            ax.tick_params(axis="y", colors=dark_gray)
            ax.xaxis.label.set_color(dark_gray)
            ax.yaxis.label.set_color(dark_gray)
            # Plot
            plt.plot(x, mean)
            plt.xscale("log")
            plt.fill_between(x, lower, upper, alpha=0.3)
            plt.hlines(baseline_ppl, x[0], x[-1], color="r")
            # Increase left margin
            lim = ax.get_xlim()
            right = lim[1] + (lim[1] - lim[0]) * 0.1
            ax.set_xlim(lim[0], right)
            plt.xlim((x[0], x[-1]))
            # Labels
            plt.xlabel("Absolute Offset (ns)")
            plt.ylabel("Average Perplexity")
            # Make title the length of the graph
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("top", size="11%", pad=0)
            cax.get_xaxis().set_visible(False)
            cax.get_yaxis().set_visible(False)
            for x in cax.spines:
                cax.spines[x].set_visible(False)
            cax.spines["top"].set_visible(False)
            cax.set_facecolor(dark_gray)
            at = AnchoredText(
                "Perplexity as a Function of the Timestamp Offset",
                loc=6,
                pad=0,
                prop=dict(backgroundcolor=dark_gray, size=18, color="w"),
            )
            at.patch.set_edgecolor("none")
            cax.add_artist(at)
            # Add legend
            line = Line2D([0], [0], label="Baseline (no delay)", color="r")
            legend = plt.legend(
                frameon=False,
                handles=[line],
                loc="upper left",
                bbox_to_anchor=(0, 0.1),
            )
            plt.setp(legend.get_texts(), color=dark_gray)
            # Save figure
            plt.savefig(
                f"{log_folder}/evaluation/delay/{i + 1}.png", dpi=300,
            )
            plt.close("all")


def plot_projection(embedding, idx2word, name, log_folder, n_sample=5):
    """Plot a 2D projection of the embedding.
    Args:
        embedding (ndarray): embedding weight matrix
        idx2word (list): vocabulary
        name (string): name of the embedding
        it (int): iteration number (to name the figure)
        n_sample (int, optional): Number of sample to consider. Default to 5.
    """

    # Create directories if necessary
    if not os.path.exists(f"{log_folder}/evaluation/embedding"):
        os.makedirs(f"{log_folder}/evaluation/embedding")

    # Ignore FutureWarning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        for i in range(n_sample):
            # See https://opentsne.readthedocs.io/en/latest/parameters.html
            # for a guideline on how to set the hyperparameters

            model = TSNE(
                perplexity=30.0,
                early_exaggeration=12.0,
                n_components=2,
                learning_rate="auto",
                init="pca",
                random_state=i,
                n_jobs=-1,
            )

            projection = model.fit_transform(embedding)

            x, y = projection[:, 0], projection[:, 1]

            dark_gray = "#808080"
            plt.figure(figsize=(10, 6), tight_layout=True)
            plt.scatter(x, y, s=5)
            x_offset = 0.01 * (max(x) - min(x))
            y_offset = 0.01 * (max(y) - min(y))
            for j, txt in enumerate(idx2word):
                plt.annotate(
                    txt, (x[j] + x_offset, y[j] + y_offset), color=dark_gray
                )
            plt.axis("off")
            name = name.replace(" ", "_").lower()
            plt.savefig(
                f"{log_folder}/evaluation/embedding/{name}_{i}.png", dpi=300
            )
            plt.close()
