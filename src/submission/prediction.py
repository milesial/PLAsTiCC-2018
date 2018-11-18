import csv
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import Process
from multiprocessing import Manager

from utils.constants import CLASSES
from utils.constants import NUM_TEST_EXAMPLES
from utils.data_utils import get_test_objects
from utils.data_utils import get_num_test_shards


def predict_test_probs_for_shard(queue, shard_id, prepare_data_fn, model_fn):
    for oid, passband_ts, meta_row in get_test_objects(shard_id):
        inp = prepare_data_fn(passband_ts, meta_row)
        output = model_fn(inp.reshape(1, -1))
        output = np.array(output, dtype=np.float16).squeeze()
        output = [oid, *output]
        queue.put(output)


def predict_test_probs(prepare_data_fn, model_fn, out_csv, n_workers):
    num_shards = get_num_test_shards()
    m = Manager()
    q = m.Queue()
    writer = Process(target=write_queue, args=(q, out_csv))
    writer.start()

    args = [(q, id, prepare_data_fn, model_fn) for id in range(num_shards)]
    with Pool(n_workers) as pool:
        pool.starmap(predict_test_probs_for_shard, args)
        pool.join()

    writer.join()


def write_queue(queue, out_csv):
    pbar = tqdm(total=NUM_TEST_EXAMPLES)
    with open(out_csv, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['object_id'] + [f'class_{i}' for i in CLASSES])
        while True:
            item = queue.get(block=True)
            writer.writerow(item)
            pbar.update(1)
