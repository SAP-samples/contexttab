# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import pickle

import torch
import zmq


from contexttab.data.sentence_embedder import SentenceEmbedder
from contexttab.utils.lru_cache import LRU_Cache
from contexttab.constants import ZMQ_PORT_DEFAULT



@torch.no_grad()
def process_data(data: list) -> list:
    # We sort them by length, so that the tokenizer can batch them efficiently
    # with minimal pad
    sorted_unique_texts = sorted(set([text for element in data for text in element['texts']]), key=len, reverse=True)

    # Check if we have the embeddings in the cache
    missing_texts = []
    found_texts = []
    all_results = []
    for text in sorted_unique_texts:
        result = cache[text]
        if result is None:
            missing_texts.append(text)
        else:
            found_texts.append(text)
            all_results.append(result)

    sorted_unique_texts = found_texts + missing_texts
    text_to_position = {text: position for position, text in enumerate(sorted_unique_texts)}
    ids_in_sorted_unique_texts = [[text_to_position[text] for text in element['texts']] for element in data]

    result_bytes = sentence_embedder.embed(missing_texts)
    for text, result in zip(missing_texts, result_bytes):
        cache[text] = result
    all_results.extend(result_bytes)

    assert len(all_results) == len(sorted_unique_texts)

    result = []
    for indices in ids_in_sorted_unique_texts:
        result.append(pickle.dumps([all_results[idx] for idx in indices]))

    return result


def main(port: int, context: zmq.Context):
    socket = context.socket(zmq.ROUTER)
    socket.bind(f'tcp://*:{port}')
    socket.setsockopt(zmq.SNDTIMEO, 0)
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    while True:
        socks = dict(poller.poll())

        if socket not in socks or socks[socket] != zmq.POLLIN:
            # This should never happen, really
            continue

        # gather messages from multiple clients
        data = []
        max_num_of_clients = 32
        for _ in range(max_num_of_clients):
            try:
                client, _, texts = socket.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                break
            else:
                data.append({'client': client, 'texts': pickle.loads(texts)})

        if not data:
            continue

        # process messages in batch
        results = process_data(data)
        for element, result in zip(data, results):
            socket.send_multipart([element['client'], b'', result])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmarking clients')
    parser.add_argument('-p', '--port', type=int, default=ZMQ_PORT_DEFAULT, help='Port number')
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index')
    parser.add_argument('-semn', '--sentence_embedding_model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='Sentence embedding model name')
    args = parser.parse_args()

    cache = LRU_Cache(max_size=100000)
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_idx}')
    else:
        device = torch.device('cpu')
    sentence_embedder = SentenceEmbedder(args.sentence_embedding_model_name, 32, device=device)
    global_context = zmq.Context()
    main(args.port, global_context)
