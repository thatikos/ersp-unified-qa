import argparse
from transformers import BertTokenizerFast, T5Tokenizer
from model import Retriever, FiDT5
from indexing import Indexer
import pickle
import numpy as np
import glob
from pathlib import Path
from data import Dataset, RetrieverCollator, ReaderCollator
from torch.utils.data import DataLoader, SequentialSampler
import tqdm
import utils
import evaluation
import torch

# def index_encoded_data(index, embedding_files, indexing_batch_size):
#     allids = []
#     allembeddings = np.array([])
#     for i, file_path in enumerate(embedding_files):
#         with open(file_path, 'rb') as fin:
#             ids, embeddings = pickle.load(fin)

#         allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
#         allids.extend(ids)
#         while allembeddings.shape[0] > indexing_batch_size:
#             allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

#     while allembeddings.shape[0] > 0:
#         allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)
        

# def add_embeddings(index, embeddings, ids, indexing_batch_size):
#     end_idx = min(indexing_batch_size, embeddings.shape[0])
#     ids_toadd = ids[:end_idx]
#     embeddings_toadd = embeddings[:end_idx]
#     ids = ids[end_idx:]
#     embeddings = embeddings[end_idx:]
#     index.index_data(ids_toadd, embeddings_toadd)
#     return embeddings, ids

# copy
def add_passages(data, passages, top_passages_and_scores):
    assert len(data) == len(top_passages_and_scores)
    for i, d in enumerate(data):
        results_and_scores = top_passages_and_scores[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(docs)
        d['ctxs'] =[
                {
                    'id': results_and_scores[0][c],
                    'title': docs[c][1],
                    'text': docs[c][0],
                    'score': scores[c],
                } for c in range(ctxs_num)
        ]

parser = argparse.ArgumentParser()

parser.add_argument("--retriever_path", required=True)
# parser.add_argument("--reader_path", required=True)
parser.add_argument("--passages_path", required=True)
parser.add_argument("--index_directory", required=True)
parser.add_argument("--validation_data", required=True)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--passage_maxlength", type=int, default=200)
parser.add_argument("--question_maxlength", type=int, default=40)
parser.add_argument("--answer_maxlength", type=int, default=40)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--num_documents", type=int, default=32)
parser.add_argument("--do_eval", action="store_true")

arr = []
acc = []
pre = []
arr.append(acc)
arr.append(pre)
def calculatingAccAndPresIndividual(top_ids_and_scores, batch):
    for query in batch:
        hit = 0
        for passage in query["ctxs"]:
            for ans in query['answers']:
                if ans in passage['text']:
                    hit += 1
        if hit > 0:
            arr[0].append(1)
            arr[1].append(hit/len(query["ctxs"]))
        else:
            arr[0].append(0)
            arr[1].append(0)
        
if __name__ == "__main__":
    opts = parser.parse_args()

    # loading retriever
    tokenizer_retriever = BertTokenizerFast.from_pretrained('bert-base-uncased')
    retriever = Retriever.from_pretrained(opts.retriever_path)
    retriever.eval()

    # loading reader
    tokenizer_reader = T5Tokenizer.from_pretrained('t5-base')
    # reader = FiDT5.from_pretrained(opts.reader_path)
    # reader.eval()

    # loading index
    index = Indexer(retriever.config.indexing_dimension)
    index.deserialize_from(opts.index_directory)

    # load data & passages
    # modify this 
    dataset = Dataset(opts.validation_data)
    passages = utils.load_passages(opts.passages_path)
    retrieval_collator = RetrieverCollator(tokenizer_retriever, opts.passage_maxlength, opts.question_maxlength)
    reader_collator = ReaderCollator(opts.passage_maxlength + opts.question_maxlength, tokenizer_reader, opts.answer_maxlength)
    data_loader = DataLoader(
        dataset = dataset,
        batch_size = opts.batch_size,
        sampler = SequentialSampler(dataset),
        num_workers = opts.num_workers,
        collate_fn = retrieval_collator
    )
# end copy

    # evaluation loop
    exactmatch = []
    total = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader):
            questions_embedding = retriever.embed_text(
                batch['retriever_token_ids'], 
                batch['retriever_masks'],
                apply_mask=retriever.config.apply_question_mask,
                extract_cls=retriever.config.extract_cls
            ).numpy()

            top_ids_and_scores = index.search_knn(questions_embedding, opts.num_documents)
            add_passages(batch['orig_batch'], passages, top_ids_and_scores)

            batch = reader_collator(batch)

            # outputs = reader.generate(
            #     input_ids=batch['passage_ids'],
            #     attention_mask=batch['passage_masks'],
            #     max_length=opts.answer_maxlength
            # )

            # if opts.do_eval:
            #     for k, o in enumerate(outputs):
            #             ans = tokenizer_reader.decode(o, skip_special_tokens=True)
            #             gold = dataset.get_example(batch['index'][k])['answers']
            #             score = evaluation.ems(ans, gold)
            #             total += 1
            #             exactmatch.append(score)
                       
            calculatingAccAndPresIndividual(top_ids_and_scores, batch["orig_batch"])
            
                             
    if opts.do_eval:
        # print(f'score = {np.mean(exactmatch)}')

        sumAcc = sum(arr[0]) / len(arr[0]) if len(arr[0]) != 0 else 0
        sumPrec = sum(arr[1]) / len(arr[1])  if len(arr[1]) != 0 else 0

        print(sumAcc, "accuracy")
        print(sumPrec, "precision")
        

            


    


