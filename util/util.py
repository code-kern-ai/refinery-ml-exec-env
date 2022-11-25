import json
import numpy as np
import pandas as pd

CONSTANT__OUTSIDE = "OUTSIDE"

pd.options.mode.chained_assignment = None  # default='warn'


# --- schema for corpus ---
# "embeddings": {
#   "distilbert-base-uncased": [...]
# },
# "labels": {
#     "manual": [...],
#     "programmatic": [...]
# }


def get_corpus():
    with open("input.json", "r") as infile:
        input_data = json.load(infile)
        information_source_id = input_data["information_source_id"]
        embedding_type = input_data["embedding_type"]
        embedding_name = input_data["embedding_name"]
        labels = input_data["labels"]
        ids = input_data["ids"]
        training_ids = input_data["active_learning_ids"]

    embedding_df = pd.read_csv("embedding.csv.bz2", quoting=1)

    try:
        if embedding_type == "ON_ATTRIBUTE":
            embeddings = {
                embedding_name: [
                    [float(y) for y in x[1:-1].split(", ")]
                    for x in embedding_df.data
                    if x != "data"
                ]
            }
        else:
            embeddings = {
                embedding_name: [
                    [[float(z) for z in y.split(", ")] for y in x[2:-2].split("], [")]
                    for x in embedding_df.data
                    if x != "data"
                ]
            }
    except Exception:
        print("Can't parse the embedding. Please contact the support.")
        raise ValueError("Can't parse the embedding. Please contact the support.")
    return (
        information_source_id,
        embeddings,
        labels,
        ids,
        training_ids,
    )


def transform_corpus_classification_inference(embeddings):
    return np.array(embeddings)


def transform_corpus_extraction_inference(embeddings_inference):
    return embeddings_inference


def transform_corpus_classification_fit(
    embeddings, labels_training, record_ids, training_ids
):
    """
    find indices that actually contain labels, reduce both embeddings and labels based on these indices, and return them
    """
    training_mask = [True if id in training_ids else False for id in record_ids]

    labels_training = np.array(labels_training)[training_mask]
    embedding_training = np.array(embeddings)[training_mask]
    return embedding_training, labels_training


def transform_corpus_extraction_fit(
    embeddings_inference, labels_, record_ids, training_ids
):
    """
    pad ragged array to the max length of the item length (i.e. [num records x item length x embedding dim] of the embeddings),
    find indices that actually contain labels, reduce both embeddings and labels based on these indices, and return them
    """

    df_labels = pd.DataFrame(labels_, columns=["record_id", "label_name", "token_list"])
    df_labels["idx"] = None
    for idx, record_id in enumerate(df_labels.record_id.unique()):
        if record_id in training_ids:
            df_labels_subset = df_labels.loc[df_labels.record_id == record_id]
            df_labels["idx"].loc[df_labels_subset.index] = idx
    df_labels = df_labels.dropna()

    labels_prepared = []
    for idx, embedding in enumerate(embeddings_inference):
        label_vector = np.full([len(embedding)], None)
        for _, row in df_labels.loc[df_labels.idx == idx].iterrows():
            for token_idx in row.token_list:
                label_vector[token_idx] = row.label_name
            np.place(label_vector, label_vector is None, CONSTANT__OUTSIDE)
        labels_prepared.append(label_vector.tolist())

    keep_idxs = list(df_labels.idx.unique())

    embeddings_training = []
    labels_training = []
    for keep_idx in keep_idxs:
        embeddings_training.append(embeddings_inference[keep_idx])
        labels_training.append(labels_prepared[keep_idx])

    return embeddings_training, labels_training
