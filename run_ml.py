#!/usr/bin/env python3
import os
import sys
from util import util
import requests
import pandas as pd
import pickle
from typing import List, Dict, Tuple, Any


def run_classification(
    information_source_id: str,
    corpus_embeddings: Dict[str, List[List[float]]],
    corpus_labels: List[str],
    corpus_ids: List[str],
    training_ids: List[str],
):
    from util.active_transfer_learning import ATLClassifier

    print("progress: 0.05")
    classifier = ATLClassifier()
    prediction_probabilities = classifier.fit_predict(
        corpus_embeddings, corpus_labels, corpus_ids, training_ids
    )
    print("progress: 0.5")
    if os.path.exists("/inference"):
        pickle_path = os.path.join(
            "/inference", f"active-learner-{information_source_id}.pkl"
        )
        with open(pickle_path, "wb") as f:
            pickle.dump(classifier, f)
            print("Saved model to disk", flush=True)

    prediction_indices = prediction_probabilities.argmax(axis=1)
    predictions_with_probabilities = []
    for probas, index in zip(prediction_probabilities, prediction_indices):
        proba = probas[index]
        prediction = classifier.model.classes_[probas.argmax()]
        predictions_with_probabilities.append([proba, prediction])

    print("progress: 0.8")
    ml_results_by_record_id = {}
    for record_id, (probability, prediction) in zip(
        corpus_ids, predictions_with_probabilities
    ):
        if (
            probability > classifier.min_confidence
            and prediction in classifier.label_names
        ):
            ml_results_by_record_id[record_id] = (
                probability,
                prediction,
            )
    print("progress: 0.9")
    if len(ml_results_by_record_id) == 0:
        print("No records were predicted. Try lowering the confidence threshold.")
    return ml_results_by_record_id


def run_extraction(
    information_source_id: str,
    corpus_embeddings: Dict[str, List[Any]],
    corpus_labels: List[Tuple[str, str, List[Any]]],
    corpus_ids: List[str],
    training_ids: List[str],
):
    from util.active_transfer_learning import ATLExtractor

    print("progress: 0.05")
    extractor = ATLExtractor()
    predictions, probabilities = extractor.fit_predict(
        corpus_embeddings, corpus_labels, corpus_ids, training_ids
    )
    print("progress: 0.5")
    if os.path.exists("/inference"):
        pickle_path = os.path.join(
            "/inference", f"active-learner-{information_source_id}.pkl"
        )
        with open(pickle_path, "wb") as f:
            pickle.dump(extractor, f)
            print("Saved model to disk", flush=True)

    ml_results_by_record_id = {}
    amount = len(corpus_ids)
    for idx, (record_id, prediction, probability) in enumerate(
        zip(corpus_ids, predictions, probabilities)
    ):
        df = pd.DataFrame(
            list(zip(prediction, probability)),
            columns=["prediction", "probability"],
        )
        df["next"] = df["prediction"].shift(-1)
        predictions_with_probabilities = []
        new_start_idx = True
        for idx, row in df.loc[
            (df.prediction != util.CONSTANT__OUTSIDE)
            & (df.prediction.isin(extractor.label_names))
            & (df.probability > extractor.min_confidence)
        ].iterrows():
            if new_start_idx:
                start_idx = idx
                new_start_idx = False
            if row.prediction != row.next:
                prob = df.loc[start_idx:idx].probability.mean()
                end_idx = idx + 1
                predictions_with_probabilities.append(
                    [float(prob), row.prediction, start_idx, end_idx]
                )
                new_start_idx = True
        ml_results_by_record_id[record_id] = predictions_with_probabilities
        if idx % 100 == 0:
            progress = round((idx + 1) / amount, 4) * 0.5 + 0.5
            print("progress: ", progress)

    print("progress: 0.9")
    if len(ml_results_by_record_id) == 0:
        print("No records were predicted. Try lowering the confidence threshold.")
    return ml_results_by_record_id


if __name__ == "__main__":
    _, payload_url = sys.argv
    print("Preparing data for machine learning.")

    (
        information_source_id,
        corpus_embeddings,
        corpus_labels,
        corpus_ids,
        training_ids,
    ) = util.get_corpus()
    is_extractor = any([isinstance(val, list) for val in corpus_labels["manual"]])

    if is_extractor:
        print("Running extractor.")
        ml_results_by_record_id = run_extraction(
            information_source_id,
            corpus_embeddings,
            corpus_labels,
            corpus_ids,
            training_ids,
        )
    else:
        print("Running classifier.")
        ml_results_by_record_id = run_classification(
            information_source_id,
            corpus_embeddings,
            corpus_labels,
            corpus_ids,
            training_ids,
        )

    print("progress: 1")
    print("Finished execution.")
    requests.put(payload_url, json=ml_results_by_record_id)
