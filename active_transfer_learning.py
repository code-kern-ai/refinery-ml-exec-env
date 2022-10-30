from abc import ABC, abstractmethod
import util
from typing import Callable, List, Optional
from joblib import load, dump


def parametrized(decorator: Callable):
    def layer(*args, **kwargs):
        def repl(function: Callable):
            return decorator(function, *args, **kwargs)

        return repl

    return layer


@parametrized
def params_fit(function: Callable, embedding_name: str, train_test_split: float):
    def aux(*args, **kwargs):
        self = args[0]
        corpus_embeddings = args[1]
        corpus_labels = args[2]
        self.label_type = (
            "manual"  # in the future, this can be manually set by the user
        )
        self.embedding_name = embedding_name
        if isinstance(self, LearningExtractor):
            (
                embeddings_training,
                labels_training,
            ) = util.transform_corpus_extraction_fit(
                corpus_embeddings[embedding_name],
                corpus_labels[self.label_type],
                self.records_ids,
                self.training_ids,
            )
        elif isinstance(self, LearningClassifier):
            (
                embeddings_training,
                labels_training,
            ) = util.transform_corpus_classification_fit(
                corpus_embeddings[self.embedding_name],
                corpus_labels[self.label_type],
                self.records_ids,
                self.training_ids,
            )
        else:
            print(
                f"Unknown class {self.__class__.__name__}. Please contact the support."
            )

        return function(self, embeddings_training, labels_training)

    return aux


@parametrized
def params_inference(
    function: Callable,
    label_names: Optional[List[str]] = None,
    min_confidence: float = 0.8,
):
    def aux(*args, **kwargs):
        self = args[0]
        corpus_embeddings = args[1]
        self.min_confidence = min_confidence
        self.label_names = label_names
        if isinstance(self, LearningExtractor):
            embeddings_inference = util.transform_corpus_extraction_inference(
                corpus_embeddings[self.embedding_name]
            )
        elif isinstance(self, LearningClassifier):
            embeddings_inference = util.transform_corpus_classification_inference(
                corpus_embeddings[self.embedding_name]
            )
        else:
            print(
                f"Unknown class {self.__class__.__name__}. Please contact the support."
            )

        return function(self, embeddings_inference)

    return aux


class BaseModel(ABC):
    @abstractmethod
    def __init__(self):
        super().__init__()
        self.embedding_name = None
        self.min_confidence = None
        self.label_names = None

    def save_model_to_path(self, file_path: str):
        configs = {
            "model": self.model,
            "embedding_name": self.embedding_name,
            "min_confidence": self.min_confidence,
            "label_names": self.label_names,
        }
        dump(configs, file_path)


    def load_model_from_path(self, file_path: str):
        configs = load(file_path)
        self.model = configs["model"]
        self.embedding_name = configs["embedding_name"]
        self.min_confidence = configs["min_confidence"]
        self.label_names = configs["label_names"]

    @abstractmethod
    def fit(self, embeddings, labels):
        pass

    @abstractmethod
    def predict_proba(self, embeddings):
        pass

    def predict(self, embeddings):
        predictions = self.predict_proba(embeddings)
        if self.label_names is None:
            self.label_names = self.model.classes_
        return predictions

    def fit_predict(self, embeddings, labels, records_ids, training_ids):
        self.records_ids = records_ids
        self.training_ids = training_ids
        self.fit(embeddings, labels)
        return self.predict(embeddings)
        


class LearningClassifier(BaseModel):
    pass


class LearningExtractor(BaseModel):
    pass
