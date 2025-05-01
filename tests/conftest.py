from typing import Any, Callable, TypeVar, ValuesView
from pandas._libs.tslibs.offsets import CBMonthBegin
import pytest
import os
import numpy as np
import json
import torch

from multihopkg.exogenous.sun_models import KGEModel
from multihopkg.utils.data_splitting import TripleIds, read_triple

T = TypeVar('T')
ValidatorDict = dict[str, Callable[[T], bool]]

def pytest_addoption(parser):
    """Add command-line options for embedding tests."""
    parser.addoption(
        "--embeddings_path",
        action="store",
        default="./models/protatE_FBWikiV4",
        help="Path to embeddings directory to test",
    )

    parser.addoption(
        "--dataset_path",
        action="store",
        default="./data/FBWikiV4",
        help="Dataset to use for performance testing",
    )
    parser.addoption(
        "--test_percent",
        action="store",
        type=float,
        default=0.01,
        help="Percentage of the test set to use for performance testing. Test sets tend to be large so this useful for a quick test.",
    )
    parser.addoption("--model_name", action="store", default="pRotatE")

    # ------------ Adding Threshold to allow the performance to be tested-------------

    parser.addoption(
        "--thresholds",
        type=ValidatorDict[int | float],
        # Current approach: change the thresholds manually here.
        # TODO: Use a path to a json or yaml file with the expected thresholds
        default={
            "MRR": lambda x: x >= 0.37,
            "MR": lambda x: x < 600, #TODO: This one we have to figure out, it should be decently lower
            "HITS@1": lambda x: x >= 0.25,
            "HITS@3": lambda x: x >= 0.44,
            "HITS@10": lambda x: x >= 0.55,
        },
        help="Minimum performance threshold lambdas for embeddings to be considered valid",
    )
 
@pytest.fixture
def validation_thresholds(request) -> ValidatorDict[int | float]:
    """Fixture to get the minimum performance threshold."""
    return request.config.getoption("--thresholds")

@pytest.fixture
def embeddings_path(request) -> str:
    """Fixture to provide the embeddings path."""
    path = request.config.getoption("--embeddings_path")
    if not path:
        # Default to a standard model path if none specified
        pytest.skip("Embeddings path not provided")

    return path


@pytest.fixture
def dataset_path(request) -> str:
    """Fixture to provide the dataset path."""
    path = request.config.getoption("--dataset_path")
    if not path:
        # Default to a standard model path if none specified
        pytest.skip("Dataset path not provided")

    return path


@pytest.fixture
def entity_embeddings(embeddings_path) -> np.ndarray:
    """Fixture to load entity embeddings."""
    entity_emb_path = os.path.join(embeddings_path, "entity_embedding.npy")
    if not os.path.exists(entity_emb_path):
        pytest.skip(f"Entity embeddings not found at {entity_emb_path}")

    # Ensure we get a numpy array
    loaded_entities = np.load(entity_emb_path)
    assert isinstance(
        loaded_entities, np.ndarray
    ), f"Entity embeddings should be a numpy array, got {type(loaded_entities)}"

    return loaded_entities


@pytest.fixture
def relation_embeddings(embeddings_path) -> np.ndarray:
    """Fixture to load relation embeddings."""
    relation_emb_path = os.path.join(embeddings_path, "relation_embedding.npy")
    if not os.path.exists(relation_emb_path):
        pytest.skip(f"Relation embeddings not found at {relation_emb_path}")

    # Ensure we get a numpy array
    loaded_relations = np.load(relation_emb_path)
    assert isinstance(
        loaded_relations, np.ndarray
    ), f"Relation embeddings should be a numpy array, got {type(loaded_relations)}"

    return loaded_relations

@pytest.fixture
def train_triples(
    request, entityRFD2id: dict[int, str], relationRFD2id: dict[int, str]
) -> TripleIds:
    """
    Fixture to get the dataset name for training.
    Note:
        the rfds are already translated to embedding dimension indices
    """
    test_dataset_path = os.path.join(
        request.config.getoption("--dataset_path"), "train.txt"
    )
    _train_triples = read_triple(test_dataset_path, entityRFD2id, relationRFD2id)

    return _train_triples


@pytest.fixture
def valid_triples(
    request, entityRFD2id: dict[int, str], relationRFD2id: dict[int, str]
) -> TripleIds:
    """
    Fixture to get the dataset name for validation.
    Note:
        the rfds are already translated to embedding dimension indices
    """
    test_dataset_path = os.path.join(
        request.config.getoption("--dataset_path"), "valid.txt"
    )
    _valid_triples = read_triple(test_dataset_path, entityRFD2id, relationRFD2id)

    return _valid_triples


@pytest.fixture
def test_triples(
    request, entityRFD2id: dict[int, str], relationRFD2id: dict[int, str]
) -> TripleIds:
    """
    Fixture toget the dataset name for testing.
    Note:
        the rfds are already translated to embedding dimension indices
    """
    test_dataset_path = os.path.join(
        request.config.getoption("--dataset_path"), "test.txt"
    )
    _test_triples = read_triple(test_dataset_path, entityRFD2id, relationRFD2id)

    return _test_triples

@pytest.fixture
def test_percent(request) -> float:
    """Fixture to get the percentage of the test set to use."""
    return request.config.getoption("--test_percent")

@pytest.fixture
def all_true_triples(
    train_triples: TripleIds, valid_triples: TripleIds, test_triples: TripleIds
) -> TripleIds:
    return train_triples + valid_triples + test_triples


@pytest.fixture
def model_name(request) -> str:
    """Fixture to get the model name for testing."""
    return request.config.getoption("--model_name")


@pytest.fixture
def num_of_entities(entity_embeddings) -> int:
    """Get number of entities."""
    return entity_embeddings.shape[0]


@pytest.fixture
def num_of_relations(relation_embeddings) -> int:
    """Get number of entities."""
    return relation_embeddings.shape[0]


@pytest.fixture
def model_config(embeddings_path) -> dict[str, Any]:
    """Get model config."""
    model_config_path = os.path.join(embeddings_path, "config.json")
    if not os.path.exists(model_config_path):
        pytest.skip(f"Config not found at {model_config_path}")

    return json.load(open(model_config_path, "r"))


# -------------------- More Complex Fixture -------------------- #


@pytest.fixture
def knowledge_graph(
    dataset_path: str,
    model_name: str,
    entity_embeddings: np.ndarray,
    relation_embeddings: np.ndarray,
    model_config: dict,
    embeddings_path: str,
) -> KGEModel:
    """Load knowledge graph for testing."""
    try:
        # Adjust path to data based on dataset name
        train_path = os.path.join(dataset_path, "train.txt")
        valid_path = os.path.join(dataset_path, "valid.txt")
        test_path = os.path.join(dataset_path, "test.txt")

        # Check if files exist
        if not all(os.path.exists(p) for p in [train_path, valid_path, test_path]):
            pytest.skip(
                f"Dataset files (train, valid, test) not found for {dataset_path}"
            )

        checkpoint = torch.load(os.path.join(embeddings_path , "checkpoint"))

        # Create knowledge graph
        kg = KGEModel.from_pretrained(
            model_name=model_name,
            entity_embedding=entity_embeddings,
            relation_embedding=relation_embeddings,
            gamma=model_config["gamma"],
            state_dict=checkpoint["model_state_dict"]
        )
        return kg
    except Exception as e:
        pytest.skip(f"Failed to load knowledge graph: {str(e)}")


@pytest.fixture
def entityRFD2id(request) -> dict[str, int]:
    dataset_path = request.config.getoption("--dataset_path")
    id2entity_path = os.path.join(dataset_path, "entities.dict")

    if not os.path.exists(id2entity_path):
        pytest.skip(f"Entities dictionary not found at {id2entity_path}")

    return_dict: dict[str, int] = {}
    with open(id2entity_path, "r") as file:
        for line in file:
            int_idx, rfd = line.strip().split("\t")
            assert rfd not in return_dict, "Breaks @ottersome's assumptions"
            return_dict[rfd] = int(int_idx)
    return return_dict


@pytest.fixture
def relationRFD2id(request) -> dict[int, str]:
    dataset_path = request.config.getoption("--dataset_path")
    id2relation_path = os.path.join(dataset_path, "relations.dict")

    if not os.path.exists(id2relation_path):
        pytest.skip(f"Relations dictionary not found at {id2relation_path}")

    return_dict: dict[str, int] = {}
    with open(id2relation_path, "r") as file:
        for line in file:
            int_idx, rfd = line.strip().split("\t")
            assert rfd not in return_dict, "Breaks @ottersome's assumptions"
            return_dict[rfd] = int(int_idx)
    return return_dict
