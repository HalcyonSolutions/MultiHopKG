import numpy as np
import os


def test_embeddings_exist(embeddings_path: str):
    """Test that embedding files exist."""
    entity_emb_path = os.path.join(embeddings_path, "entity_embedding.npy")
    relation_emb_path = os.path.join(embeddings_path, "relation_embedding.npy")
    
    assert os.path.exists(entity_emb_path), f"Entity embeddings not found at {entity_emb_path}"
    assert os.path.exists(relation_emb_path), f"Relation embeddings not found at {relation_emb_path}"

def test_embeddings_dimensions(entity_embeddings, relation_embeddings, model_config):
    """Test that embeddings have correct dimensions."""
    # Get expected dimensions from model_config if available
    embedding_dim = model_config.get("entity_dim", None)
    
    # Test entity embeddings
    assert entity_embeddings.ndim == 2, "Entity embeddings should be 2-dimensional"
    if embedding_dim:
        assert entity_embeddings.shape[1] == embedding_dim, f"Entity embeddings dimension should be {embedding_dim}"
    
    # Test relation embeddings
    assert relation_embeddings.ndim == 2, "Relation embeddings should be 2-dimensional"
    if embedding_dim:
        assert relation_embeddings.shape[1] == embedding_dim, f"Relation embeddings dimension should be {embedding_dim}"

def test_embeddings_not_empty(entity_embeddings, relation_embeddings):
    """Test that embeddings are not empty."""
    assert entity_embeddings.size > 0, "Entity embeddings are empty"
    assert relation_embeddings.size > 0, "Relation embeddings are empty"

def test_embeddings_not_all_zeros(entity_embeddings, relation_embeddings):
    """Test that embeddings are not all zeros."""
    assert not np.allclose(entity_embeddings, 0), "Entity embeddings are all zeros"
    assert not np.allclose(relation_embeddings, 0), "Relation embeddings are all zeros"

def test_embeddings_finite(entity_embeddings, relation_embeddings):
    """Test that embeddings have finite values (no NaN or inf)."""
    assert np.isfinite(entity_embeddings).all(), "Entity embeddings contain NaN or infinite values"
    assert np.isfinite(relation_embeddings).all(), "Relation embeddings contain NaN or infinite values"

def test_embeddings_normalized(entity_embeddings, relation_embeddings, model_config):
    """Test that embeddings are normalized if that's expected."""
    # Check if model type requires normalization (like TransE often does)
    model_type = model_config.get("model", "").lower()
    requires_normalization = model_type in ["rotate", "complex"]
    
    if requires_normalization:
        # Check entity embeddings normalization
        entity_norms = np.linalg.norm(entity_embeddings, axis=1)
        is_normalized = np.allclose(entity_norms, 1.0, atol=1e-5)
        assert is_normalized, "Entity embeddings should be normalized for this model type"
        
        # Some models like ComplEx only normalize entity embeddings
        if model_type not in ["complex"]:
            relation_norms = np.linalg.norm(relation_embeddings, axis=1)
            is_normalized = np.allclose(relation_norms, 1.0, atol=1e-5)
            assert is_normalized, "Relation embeddings should be normalized for this model type"

def test_model_config_consistency(entity_embeddings, relation_embeddings, model_config, embeddings_path):
    """Test that the model_config is consistent with embeddings."""
    # Check if entity count matches
    if "num_entities" in model_config:
        assert entity_embeddings.shape[0] == model_config["num_entities"], \
            f"Entity count mismatch: {entity_embeddings.shape[0]} vs {model_config['num_entities']}"
    
    # Check if relation count matches
    if "num_relations" in model_config:
        assert relation_embeddings.shape[0] == model_config["num_relations"], \
            f"Relation count mismatch: {relation_embeddings.shape[0]} vs {model_config['num_relations']}"
    
    # Check embedding dimension consistency
    if "entity_dim" in model_config:
        assert entity_embeddings.shape[1] == model_config["entity_dim"], \
            f"Entity dimension mismatch: {entity_embeddings.shape[1]} vs {model_config['entity_dim']}"
        
    if "relation_dim" in model_config:
        assert relation_embeddings.shape[1] == model_config["relation_dim"], \
            f"Relation dimension mismatch: {relation_embeddings.shape[1]} vs {model_config['relation_dim']}"
