import pytest
import pandas as pd
from cbmir.io import index_hash


def test_index_order_is_part_of_identity():
    a = pd.DataFrame({"image_path": ["a", "b"], "embedding_id": [0, 1]})
    b = pd.DataFrame({"image_path": ["b", "a"], "embedding_id": [0, 1]})
    assert index_hash(a) != index_hash(b)


def test_index_rejects_nonconsecutive_ids():
    with pytest.raises(ValueError):
        index_hash(pd.DataFrame({"image_path": ["a", "b"], "embedding_id": [1, 0]}))


def test_exact_retrieval_ties():
    torch = pytest.importorskip("torch")
    from cbmir.retrieval import retrieve_topk

    q = torch.tensor([[1.0, 0.0]])
    db = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    assert retrieve_topk(q, db, 3).tolist() == [[1, 2, 0]]


def test_supcon_finite_gradient():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from cbmir.learning import weighted_supcon

    features = torch.randn(8, 16, requires_grad=True)
    labels = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    loss = weighted_supcon(features, labels)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 0
