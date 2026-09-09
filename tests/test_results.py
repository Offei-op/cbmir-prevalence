import json
from pathlib import Path
import pandas as pd
import pytest
from cbmir.io import QUERIES
from cbmir.analysis import validate_results, run


def fixture():
    config = json.loads((Path(__file__).parents[1] / "configs/experiment.json").read_text())
    config.update(queries_per_pathology=3, database_seeds=[0, 1], bootstrap_samples=20)
    rows = []
    for mi, m in enumerate(config["models"]):
        for target in QUERIES:
            for p in config["prevalences"]:
                for seed in config["database_seeds"]:
                    for q in range(3):
                        for k in config["ks"]:
                            value = (mi + q + seed) / 20
                            rows.append(
                                dict(
                                    model=m,
                                    pathology=target,
                                    prevalence=p,
                                    database_seed=seed,
                                    query_id=q,
                                    query_path=f"{target}/{q}",
                                    K=k,
                                    precision=value,
                                    ndcg=value,
                                )
                            )
    return config, pd.DataFrame(rows)


def test_missing_model_query_is_not_silently_dropped():
    config, df = fixture()
    with pytest.raises(ValueError):
        validate_results(df.iloc[1:], config)


def test_query_identity_drift_rejected():
    config, df = fixture()
    df.loc[0, "query_path"] = "wrong"
    with pytest.raises(ValueError):
        validate_results(df, config)


def test_analysis_pipeline(tmp_path):
    config, df = fixture()
    config["artifacts_dir"] = str(tmp_path)
    source = tmp_path / "fixture.csv"
    df.to_csv(source, index=False)
    run(config, source)
    expected = [
        "headline.csv",
        "model_comparisons.csv",
        "prevalence_changes.csv",
        "precision_at10.pdf",
        "ndcg_at10.png",
        "analysis_notes.json",
    ]
    assert all((tmp_path / "analysis" / p).exists() for p in expected)
    contrasts = pd.read_csv(tmp_path / "analysis/model_comparisons.csv")
    assert len(contrasts) == 4 * 4 * 6
    assert contrasts.p_holm_global.between(0, 1).all()
