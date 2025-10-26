import json
from pathlib import Path

from systems.close_precision_ensemble_system import ClosePrecisionEnsembleTrainer


def test_ensemble_trainer_runs_with_lightgbm_only(tmp_path):
    output_dir = tmp_path / "models"
    analysis_dir = tmp_path / "analysis"

    trainer = ClosePrecisionEnsembleTrainer(
        n_splits=2,
        test_size=10,
        evaluation_window=60,
        base_model_names=("lightgbm",),
        max_training_rows=20000,
        output_dir=output_dir,
        analysis_dir=analysis_dir,
    )

    metrics = trainer.train_and_evaluate()

    assert metrics["ensemble_cv"]["ensemble_oof_precision"] >= 0.0
    assert metrics["ensemble_holdout"]["ensemble_precision"] >= 0.0
    assert f"ensemble_top{trainer.top_n}_precision" in metrics["ensemble_holdout"]

    artifact_path = output_dir / "latest_ensemble_model.joblib"
    assert artifact_path.exists()

    metrics_path = analysis_dir / "ensemble_precision_metrics.json"
    assert metrics_path.exists()
    summary = json.loads(metrics_path.read_text())
    assert "holdout_metrics" in summary
    assert "ensemble" in summary["holdout_metrics"]
    assert summary.get("top_n") == trainer.top_n
