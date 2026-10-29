import importlib.util
from pathlib import Path


def test_space_overlay_preserves_streamlit_and_runtime_config(tmp_path):
    script = Path(__file__).resolve().parents[1] / "deploy/huggingface/deploy.py"
    spec = importlib.util.spec_from_file_location("space_deploy", script)
    deploy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deploy)
    (tmp_path / "README.md").write_text("# Notes Chat\n")
    (tmp_path / "deploy").mkdir()
    (tmp_path / ".env.example").write_text("OPENROUTER_API_KEY=example\n")
    deploy.apply_huggingface_overlay(tmp_path)
    assert not (tmp_path / "deploy").exists()
    readme = (tmp_path / "README.md").read_text()
    assert "sdk: docker" in readme
    assert "app_port: 8501" in readme
    dockerfile = (tmp_path / "Dockerfile").read_text()
    assert "download.pytorch.org/whl/cpu" in dockerfile
    assert '"streamlit", "run", "app.py"' in dockerfile
    assert not (tmp_path / ".env").exists()
    assert (tmp_path / ".env.example").is_file()
