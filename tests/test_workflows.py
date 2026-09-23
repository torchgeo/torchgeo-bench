"""Workflow coverage for independently reviewable stacked pull requests."""

from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


@pytest.mark.parametrize("name", ["ci", "docs"])
def test_checks_cover_every_pull_request_base(name: str) -> None:
    # BaseLoader keeps GitHub's "on" key instead of treating it as a YAML 1.1 boolean.
    config = yaml.load((WORKFLOWS / f"{name}.yaml").read_text(), Loader=yaml.BaseLoader)
    events = config["on"]
    assert "pull_request" in events
    selection = events["pull_request"] or {}
    assert "branches" not in selection
    assert "branches-ignore" not in selection
    assert events["push"]["branches"] == ["main"]


def test_pull_requests_cannot_deploy_documentation() -> None:
    config = yaml.load((WORKFLOWS / "docs.yaml").read_text(), Loader=yaml.BaseLoader)
    not_pull_request = "github.event_name != 'pull_request'"
    assert config["jobs"]["deploy"]["if"] == not_pull_request
    upload = next(
        step
        for step in config["jobs"]["build"]["steps"]
        if step.get("name") == "Upload Pages artifact"
    )
    assert upload["if"] == not_pull_request


def test_ranking_explorer_workflow_is_scoped_to_results_csv_pushes() -> None:
    config = yaml.load((WORKFLOWS / "ranking-explorer.yaml").read_text(), Loader=yaml.BaseLoader)
    events = config["on"]
    assert "pull_request" not in events
    assert "workflow_dispatch" in events
    assert events["push"]["branches"] == ["main"]
    assert events["push"]["paths"] == ["results/models/**/*.csv", "results/compute_cost.csv"]

    refresh = config["jobs"]["regenerate"]["steps"][-1]
    assert refresh["uses"] == "peter-evans/create-pull-request@v7"
    assert refresh["with"]["add-paths"] == "docs/_static/ranking_explorer.html"
    assert refresh["with"]["commit-message"] == "chore(docs): refresh ranking explorer"
    assert refresh["with"]["title"] == "chore(docs): refresh ranking explorer"
