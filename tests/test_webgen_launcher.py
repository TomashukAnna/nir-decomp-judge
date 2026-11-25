"""Exercise the PowerShell launcher with Docker replaced by a command recorder."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

POWERSHELL = shutil.which("pwsh")
pytestmark = pytest.mark.skipif(POWERSHELL is None, reason="PowerShell is not installed")
LAUNCHER = Path(__file__).resolve().parents[1] / "scripts/webgen.ps1"


def launch(
    tmp_path: Path,
    count: int | None,
    *,
    owned: bool = True,
    running: bool = False,
    key: bool = True,
    other_container: bool = False,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]], Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "input.jsonl"
    source.write_text('{"id":"000002","instruction":"Build an app"}\n')
    log = tmp_path / "docker.jsonl"
    driver = tmp_path / "driver.ps1"
    driver.write_text(
        """param([string]$Launcher,[string]$Workspace,[string]$InputFile,[string]$Count)
$ErrorActionPreference = 'Stop'
function global:docker {
    $items = @($args | ForEach-Object { [string]$_ })
    ConvertTo-Json -InputObject $items -Compress | Add-Content -LiteralPath $env:MOCK_DOCKER_LOG
    $global:LASTEXITCODE = 0
    if ($items[0] -eq 'ps') {
        $labelQuery = [bool]($items | Where-Object { $_ -like 'label=*' })
        $nameQuery = [bool]($items | Where-Object { $_ -like 'name=*' })
        if (-not $labelQuery -or $env:MOCK_OWNED -eq 'true') { 'nir-webgen-qwen' }
        if (-not $nameQuery -and $env:MOCK_OTHER -eq 'true') { 'another-generator' }
    } elseif ($items[0] -eq 'image') {
        'sha256:verified-image'
    } elseif ($items[0] -eq 'inspect') {
        $env:MOCK_RUNNING
    } elseif ($items[0] -eq 'logs') {
        'preserved old controller log'
    } elseif ($items[0] -eq 'run') {
        'new-container-id'
    }
}
$parameters = @{ Action='Start'; WorkspaceRoot=$Workspace; InputPath=$InputFile }
if ($Count -ne 'missing') { $parameters.Count = [int]$Count }
& $Launcher @parameters
""",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env.update(
        MOCK_DOCKER_LOG=str(log),
        MOCK_OWNED=str(owned).lower(),
        MOCK_RUNNING=str(running).lower(),
        MOCK_OTHER=str(other_container).lower(),
        LITELLM_API_KEY="fake-test-key" if key else "",
    )
    assert POWERSHELL is not None
    result = subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-File",
            str(driver),
            "-Launcher",
            str(LAUNCHER),
            "-Workspace",
            str(workspace),
            "-InputFile",
            str(source),
            "-Count",
            str(count) if count is not None else "missing",
        ],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return result, calls, workspace


@pytest.mark.parametrize("count", [1, 5, 19])
def test_launch_uses_requested_limit_and_one_name(tmp_path: Path, count: int) -> None:
    result, calls, workspace = launch(tmp_path, count)
    assert result.returncode == 0, result.stderr
    runs = [row for row in calls if row[0] == "run"]
    assert len(runs) == 1
    run = runs[0]
    assert run[run.index("--limit") + 1] == str(count)
    assert run[run.index("--name") + 1] == "nir-webgen-qwen"
    assert "--detach" in run
    assert "sha256:verified-image" in run
    assert "fake-test-key" not in json.dumps(calls)
    operations = [row[0] for row in calls]
    assert operations.index("logs") < operations.index("rm") < operations.index("run")
    saved = list((workspace / "data/generated-apps/controller-logs").glob("*.log"))
    assert len(saved) == 1
    assert "preserved old controller log" in saved[0].read_text()


@pytest.mark.parametrize("count", [None, 0, -1])
def test_missing_or_invalid_count_does_not_touch_docker(tmp_path: Path, count: int | None) -> None:
    result, calls, _ = launch(tmp_path, count)
    assert result.returncode != 0
    assert calls == []


@pytest.mark.parametrize(
    "options",
    [{"running": True}, {"key": False}, {"owned": False}, {"other_container": True}],
)
def test_existing_container_is_retained_when_launch_is_rejected(
    tmp_path: Path, options: dict[str, bool]
) -> None:
    result, calls, _ = launch(tmp_path, 1, **options)
    assert result.returncode != 0
    assert not any(row[0] in {"rm", "run"} for row in calls)
