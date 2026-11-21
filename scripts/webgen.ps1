param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('Build','Check','Pilot','Start','Resume','Status','Stop')]
    [string]$Action,
    [string]$InputPath,
    [string]$OutputPath,
    [string]$ContainerName = 'nir-webgen-qwen',
    [string]$Image = 'nir-webgen-generator:1.19.4'
)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not $InputPath) { $InputPath = Join-Path $repo '../data WebGen-Bench/test.jsonl' }
if (-not $OutputPath) { $OutputPath = Join-Path $repo 'data/generated-apps/webgen-qwen-001' }
$InputPath = [IO.Path]::GetFullPath($InputPath)
$OutputPath = [IO.Path]::GetFullPath($OutputPath)
$dataRoot = [IO.Path]::GetFullPath((Join-Path $repo 'data'))
if (-not $OutputPath.StartsWith($dataRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) {
    throw 'OutputPath must be under the ignored project data directory.'
}
if ($InputPath.Contains(',') -or $OutputPath.Contains(',')) { throw 'Bind paths cannot contain commas.' }
$label = 'org.nir-decomp-judge.purpose=webgen-generation'

function Invoke-DockerCommand([string[]]$Arguments) {
    & docker @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Docker command failed (exit $LASTEXITCODE)." }
}

if ($Action -eq 'Build') {
    $dirty = & git -C $repo status --porcelain
    if ($LASTEXITCODE -ne 0 -or $dirty) { throw 'Commit project changes before building an experiment image.' }
    $commit = & git -C $repo rev-parse HEAD
    Invoke-DockerCommand @('build','--file',(Join-Path $repo 'docker/webgen.Dockerfile'),
        '--build-arg',"SOURCE_COMMIT=$commit",'--tag',$Image,$repo)
    exit
}

$owned = & docker ps -a --filter "name=^/$ContainerName`$" --filter "label=$label" --format '{{.Names}}'
if ($LASTEXITCODE -ne 0) { throw 'Docker engine is unavailable. Open Docker Desktop first.' }
if ($Action -eq 'Status') {
    if ($owned) {
        Invoke-DockerCommand @('ps','-a','--filter',"name=^/$ContainerName`$",'--format','{{.Names}}: {{.Status}}')
        Invoke-DockerCommand @('logs','--tail','12',$ContainerName)
    }
    $status = Join-Path $OutputPath 'status.json'
    if (Test-Path -LiteralPath $status) { Get-Content -LiteralPath $status }
    Write-Output "Results: $OutputPath"
    exit
}
if ($Action -eq 'Stop') {
    if ($owned) { Invoke-DockerCommand @('stop','--time','20',$ContainerName) }
    exit
}
if ($owned) {
    $running = & docker inspect --format '{{.State.Running}}' $ContainerName
    if ($running -eq 'true') { throw 'The batch is already running. Use Status or Stop.' }
    New-Item -ItemType Directory -Path $OutputPath -Force | Out-Null
    $logDirectory = Join-Path $dataRoot 'generated-apps/controller-logs'
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    $log = Join-Path $logDirectory ('controller-'+(Get-Date -Format 'yyyyMMdd-HHmmss')+'.log')
    & docker logs $ContainerName 2>&1 | Set-Content -LiteralPath $log -Encoding utf8
    Invoke-DockerCommand @('rm',$ContainerName)
}
if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) { throw 'Dataset not found.' }
New-Item -ItemType Directory -Path $OutputPath -Force | Out-Null
$arguments = @('run','--name',$ContainerName,'--init','--label',$label,
    '--cpus','2','--memory','4g','--pids-limit','256',
    '--cap-drop','ALL','--security-opt','no-new-privileges',
    '--read-only','--tmpfs','/tmp:rw,size=2g',
    '--tmpfs','/home/generator:rw,uid=1000,gid=1000,size=512m',
    '--mount',"type=bind,source=$InputPath,target=/input/test.jsonl,readonly",
    '--mount',"type=bind,source=$OutputPath,target=/output",
    '--env','npm_config_cache=/output/.cache/npm',
    '--env','UV_CACHE_DIR=/output/.cache/uv','--env','PIP_CACHE_DIR=/output/.cache/pip')
if ($Action -ne 'Check') {
    if ([string]::IsNullOrWhiteSpace($env:LITELLM_API_KEY)) { throw 'Set LITELLM_API_KEY in the current process.' }
    $arguments += @('--env','LITELLM_API_KEY')
}
if ($Action -in @('Start','Resume')) { $arguments += '--detach' }
$arguments += @($Image,'--input','/input/test.jsonl','--output','/output')
if ($Action -ne 'Check') { $arguments += '--execute' }
if (Test-Path -LiteralPath (Join-Path $OutputPath 'manifest.json')) { $arguments += '--resume' }
if ($Action -eq 'Pilot') { $arguments += @('--limit','1') }
Invoke-DockerCommand $arguments
Write-Output "Results: $OutputPath"

