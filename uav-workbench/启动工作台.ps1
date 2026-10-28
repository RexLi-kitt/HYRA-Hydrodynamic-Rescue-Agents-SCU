param([int]$Port = 8767)
$ErrorActionPreference = 'Stop'
$uavRoot = $PSScriptRoot
$uavCandidates = @(
    (Join-Path $uavRoot '.venv\Scripts\python.exe'),
    'python',
    (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe')
)
foreach ($uavPython in $uavCandidates) {
    if (-not (Get-Command $uavPython -ErrorAction SilentlyContinue)) { continue }
    $uavHasNumpy = & $uavPython -c 'import importlib.util; print(int(importlib.util.find_spec("numpy") is not None))'
    if ($uavHasNumpy -eq '1') {
        Push-Location $uavRoot
        try { & $uavPython console_app.py --port $Port } finally { Pop-Location }
        exit $LASTEXITCODE
    }
}
throw '没有找到含NumPy的Python环境。请先在虚拟环境中安装本目录requirements.txt，再运行python console_app.py。'
