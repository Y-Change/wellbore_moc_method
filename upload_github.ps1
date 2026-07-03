# 将 wellbore_moc_method 推送到 GitHub 独立仓库
# 用法（在本目录 PowerShell 中）:
#   1. gh auth login          # 首次需登录 GitHub
#   2. .\upload_github.ps1    # 创建仓库并推送

$ErrorActionPreference = "Stop"
$RepoName = "wellbore_moc_method"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here

# 确保 gh 可用
$gh = Get-Command gh -ErrorAction SilentlyContinue
if (-not $gh) {
    Write-Error "未找到 gh CLI，请先安装: winget install GitHub.cli"
}

gh auth status 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "请先登录 GitHub: gh auth login"
    exit 1
}

# 若已有 origin 则直接推送
$remotes = git remote 2>$null
if ($remotes -match "^origin$") {
    Write-Host "远程 origin 已存在，执行 push..."
    git push -u origin main
    gh repo view --web
    exit 0
}

Write-Host "创建 GitHub 仓库: $RepoName"
gh repo create $RepoName `
    --public `
    --source=. `
    --remote=origin `
    --description "自研轻量 MOC 井筒-裂缝水击仿真器及验证脚本" `
    --push

Write-Host "完成。仓库地址:"
gh repo view --json url -q .url
