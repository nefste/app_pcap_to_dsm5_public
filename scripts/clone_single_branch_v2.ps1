Param(
    [string]$Source = "C:\Users\StephanNef\Documents\GitHub\app_pcap_to_dsm5",
    [string]$Destination = "C:\Users\StephanNef\Documents\GitHub\app_pcap_to_dsm5_v2",
    [string]$Branch = ""
)

function Fail($msg) { Write-Error $msg; exit 1 }

# Check git availability
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Fail "git is not available in PATH. Please install Git for Windows and retry."
}

if (-not (Test-Path -LiteralPath $Source)) {
    Fail "Source path not found: $Source"
}

if (Test-Path -LiteralPath $Destination) {
    Fail "Destination already exists: $Destination"
}

# Determine source branch
if ([string]::IsNullOrEmpty($Branch)) {
    $current = (git -C "$Source" rev-parse --abbrev-ref HEAD 2>$null)
    if ([string]::IsNullOrWhiteSpace($current)) { $current = "main" }
    $hasMain = (git -C "$Source" show-ref --verify --quiet refs/heads/main; $?)
    $hasMaster = (git -C "$Source" show-ref --verify --quiet refs/heads/master; $?)
    if ($hasMain) { $Branch = "main" }
    elseif ($hasMaster) { $Branch = "master" }
    else { $Branch = $current }
}

Write-Host "Cloning single branch '$Branch' from" -NoNewline; Write-Host " `"$Source`"" -ForegroundColor Cyan
Write-Host "→ to `"$Destination`" as new repo (no remote)"

# Create destination parent if needed
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null

# Clone only the chosen branch
git clone --no-local --single-branch --branch "$Branch" "$Source" "$Destination" || Fail "git clone failed"

# Switch to destination and normalize to main
if ($Branch -ne "main") {
    git -C "$Destination" branch -m "$Branch" "main" || Fail "Failed to rename branch to main"
}

# Remove remote so the repo is clean
git -C "$Destination" remote remove origin 2>$null | Out-Null

# Unset upstream config if any
git -C "$Destination" config --unset branch.main.remote 2>$null | Out-Null
git -C "$Destination" config --unset branch.main.merge 2>$null | Out-Null

# Verify
$branches = git -C "$Destination" branch --list
Write-Host "Branches in new repo:" -ForegroundColor Green
Write-Host $branches
if (-not ($branches -match "\bmain\b")) {
    Fail "Expected 'main' branch in destination, but it was not found."
}

Write-Host "Done. New repo created at: $Destination" -ForegroundColor Green
Write-Host "You can now add a remote and push, e.g.:"
Write-Host "  git -C `"$Destination`" remote add origin https://github.com/<user>/app_pcap_to_dsm5_v2.git"
Write-Host "  git -C `"$Destination`" push -u origin main"

