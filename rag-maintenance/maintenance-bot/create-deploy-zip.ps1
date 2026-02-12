$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$tmpPath = [System.IO.Path]::GetTempPath()
$zipPath = [System.IO.Path]::Combine($tmpPath, "bot-deploy2.zip")
$stagingDir = [System.IO.Path]::Combine($tmpPath, "bot-staging")

if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
if (Test-Path $stagingDir) { Remove-Item $stagingDir -Recurse -Force }

Write-Host "Creating deployment zip at $zipPath ..."

New-Item $stagingDir -ItemType Directory -Force | Out-Null

# 必要ファイルのみコピー
Copy-Item "package.json" $stagingDir
Copy-Item "package-lock.json" $stagingDir
Copy-Item "lib" "$stagingDir\lib" -Recurse

# node_modules (除外対象を省く)
robocopy "node_modules" "$stagingDir\node_modules" /E /NFL /NDL /NJH /NJS /nc /ns /np /XD ".bin" "ts-node" "typescript" | Out-Null

Compress-Archive -Path "$stagingDir\*" -DestinationPath $zipPath -Force
Remove-Item $stagingDir -Recurse -Force

$sizeMB = [math]::Round((Get-Item $zipPath).Length / 1MB, 2)
Write-Host "Done: $zipPath ($sizeMB MB)"
