param(
    [Parameter(Mandatory = $true)]
    [string]$Competition,

    [string]$DataDir = "data"
)

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$tokenPath = Join-Path $env:USERPROFILE ".kaggle\kaggle.json"
if (-not (Test-Path $tokenPath)) {
    throw "Missing Kaggle token: $tokenPath"
}

$token = Get-Content $tokenPath -Raw | ConvertFrom-Json
if (-not $token.username -or -not $token.key) {
    throw "Kaggle token must contain username and key fields."
}

$pair = "{0}:{1}" -f $token.username.Trim(), $token.key.Trim()
$auth = [Convert]::ToBase64String([System.Text.Encoding]::ASCII.GetBytes($pair))
$headers = @{
    Authorization = "Basic $auth"
    "User-Agent" = "Kaggle/1.6.17"
}

function Invoke-KaggleRequest {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Uri,

        [string]$OutFile
    )

    for ($attempt = 1; $attempt -le 8; $attempt++) {
        try {
            if ($OutFile) {
                Invoke-WebRequest -Uri $Uri -Headers $headers -OutFile $OutFile
            } else {
                Invoke-RestMethod -Uri $Uri -Headers $headers
            }
            return
        } catch {
            $statusCode = $null
            if ($_.Exception.Response) {
                $statusCode = [int]$_.Exception.Response.StatusCode
            }

            if ($statusCode -eq 429 -and $attempt -lt 8) {
                $seconds = [Math]::Min(120, 10 * $attempt)
                Write-Host "Rate limited. Waiting $seconds seconds before retry $($attempt + 1)."
                Start-Sleep -Seconds $seconds
                continue
            }

            throw
        }
    }
}

New-Item -ItemType Directory -Force -Path $DataDir | Out-Null

$cachePath = Join-Path $DataDir "kaggle_files.json"
if (Test-Path $cachePath) {
    Write-Host "Using cached file list $cachePath"
    $files = Get-Content $cachePath -Raw | ConvertFrom-Json
} else {
    $files = @()
    $pageToken = $null
    $page = 1
    do {
        $listUrl = "https://www.kaggle.com/api/v1/competitions/data/list/${Competition}?pageSize=100"
        if ($pageToken) {
            $listUrl = "${listUrl}&pageToken=$([uri]::EscapeDataString($pageToken))"
        }

        Write-Host "Fetching file list page $page"
        $response = Invoke-KaggleRequest -Uri $listUrl
        if ($response.files) {
            $files += $response.files
        } elseif ($response -is [array]) {
            $files += $response
        }

        $pageToken = $response.nextPageTokenNullable
        $page += 1
        Start-Sleep -Milliseconds 500
    } while ($pageToken)

    $files | ConvertTo-Json -Depth 5 | Set-Content -Path $cachePath -Encoding UTF8
}

foreach ($file in $files) {
    $name = $file.name
    if (-not $name) {
        continue
    }

    $outPath = Join-Path $DataDir $name
    $outDir = Split-Path $outPath -Parent
    if ($outDir) {
        New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    }
    if ((Test-Path $outPath) -and ((Get-Item $outPath).Length -eq $file.totalBytes)) {
        Write-Host "Skipping existing $name"
        continue
    }

    $encodedName = [uri]::EscapeDataString($name)
    $downloadUrl = "https://www.kaggle.com/api/v1/competitions/data/download/${Competition}/$encodedName"
    Write-Host "Downloading $name"
    Invoke-KaggleRequest -Uri $downloadUrl -OutFile $outPath
    Start-Sleep -Milliseconds 200
}

Write-Host "Downloaded $($files.Count) files to $DataDir"
