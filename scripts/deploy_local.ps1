# =============================================================================
# Egg Quality - Push y deploy desde Windows (un solo comando)
# -----------------------------------------------------------------------------
# Uso (PowerShell):
#   powershell -ExecutionPolicy Bypass -File .\scripts\deploy_local.ps1 ^
#       -Pem C:\rutas\llave.pem [-Zip .\egg_quality_models.zip]
#
# Se conecta a la EC2 con IP elástica 54.227.194.211, sube el zip y la carpeta
# api/, y ejecuta el deploy remoto (deploy_remote.sh) en la VM.
# =============================================================================
param(
    [Parameter(Mandatory = $true)][string]$Pem,
    [string]$Zip = ".\egg_quality_models.zip",
    [string]$IP = "54.227.194.211",
    [string]$ModelFile = "mejor_cnn.keras"
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path $Pem)) { throw "No existe la llave: $Pem" }
if (-not (Test-Path $Zip)) { throw "No existe el zip: $Zip" }

$zipName = Split-Path -Leaf $Zip

Write-Host "==> 1/4 Subir zip ($zipName) a $IP"
scp -i $Pem $Zip "ubuntu@${IP}:/home/ubuntu/egg_quality_models.zip"

Write-Host "==> 2/4 Subir carpeta api/ a $IP"
ssh -i $Pem "ubuntu@${IP}" "mkdir -p /home/ubuntu/damaged_eggs_CNN"
scp -i $Pem -r "$Repo\api" "ubuntu@${IP}:/home/ubuntu/damaged_eggs_CNN/"

Write-Host "==> 3/4 Subir script de deploy y ejecutarlo"
scp -i $Pem "$Repo\scripts\deploy_remote.sh" "ubuntu@${IP}:/home/ubuntu/"
ssh -i $Pem "ubuntu@${IP}" "chmod +x /home/ubuntu/deploy_remote.sh && bash /home/ubuntu/deploy_remote.sh /home/ubuntu/egg_quality_models.zip $ModelFile"

Write-Host "==> 4/4 Verificar desde la IP pública"
try {
    $health = Invoke-RestMethod -Uri "http://${IP}:8000/health" -TimeoutSec 15
    $health | ConvertTo-Json
} catch {
    Write-Warning "El health local funcionó pero el público no responde: $($_.Exception.Message)"
    Write-Warning "Revisa en AWS el Security Group: Custom TCP 8000 abierto a 0.0.0.0/0."
}