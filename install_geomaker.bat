@echo off
REM install_geomaker.bat
REM Script de instalação automática para Geomaker v2.0 (Windows)

echo ======================================
echo GEOMAKER v2.0 - Script de Instalação
echo ======================================
echo.

REM Verificar Python
echo 🐍 Verificando versão do Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python não encontrado. Por favor, instale Python 3.9 ou superior.
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
echo    Python %python_version% detectado

REM Criar ambiente virtual
echo.
echo 📦 Criando ambiente virtual...
if exist geomaker_env (
    echo    Ambiente virtual já existe. Pulando...
) else (
    python -m venv geomaker_env
    echo    ✓ Ambiente virtual criado
)

REM Ativar ambiente virtual
echo 🔄 Ativando ambiente virtual...
call geomaker_env\Scripts\activate.bat

REM Atualizar pip
echo 📈 Atualizando pip...
python -m pip install --upgrade pip setuptools wheel

REM Instalar PyTorch
echo.
echo 🔥 Instalando PyTorch...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo    CUDA não detectado. Instalando PyTorch para CPU...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
) else (
    echo    CUDA detectado! Instalando PyTorch com suporte CUDA...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
)

REM Instalar dependências
echo.
echo 📚 Instalando dependências...
pip install -r requirements.txt

REM Verificar instalação
echo.
echo 🔍 Verificando instalação...
python check_installation.py

echo.
echo ✅ Instalação concluída com sucesso!
echo.
echo Para usar o Geomaker:
echo   1. Ative o ambiente: geomaker_env\Scripts\activate.bat
echo   2. Execute o app: streamlit run app.py
echo.
echo Para app avançado com recursos Qualis A1:
echo   streamlit run app4.py
echo.
pause
