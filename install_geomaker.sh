#!/bin/bash
# install_geomaker.sh
# Script de instalação automática para Geomaker v2.0

set -e

echo "======================================"
echo "GEOMAKER v2.0 - Script de Instalação"
echo "======================================"
echo ""

# Verificar Python
echo "🐍 Verificando versão do Python..."
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    echo "   Python $python_version detectado"
else
    echo "❌ Python 3 não encontrado. Por favor, instale Python 3.9 ou superior."
    exit 1
fi

# Criar ambiente virtual
echo ""
echo "📦 Criando ambiente virtual..."
if [ -d "geomaker_env" ]; then
    echo "   Ambiente virtual já existe. Pulando..."
else
    python3 -m venv geomaker_env
    echo "   ✓ Ambiente virtual criado"
fi

# Ativar ambiente virtual
echo "🔄 Ativando ambiente virtual..."
source geomaker_env/bin/activate

# Atualizar pip
echo "📈 Atualizando pip..."
pip install --upgrade pip setuptools wheel

# Instalar PyTorch (detectar CUDA se disponível)
echo ""
echo "🔥 Instalando PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "   CUDA detectado! Instalando PyTorch com suporte CUDA..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "   CUDA não detectado. Instalando PyTorch para CPU..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Instalar dependências
echo ""
echo "📚 Instalando dependências..."
pip install -r requirements.txt

# Verificar instalação
echo ""
echo "🔍 Verificando instalação..."
python check_installation.py

echo ""
echo "✅ Instalação concluída com sucesso!"
echo ""
echo "Para usar o Geomaker:"
echo "  1. Ative o ambiente: source geomaker_env/bin/activate"
echo "  2. Execute o app: streamlit run app.py"
echo ""
echo "Para app avançado com recursos Qualis A1:"
echo "  streamlit run app4.py"
echo ""
