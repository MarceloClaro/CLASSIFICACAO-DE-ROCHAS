"""
check_installation.py
Verifica se todas as dependências estão instaladas corretamente.
"""

import sys
from importlib import import_module

# Lista de dependências críticas
CRITICAL_PACKAGES = {
    'torch': '2.0.0',
    'torchvision': '0.15.0',
    'numpy': '1.23.0',
    'pandas': '2.0.0',
    'scipy': '1.10.0',
    'sklearn': '1.3.0',
    'matplotlib': '3.7.0',
    'seaborn': '0.12.0',
    'PIL': '10.0.0',
    'cv2': '4.8.0',
    'streamlit': '1.28.0',
}

# Lista de dependências opcionais
OPTIONAL_PACKAGES = {
    'timm': '0.9.0',
    'torchcam': '0.4.0',
    'plotly': '5.14.0',
    'umap': '0.5.0',
    'albumentations': '1.3.0',
    'transformers': '4.30.0',
    'wandb': '0.15.0',
    'openai': '1.0.0',
}

def check_version(package_name, min_version):
    """Verifica se a versão do pacote é >= min_version."""
    try:
        module = import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        if version != 'unknown':
            from packaging import version as pkg_version
            return pkg_version.parse(version) >= pkg_version.parse(min_version), version
        return False, version
    except ImportError:
        return False, None

def main():
    print("="*60)
    print("VERIFICAÇÃO DE INSTALAÇÃO - GEOMAKER v2.0")
    print("="*60)
    print(f"Python: {sys.version}")
    print("="*60)
    
    # Verificar pacotes críticos
    print("\n📦 PACOTES CRÍTICOS (obrigatórios):")
    print("-"*60)
    all_critical_ok = True
    
    for package, min_ver in CRITICAL_PACKAGES.items():
        installed, version = check_version(package, min_ver)
        status = "✅" if installed else "❌"
        version_str = f"v{version}" if version else "não instalado"
        min_ver_str = f" (mínimo: v{min_ver})" if not installed or version < min_ver else ""
        
        print(f"{status} {package:20s} {version_str}{min_ver_str}")
        
        if not installed:
            all_critical_ok = False
    
    # Verificar CUDA
    print("\n🎮 SUPORTE CUDA:")
    print("-"*60)
    try:
        import torch
        print(f"✅ PyTorch CUDA: {torch.version.cuda}")
        print(f"✅ CUDA disponível: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ Número de GPUs: {torch.cuda.device_count()}")
        else:
            print("⚠️  CUDA não disponível (será usado CPU)")
    except ImportError:
        print("❌ PyTorch não instalado")
    
    # Verificar pacotes opcionais
    print("\n📦 PACOTES OPCIONAIS:")
    print("-"*60)
    
    for package, min_ver in OPTIONAL_PACKAGES.items():
        installed, version = check_version(package, min_ver)
        status = "✅" if installed else "⚠️ "
        version_str = f"v{version}" if version else "não instalado"
        
        print(f"{status} {package:20s} {version_str}")
    
    # Resumo
    print("\n" + "="*60)
    if all_critical_ok:
        print("✅ TODAS AS DEPENDÊNCIAS CRÍTICAS ESTÃO INSTALADAS!")
        print("   O sistema está pronto para uso.")
    else:
        print("❌ FALTAM DEPENDÊNCIAS CRÍTICAS!")
        print("   Por favor, instale com: pip install -r requirements.txt")
        sys.exit(1)
    print("="*60)

if __name__ == "__main__":
    try:
        import packaging
    except ImportError:
        print("⚠️  Instalando 'packaging'...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "packaging"])
        import packaging
    
    main()
