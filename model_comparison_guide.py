"""
Script de Demonstração: Comparação de Modelos
Demonstra análise de eficiência para múltiplos modelos (ResNet18, ResNet50, DenseNet121)
"""

import streamlit as st
import torch
from performance_analyzer import PerformanceAnalyzer

def create_comparison_section():
    """
    Cria seção de comparação entre múltiplos modelos
    """
    st.write("---")
    st.write("## 🔬 Comparação entre Modelos")
    st.write("Análise comparativa científica entre diferentes arquiteturas")
    
    # Explicação
    with st.expander("ℹ️ Sobre a Comparação de Modelos"):
        st.write("""
        ### Por que comparar modelos?
        
        A escolha do modelo adequado depende de múltiplos fatores:
        
        1. **Acurácia**: Qualidade das predições
        2. **Velocidade**: Tempo de inferência
        3. **Memória**: Recursos computacionais necessários
        4. **Complexidade**: Número de parâmetros
        
        ### Modelos Disponíveis:
        
        - **ResNet18**: Mais leve e rápido (11M parâmetros)
        - **ResNet50**: Balanceado (25M parâmetros)
        - **DenseNet121**: Mais profundo e preciso (8M parâmetros)
        
        ### Métricas de Comparação:
        
        - **Trade-off Acurácia/Velocidade**: Modelos mais complexos geralmente são mais lentos
        - **Trade-off Acurácia/Memória**: Modelos maiores precisam de mais recursos
        - **Score de Eficiência**: Métrica composta que balanceia todos os fatores
        """)
    
    # Tabela comparativa teórica
    st.write("### 📊 Comparação Teórica")
    
    comparison_data = {
        'Modelo': ['ResNet18', 'ResNet50', 'DenseNet121'],
        'Parâmetros': ['11.7M', '25.6M', '8.0M'],
        'Camadas': ['18', '50', '121'],
        'Velocidade Esperada': ['⚡⚡⚡ Rápido', '⚡⚡ Médio', '⚡ Lento'],
        'Acurácia Esperada': ['⭐⭐ Boa', '⭐⭐⭐ Ótima', '⭐⭐⭐ Ótima'],
        'Uso de Memória': ['💾 Baixo', '💾💾 Médio', '💾 Baixo'],
    }
    
    import pandas as pd
    df_comparison = pd.DataFrame(comparison_data)
    st.table(df_comparison)
    
    # Guia de seleção
    st.write("### 🎯 Guia de Seleção")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **Use ResNet18 quando:**
        - Prioridade é velocidade
        - Recursos limitados
        - Aplicação em tempo real
        - Dataset pequeno/médio
        """)
    
    with col2:
        st.success("""
        **Use ResNet50 quando:**
        - Balanceamento de qualidade e velocidade
        - Recursos moderados disponíveis
        - Maior acurácia necessária
        - Dataset médio/grande
        """)
    
    with col3:
        st.warning("""
        **Use DenseNet121 quando:**
        - Máxima acurácia é prioridade
        - Eficiência de parâmetros importante
        - Pode esperar mais tempo
        - Dataset grande e complexo
        """)
    
    # Recomendações baseadas em cenários
    st.write("### 💡 Recomendações por Caso de Uso")
    
    scenarios = {
        '🏥 Diagnóstico Médico': {
            'modelo': 'DenseNet121 ou ResNet50',
            'razão': 'Acurácia é crítica, tempo de inferência menos relevante'
        },
        '📱 Aplicativo Mobile': {
            'modelo': 'ResNet18',
            'razão': 'Recursos limitados, necessita ser rápido e leve'
        },
        '🏭 Controle de Qualidade Industrial': {
            'modelo': 'ResNet50',
            'razão': 'Balanceamento entre acurácia e velocidade para linha de produção'
        },
        '🎓 Pesquisa Acadêmica': {
            'modelo': 'DenseNet121',
            'razão': 'Maximizar métricas para publicação Qualis A1'
        },
        '☁️ Cloud/API': {
            'modelo': 'ResNet50',
            'razão': 'Bom balanceamento com escalabilidade'
        }
    }
    
    for scenario, info in scenarios.items():
        with st.expander(f"{scenario}"):
            st.write(f"**Modelo Recomendado:** {info['modelo']}")
            st.write(f"**Justificativa:** {info['razão']}")
    
    # Exemplo de análise comparativa
    st.write("### 📈 Exemplo de Análise Comparativa")
    st.write("Após treinar modelos, você verá gráficos como:")
    
    st.code("""
    # Exemplo de resultado de comparação:
    
    Modelo          Acurácia    Tempo (ms)    Memória (MB)    Score
    ─────────────────────────────────────────────────────────────────
    ResNet18        0.925       15.2          45.2            0.852
    ResNet50        0.948       32.7          98.5            0.831
    DenseNet121     0.952       45.8          32.1            0.798
    
    Análise:
    - ResNet50 tem melhor balanceamento geral
    - ResNet18 é o mais rápido
    - DenseNet121 tem melhor acurácia mas é mais lento
    """, language='text')
    
    # Dicas para otimização
    st.write("### ⚙️ Dicas de Otimização")
    
    tips = [
        "Use **Fine-Tuning** apenas quando necessário (aumenta tempo mas melhora acurácia)",
        "Ajuste o **batch size** conforme memória disponível (maior = mais rápido)",
        "Considere **quantização** para deployment em produção",
        "Use **early stopping** para evitar treinar desnecessariamente",
        "Teste com **dados aumentados** para melhorar generalização"
    ]
    
    for i, tip in enumerate(tips, 1):
        st.write(f"{i}. {tip}")

def create_metrics_explanation():
    """
    Cria seção explicativa sobre as métricas
    """
    st.write("---")
    st.write("## 📚 Entendendo as Métricas")
    
    with st.expander("🎯 Métricas de Classificação"):
        st.write("""
        ### Acurácia (Accuracy)
        Percentual de predições corretas sobre o total.
        
        `Acurácia = (TP + TN) / (TP + TN + FP + FN)`
        
        **Quando usar**: Bom para datasets balanceados
        **Limitação**: Pode ser enganosa em datasets desbalanceados
        
        ---
        
        ### Precisão (Precision)
        Das amostras preditas como positivas, quantas realmente são?
        
        `Precisão = TP / (TP + FP)`
        
        **Quando usar**: Quando custo de falso positivo é alto
        **Exemplo**: Spam detection (não queremos marcar emails legítimos como spam)
        
        ---
        
        ### Recall (Sensibilidade)
        Das amostras realmente positivas, quantas foram identificadas?
        
        `Recall = TP / (TP + FN)`
        
        **Quando usar**: Quando custo de falso negativo é alto
        **Exemplo**: Diagnóstico médico (não queremos perder casos de doença)
        
        ---
        
        ### F1-Score
        Média harmônica entre Precisão e Recall.
        
        `F1 = 2 * (Precisão * Recall) / (Precisão + Recall)`
        
        **Quando usar**: Quando precisão e recall são igualmente importantes
        **Vantagem**: Balanceia ambas as métricas
        
        ---
        
        ### AUC-ROC
        Área sob a curva ROC (Receiver Operating Characteristic).
        
        **Interpretação**:
        - 0.5: Classificador aleatório
        - 0.7-0.8: Razoável
        - 0.8-0.9: Bom
        - > 0.9: Excelente
        
        **Vantagem**: Independente do threshold escolhido
        """)
    
    with st.expander("⚡ Métricas de Eficiência"):
        st.write("""
        ### Tempo de Inferência
        Tempo necessário para processar uma amostra.
        
        **Medido em**: Milissegundos (ms)
        **Objetivo**: Menor é melhor
        
        **Benchmarks**:
        - < 10ms: Excelente (tempo real)
        - 10-50ms: Bom (aplicações interativas)
        - 50-200ms: Aceitável (batch processing)
        - > 200ms: Lento (otimização recomendada)
        
        ---
        
        ### Throughput
        Número de amostras processadas por segundo.
        
        **Medido em**: Amostras/segundo
        **Objetivo**: Maior é melhor
        
        **Cálculo**: `Throughput = 1 / Tempo_Inferência`
        
        ---
        
        ### Uso de Memória
        Recursos de memória necessários.
        
        **Componentes**:
        - **Modelo**: Parâmetros e buffers
        - **Sistema**: RAM total usada
        - **GPU**: VRAM quando disponível
        
        **Otimização**:
        - Quantização (reduz precisão)
        - Pruning (remove pesos pequenos)
        - Knowledge distillation (modelo menor)
        """)
    
    with st.expander("🏆 Score de Eficiência"):
        st.write("""
        ### Score Composto
        Métrica única que combina múltiplos aspectos:
        
        `Score = 0.5 * Acurácia + 0.3 * Eficiência_Tempo + 0.2 * Eficiência_Memória`
        
        ### Pesos Justificados:
        - **50% Acurácia**: Principal objetivo do modelo
        - **30% Tempo**: Importante para experiência do usuário
        - **20% Memória**: Relevante para deployment
        
        ### Interpretação:
        - **≥ 0.80**: 🥇 Excelente - Publicável em Qualis A1
        - **0.60-0.79**: 🥈 Bom - Aceitável para maioria das aplicações
        - **< 0.60**: 🥉 Requer otimização
        
        ### Ajustando Pesos:
        Você pode modificar os pesos conforme sua aplicação:
        - Tempo real: Aumentar peso do tempo
        - Edge computing: Aumentar peso da memória
        - Pesquisa: Focar apenas em acurácia
        """)

if __name__ == "__main__":
    st.set_page_config(page_title="Comparação de Modelos", page_icon="🔬", layout="wide")
    
    st.title("🔬 Guia de Comparação e Análise de Modelos")
    st.write("Documentação completa para análise científica de qualidade Qualis A1")
    
    create_comparison_section()
    create_metrics_explanation()
    
    st.write("---")
    st.success("""
    💡 **Dica**: Para uma análise completa, treine múltiplos modelos com os mesmos dados
    e compare os resultados usando as métricas apresentadas neste guia.
    """)
