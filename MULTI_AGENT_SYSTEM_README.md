# Sistema Multi-Agente para Análise Diagnóstica

## Visão Geral

Este módulo implementa um sistema robusto com **15 agentes especializados + 1 agente gerente** para análise diagnóstica avançada. O sistema coordena múltiplas perspectivas especializadas para fornecer análises mais confiáveis e abrangentes.

## Arquitetura do Sistema

### Agente Gerente (Manager Agent)
- **Função**: Coordena todos os 15 agentes especializados
- **Responsabilidades**:
  - Distribuir tarefas para os agentes
  - Coletar e agregar respostas
  - Calcular confiança ponderada
  - Sintetizar relatório integrado
  - Priorizar análises por relevância

### 15 Agentes Especializados

#### 1. **MorphologyAgent** (Prioridade: 4)
- **Especialidade**: Análise de Estrutura e Forma
- **Áreas de Expertise**: morfologia, geometria, contornos, dimensões
- **Função**: Analisa características estruturais e morfológicas

#### 2. **TextureAgent** (Prioridade: 4)
- **Especialidade**: Análise de Textura e Padrões
- **Áreas de Expertise**: textura, granularidade, rugosidade, padrões
- **Função**: Examina propriedades texturais e padrões

#### 3. **ColorAnalysisAgent** (Prioridade: 3)
- **Especialidade**: Análise de Cor e Tonalidade
- **Áreas de Expertise**: cor, tonalidade, saturação, luminosidade
- **Função**: Avalia distribuição cromática

#### 4. **SpatialAgent** (Prioridade: 3)
- **Especialidade**: Análise de Distribuição Espacial
- **Áreas de Expertise**: distribuição, localização, arranjo, topologia
- **Função**: Analisa organização e arranjo espacial

#### 5. **StatisticalAgent** (Prioridade: 5)
- **Especialidade**: Análise Estatística e Métricas
- **Áreas de Expertise**: estatística, probabilidade, métricas, distribuições
- **Função**: Calcula e interpreta parâmetros estatísticos

#### 6. **DifferentialDiagnosisAgent** (Prioridade: 5)
- **Especialidade**: Diagnóstico Diferencial e Alternativas
- **Áreas de Expertise**: diferencial, alternativas, exclusão, comparação
- **Função**: Identifica e avalia diagnósticos alternativos

#### 7. **QualityAssuranceAgent** (Prioridade: 4)
- **Especialidade**: Controle de Qualidade e Validação
- **Áreas de Expertise**: qualidade, validação, verificação, confiabilidade
- **Função**: Avalia qualidade e confiabilidade dos resultados

#### 8. **ContextualAgent** (Prioridade: 3)
- **Especialidade**: Análise de Contexto e Ambiente
- **Áreas de Expertise**: contexto, ambiente, situação, condições
- **Função**: Considera fatores contextuais e ambientais

#### 9. **LiteratureAgent** (Prioridade: 3)
- **Especialidade**: Revisão de Literatura e Evidências
- **Áreas de Expertise**: literatura, evidências, estudos, publicações
- **Função**: Busca suporte na literatura científica

#### 10. **MethodologyAgent** (Prioridade: 4)
- **Especialidade**: Avaliação Metodológica
- **Áreas de Expertise**: metodologia, procedimentos, protocolos, técnicas
- **Função**: Avalia aderência a metodologias estabelecidas

#### 11. **RiskAssessmentAgent** (Prioridade: 5)
- **Especialidade**: Avaliação de Risco e Incertezas
- **Áreas de Expertise**: risco, incerteza, probabilidade, confiabilidade
- **Função**: Quantifica e avalia riscos associados

#### 12. **ComparativeAgent** (Prioridade: 3)
- **Especialidade**: Análise Comparativa e Benchmarking
- **Áreas de Expertise**: comparação, benchmark, padrões, referências
- **Função**: Compara com padrões e casos de referência

#### 13. **ClinicalRelevanceAgent** (Prioridade: 5)
- **Especialidade**: Relevância Clínica e Aplicabilidade
- **Áreas de Expertise**: clínica, prática, aplicabilidade, utilidade
- **Função**: Avalia implicações clínicas/práticas

#### 14. **IntegrationAgent** (Prioridade: 4)
- **Especialidade**: Integração Multi-modal de Dados
- **Áreas de Expertise**: integração, fusão, multi-modal, síntese
- **Função**: Integra múltiplas fontes de informação

#### 15. **ValidationAgent** (Prioridade: 5)
- **Especialidade**: Validação Cruzada e Verificação
- **Áreas de Expertise**: validação, verificação, confirmação, checagem
- **Função**: Valida resultados através de múltiplos métodos

## Sistema de Prioridades

Os agentes são classificados em 5 níveis de prioridade (1-5):

- **Prioridade 5 (Crítica)**: Agentes cujas análises são essenciais
  - StatisticalAgent
  - DifferentialDiagnosisAgent
  - RiskAssessmentAgent
  - ClinicalRelevanceAgent
  - ValidationAgent

- **Prioridade 4 (Alta)**: Agentes com análises muito importantes
  - MorphologyAgent
  - TextureAgent
  - QualityAssuranceAgent
  - MethodologyAgent
  - IntegrationAgent

- **Prioridade 3 (Média)**: Agentes com análises complementares importantes
  - ColorAnalysisAgent
  - SpatialAgent
  - ContextualAgent
  - LiteratureAgent
  - ComparativeAgent

## Fluxo de Trabalho

```
1. Usuário solicita análise
   ↓
2. Gerente recebe a solicitação
   ↓
3. Gerente distribui para 15 agentes especializados
   ↓
4. Cada agente realiza análise independente
   ↓
5. Gerente coleta todas as respostas
   ↓
6. Gerente calcula confiança agregada ponderada
   ↓
7. Gerente ordena por prioridade
   ↓
8. Gerente sintetiza relatório integrado
   ↓
9. Sistema retorna relatório completo ao usuário
```

## Métricas de Agregação

### Confiança Ponderada
```python
confiança_agregada = Σ(confiança_agente × prioridade_agente) / Σ(prioridade_agente)
```

### Consenso
- **Alto Consenso**: >80% dos agentes com confiança >0.9
- **Consenso Moderado**: 50-80% dos agentes com confiança >0.9
- **Baixo Consenso**: <50% dos agentes com confiança >0.9

## Uso

### Exemplo Básico

```python
from multi_agent_system import ManagerAgent

# Inicializar gerente
manager = ManagerAgent()

# Preparar contexto
context = {
    'training_stats': {...},
    'statistical_results': {...},
    'gradcam_description': "...",
}

# Executar análise multi-agente
report = manager.coordinate_analysis(
    predicted_class="Melanoma",
    confidence=0.945,
    context=context
)

print(report)
```

### Integração em Streamlit

```python
import streamlit as st
from multi_agent_system import ManagerAgent

# Checkbox para ativar análise multi-agente
use_multiagent = st.checkbox("Ativar Sistema Multi-Agente", value=True)

if use_multiagent:
    with st.spinner("Coordenando 15 agentes especializados..."):
        manager = ManagerAgent()
        report = manager.coordinate_analysis(
            predicted_class=class_name,
            confidence=confidence,
            context=context
        )
        st.markdown(report)
        st.success("✅ Análise Multi-Agente Concluída!")
```

## Estrutura do Relatório

O relatório gerado contém as seguintes seções:

1. **Resumo Executivo**
   - Classificação e confiança
   - Confiança agregada de 15 agentes
   - Estatísticas de consenso

2. **Análises por Prioridade**
   - Análises agrupadas por nível de prioridade
   - Análise individual de cada agente
   - Recomendações específicas

3. **Consenso e Síntese**
   - Visão consolidada de todos os agentes
   - Confiança média ajustada
   - Convergência de perspectivas

4. **Recomendações Consolidadas**
   - Top 10 recomendações mais importantes
   - Sem duplicatas

5. **Conclusão Gerencial**
   - Avaliação final integrada
   - Nível de certeza (ALTO/MODERADO/BAIXO)
   - Recomendação de ação

## Benefícios do Sistema

### 1. **Robustez**
- 15 perspectivas independentes reduzem viés
- Múltiplas validações cruzadas

### 2. **Abrangência**
- Cobertura completa de aspectos relevantes
- Análise multi-dimensional

### 3. **Confiabilidade**
- Agregação ponderada de confiança
- Sistema de prioridades baseado em relevância

### 4. **Transparência**
- Análises individuais documentadas
- Rastreabilidade de recomendações

### 5. **Escalabilidade**
- Fácil adicionar novos agentes especializados
- Arquitetura modular

## Requisitos

- Python 3.8+
- numpy
- typing (built-in)
- dataclasses (built-in)

## Limitações e Considerações

1. **Performance**: 15 agentes + gerente podem aumentar tempo de processamento
2. **Consistência**: Cada agente fornece análise independente, pode haver pequenas variações
3. **Contexto**: Quanto mais contexto fornecido, melhor a análise
4. **Interpretação**: Sistema fornece análise técnica, interpretação final requer expertise humana

## Versão

- **Versão Atual**: 1.0.0
- **Data**: 2025-01-20
- **Status**: Produção

## Autor

Projeto Geomaker + IA
Professor Marcelo Claro

## Licença

Este módulo faz parte do projeto CLASSIFICACAO-DE-ROCHAS.
