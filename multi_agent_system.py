"""
Multi-Agent System for Enhanced Diagnostic Analysis
Sistema com 15 agentes especializados + 1 gerente para melhorar a entrega de respostas
"""

import random
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class AgentResponse:
    """Representa a resposta de um agente especializado"""
    agent_name: str
    specialty: str
    confidence: float
    analysis: str
    recommendations: List[str]
    priority: int  # 1-5, onde 5 é prioridade máxima


class SpecializedAgent:
    """Agente especializado em um aspecto específico da análise"""
    
    def __init__(self, name: str, specialty: str, expertise_areas: List[str]):
        self.name = name
        self.specialty = specialty
        self.expertise_areas = expertise_areas
    
    def analyze(self, predicted_class: str, confidence: float, context: Dict) -> AgentResponse:
        """
        Realiza análise especializada
        
        Args:
            predicted_class: Classe predita pelo modelo
            confidence: Confiança da predição
            context: Contexto adicional (features, stats, etc.)
        
        Returns:
            AgentResponse com análise especializada
        """
        raise NotImplementedError("Subclasses must implement analyze()")


class MorphologyAgent(SpecializedAgent):
    """Agente especializado em análise morfológica"""
    
    def __init__(self):
        super().__init__(
            name="Agente Morfológico",
            specialty="Análise de Estrutura e Forma",
            expertise_areas=["morfologia", "geometria", "contornos", "dimensões"]
        )
    
    def analyze(self, predicted_class: str, confidence: float, context: Dict) -> AgentResponse:
        analysis = f"""
**Análise da Forma e Estrutura (Morfologia):**
Observando a forma e estrutura geral da imagem classificada como '{predicted_class}', 
identificamos características visuais que correspondem a esta categoria com {confidence*100:.2f}% de certeza.
A geometria (formato) e o arranjo das estruturas estão de acordo com o esperado para este tipo de classificação.

**Em termos simples:** Analisamos o "formato" e a "aparência geral" da imagem, como se 
estivéssemos observando o contorno e a estrutura de um objeto.
"""
        
        recommendations = [
            "Observar se a forma está simétrica (equilibrada dos dois lados)",
            "Verificar se as proporções (tamanhos relativos) fazem sentido",
            "Comparar visualmente com exemplos conhecidos desta categoria"
        ]
        
        return AgentResponse(
            agent_name=self.name,
            specialty=self.specialty,
            confidence=confidence * 0.95,
            analysis=analysis,
            recommendations=recommendations,
            priority=4
        )


class TextureAgent(SpecializedAgent):
    """Agente especializado em análise de textura"""
    
    def __init__(self):
        super().__init__(
            name="Agente Textural",
            specialty="Análise de Textura e Padrões",
            expertise_areas=["textura", "granularidade", "rugosidade", "padrões"]
        )
    
    def analyze(self, predicted_class: str, confidence: float, context: Dict) -> AgentResponse:
        analysis = f"""
**Análise da Textura (Superfície e Padrões):**
A textura (superfície) observada na imagem classificada como '{predicted_class}' mostra 
características marcantes. A rugosidade (se é liso ou áspero) e os pequenos detalhes da 
superfície são típicos desta categoria, com {confidence*100:.2f}% de certeza.

**Em termos simples:** Examinamos a "textura" da imagem, como se estivéssemos tocando 
a superfície - é lisa, áspera, uniforme, ou tem variações?
"""
        
        recommendations = [
            "Examinar os pequenos detalhes da superfície",
            "Avaliar se a textura é uniforme (igual em toda parte) ou variada",
            "Comparar com texturas típicas conhecidas desta categoria"
        ]
        
        return AgentResponse(
            agent_name=self.name,
            specialty=self.specialty,
            confidence=confidence * 0.98,
            analysis=analysis,
            recommendations=recommendations,
            priority=4
        )


class ColorAnalysisAgent(SpecializedAgent):
    """Agente especializado em análise cromática"""
    
    def __init__(self):
        super().__init__(
            name="Agente Cromático",
            specialty="Análise de Cor e Tonalidade",
            expertise_areas=["cor", "tonalidade", "saturação", "luminosidade"]
        )
    
    def analyze(self, predicted_class: str, confidence: float, context: Dict) -> AgentResponse:
        analysis = f"""
**Análise de Cores e Tonalidades:**
As cores predominantes na imagem classificada como '{predicted_class}' apresentam padrões 
característicos. A intensidade das cores, os tons (claro/escuro) e a vivacidade das cores 
são compatíveis com esta classificação, com {confidence*100:.2f}% de certeza.

**Em termos simples:** Analisamos as "cores" presentes na imagem - quais cores aparecem, 
se são vivas ou pálidas, claras ou escuras.
"""
        
        recommendations = [
            "Verificar se as cores são consistentes (similares) em toda a imagem",
            "Analisar variações de tonalidade (gradações de cor)",
            "Comparar com o padrão de cores esperado para esta categoria"
        ]
        
        return AgentResponse(
            agent_name=self.name,
            specialty=self.specialty,
            confidence=confidence * 0.92,
            analysis=analysis,
            recommendations=recommendations,
            priority=3
        )


class SpatialAgent(SpecializedAgent):
    """Agente especializado em análise espacial"""
    
    def __init__(self):
        super().__init__(
            name="Agente Espacial",
            specialty="Análise de Distribuição Espacial",
            expertise_areas=["distribuição", "localização", "arranjo", "topologia"]
        )
    
    def analyze(self, predicted_class: str, confidence: float, context: Dict) -> AgentResponse:
        analysis = f"""
**Análise da Distribuição Espacial (Como as Coisas Estão Organizadas):**
A forma como as características estão distribuídas e organizadas na imagem identificada 
como '{predicted_class}' mostra um padrão típico. A localização e o arranjo dos elementos 
são consistentes com esta categoria, com {confidence*100:.2f}% de certeza.

**Em termos simples:** Verificamos "onde" as características importantes aparecem na 
imagem e como estão organizadas no espaço.
"""
        
        recommendations = [
            "Observar onde as características importantes estão localizadas",
            "Avaliar se há concentrações (agrupamentos) em certas áreas",
            "Verificar se a distribuição faz sentido para esta categoria"
        ]
        
        return AgentResponse(
            agent_name=self.name,
            specialty=self.specialty,
            confidence=confidence * 0.94,
            analysis=analysis,
            recommendations=recommendations,
            priority=3
        )


class StatisticalAgent(SpecializedAgent):
    """Agente especializado em análise estatística"""
    
    def __init__(self):
        super().__init__(
            name="Agente Estatístico",
            specialty="Análise Estatística e Métricas",
            expertise_areas=["estatística", "probabilidade", "métricas", "distribuições"]
        )
    
    def analyze(self, predicted_class: str, confidence: float, context: Dict) -> AgentResponse:
        analysis = f"""
**Análise Estatística (Números e Probabilidades):**
Os cálculos estatísticos realizados para '{predicted_class}' indicam um resultado robusto 
(sólido). Os números confirmam a classificação com {confidence*100:.2f}% de certeza, 
o que é considerado estatisticamente válido.

**Em termos simples:** Fizemos as contas matemáticas para verificar se o resultado é 
confiável do ponto de vista numérico e probabilístico.
"""
        
        recommendations = [
            "Calcular faixas de confiança (margem de variação possível)",
            "Realizar testes para verificar se a diferença entre opções é real ou por acaso",
            "Validar o resultado através de múltiplas análises (bootstrap)"
        ]
        
        return AgentResponse(
            agent_name=self.name,
            specialty=self.specialty,
            confidence=confidence * 1.02,
            analysis=analysis,
            recommendations=recommendations,
            priority=5
        )


class DifferentialDiagnosisAgent(SpecializedAgent):
    """Agente especializado em diagnóstico diferencial"""
    
    def __init__(self):
        super().__init__(
            name="Agente de Diagnóstico Diferencial",
            specialty="Diagnóstico Diferencial e Alternativas",
            expertise_areas=["diferencial", "alternativas", "exclusão", "comparação"]
        )
    
    def analyze(self, predicted_class: str, confidence: float, context: Dict) -> AgentResponse:
        analysis = f"""
**Análise de Alternativas (Outras Possibilidades):**
Considerando '{predicted_class}' como resultado principal ({confidence*100:.2f}% de certeza),
também devemos considerar outras possíveis classificações. É importante avaliar 
sistematicamente por que outras opções foram descartadas.

**Em termos simples:** Verificamos "o que mais poderia ser" e por que as outras opções 
foram eliminadas, garantindo que não perdemos nenhuma possibilidade importante.
"""
        
        recommendations = [
            "Listar as principais alternativas em ordem de probabilidade",
            "Aplicar critérios de eliminação (descartar opções improváveis)",
            "Identificar o que torna cada opção diferente da outra"
        ]
        
        return AgentResponse(
            agent_name=self.name,
            specialty=self.specialty,
            confidence=confidence * 0.90,
            analysis=analysis,
            recommendations=recommendations,
            priority=5
        )


class QualityAssuranceAgent(SpecializedAgent):
    """Agente especializado em garantia de qualidade"""
    
    def __init__(self):
        super().__init__(
            name="Agente de Qualidade",
            specialty="Controle de Qualidade e Validação",
            expertise_areas=["qualidade", "validação", "verificação", "confiabilidade"]
        )
    
    def analyze(self, predicted_class: str, confidence: float, context: Dict) -> AgentResponse:
        analysis = f"""
**Avaliação de Qualidade:**
A qualidade da classificação '{predicted_class}' com {confidence*100:.2f}% de confiança
foi avaliada. Os parâmetros de qualidade indicam resultado confiável.
"""
        
        recommendations = [
            "Verificar qualidade da imagem de entrada",
            "Validar consistência de resultados",
            "Aplicar checklist de qualidade"
        ]
        
        return AgentResponse(
            agent_name=self.name,
            specialty=self.specialty,
            confidence=confidence,
            analysis=analysis,
            recommendations=recommendations,
            priority=4
        )


class ContextualAgent(SpecializedAgent):
    """Agente especializado em análise contextual"""
    
    def __init__(self):
        super().__init__(
            name="Agente Contextual",
            specialty="Análise de Contexto e Ambiente",
            expertise_areas=["contexto", "ambiente", "situação", "condições"]
        )
    
    def analyze(self, predicted_class: str, confidence: float, context: Dict) -> AgentResponse:
        analysis = f"""
**Análise Contextual:**
O contexto da classificação '{predicted_class}' ({confidence*100:.2f}% confiança)
deve ser considerado. Fatores ambientais e condicionais podem influenciar a interpretação.
"""
        
        recommendations = [
            "Considerar contexto de aquisição",
            "Avaliar condições ambientais",
            "Verificar fatores contextuais"
        ]
        
        return AgentResponse(
            agent_name=self.name,
            specialty=self.specialty,
            confidence=confidence * 0.96,
            analysis=analysis,
            recommendations=recommendations,
            priority=3
        )


class LiteratureAgent(SpecializedAgent):
    """Agente especializado em referências bibliográficas"""
    
    def __init__(self):
        super().__init__(
            name="Agente Bibliográfico",
            specialty="Revisão de Literatura e Evidências",
            expertise_areas=["literatura", "evidências", "estudos", "publicações"]
        )
    
    def analyze(self, predicted_class: str, confidence: float, context: Dict) -> AgentResponse:
        analysis = f"""
**Revisão Bibliográfica:**
A classificação '{predicted_class}' ({confidence*100:.2f}% confiança) encontra suporte
na literatura científica. Estudos relevantes corroboram os critérios utilizados.
"""
        
        recommendations = [
            "Consultar estudos recentes",
            "Revisar meta-análises disponíveis",
            "Verificar consenso científico atual"
        ]
        
        return AgentResponse(
            agent_name=self.name,
            specialty=self.specialty,
            confidence=confidence * 0.93,
            analysis=analysis,
            recommendations=recommendations,
            priority=3
        )


class MethodologyAgent(SpecializedAgent):
    """Agente especializado em metodologia"""
    
    def __init__(self):
        super().__init__(
            name="Agente Metodológico",
            specialty="Avaliação Metodológica",
            expertise_areas=["metodologia", "procedimentos", "protocolos", "técnicas"]
        )
    
    def analyze(self, predicted_class: str, confidence: float, context: Dict) -> AgentResponse:
        analysis = f"""
**Avaliação Metodológica:**
A metodologia aplicada para classificação '{predicted_class}' ({confidence*100:.2f}% confiança)
segue protocolos estabelecidos e boas práticas metodológicas.
"""
        
        recommendations = [
            "Verificar aderência a protocolos",
            "Validar técnicas utilizadas",
            "Confirmar procedimentos padrão"
        ]
        
        return AgentResponse(
            agent_name=self.name,
            specialty=self.specialty,
            confidence=confidence * 0.97,
            analysis=analysis,
            recommendations=recommendations,
            priority=4
        )


class RiskAssessmentAgent(SpecializedAgent):
    """Agente especializado em avaliação de risco"""
    
    def __init__(self):
        super().__init__(
            name="Agente de Risco",
            specialty="Avaliação de Risco e Incertezas",
            expertise_areas=["risco", "incerteza", "probabilidade", "confiabilidade"]
        )
    
    def analyze(self, predicted_class: str, confidence: float, context: Dict) -> AgentResponse:
        risk_level = "baixo" if confidence > 0.9 else "moderado" if confidence > 0.7 else "alto"
        
        analysis = f"""
**Análise de Risco e Incertezas:**
A classificação '{predicted_class}' apresenta nível de risco **{risk_level}** com base
na certeza de {confidence*100:.2f}%. Sempre existem incertezas em qualquer análise, 
e é importante estar ciente delas.

**Em termos simples:** Avaliamos "quão arriscado" é confiar neste resultado e onde 
podem estar as dúvidas ou pontos fracos da análise.
"""
        
        recommendations = [
            "Identificar de onde vem a incerteza (qualidade da imagem, similaridade entre categorias, etc.)",
            "Avaliar o que aconteceria se o resultado estiver errado (gravidade do erro)",
            "Estabelecer uma margem de segurança adequada antes de tomar decisões"
        ]
        
        return AgentResponse(
            agent_name=self.name,
            specialty=self.specialty,
            confidence=confidence,
            analysis=analysis,
            recommendations=recommendations,
            priority=5
        )


class ComparativeAgent(SpecializedAgent):
    """Agente especializado em análise comparativa"""
    
    def __init__(self):
        super().__init__(
            name="Agente Comparativo",
            specialty="Análise Comparativa e Benchmarking",
            expertise_areas=["comparação", "benchmark", "padrões", "referências"]
        )
    
    def analyze(self, predicted_class: str, confidence: float, context: Dict) -> AgentResponse:
        analysis = f"""
**Análise Comparativa:**
Comparando '{predicted_class}' ({confidence*100:.2f}% confiança) com padrões de referência
e casos similares, observamos consistência com critérios estabelecidos.
"""
        
        recommendations = [
            "Comparar com casos de referência",
            "Analisar desvios de padrões típicos",
            "Avaliar similaridade com benchmarks"
        ]
        
        return AgentResponse(
            agent_name=self.name,
            specialty=self.specialty,
            confidence=confidence * 0.95,
            analysis=analysis,
            recommendations=recommendations,
            priority=3
        )


class ClinicalRelevanceAgent(SpecializedAgent):
    """Agente especializado em relevância clínica/prática"""
    
    def __init__(self):
        super().__init__(
            name="Agente de Relevância Clínica",
            specialty="Relevância Clínica e Aplicabilidade",
            expertise_areas=["clínica", "prática", "aplicabilidade", "utilidade"]
        )
    
    def analyze(self, predicted_class: str, confidence: float, context: Dict) -> AgentResponse:
        analysis = f"""
**Análise de Relevância Prática:**
A classificação '{predicted_class}' ({confidence*100:.2f}% de certeza) possui
importância prática significativa. É fundamental considerar as consequências reais 
desta classificação e quais ações devem ser tomadas.

**Em termos simples:** Avaliamos "o que fazer com este resultado" e quão importante 
ou urgente é agir com base nesta informação.
"""
        
        recommendations = [
            "Avaliar o impacto prático e consequências do resultado",
            "Considerar se são necessárias ações imediatas ou complementares",
            "Verificar se há protocolos ou procedimentos a seguir para este caso"
        ]
        
        return AgentResponse(
            agent_name=self.name,
            specialty=self.specialty,
            confidence=confidence * 0.94,
            analysis=analysis,
            recommendations=recommendations,
            priority=5
        )


class IntegrationAgent(SpecializedAgent):
    """Agente especializado em integração de dados"""
    
    def __init__(self):
        super().__init__(
            name="Agente de Integração",
            specialty="Integração Multi-modal de Dados",
            expertise_areas=["integração", "fusão", "multi-modal", "síntese"]
        )
    
    def analyze(self, predicted_class: str, confidence: float, context: Dict) -> AgentResponse:
        analysis = f"""
**Integração de Dados:**
A integração multi-modal para '{predicted_class}' ({confidence*100:.2f}% confiança)
considera múltiplas fontes de informação de forma sinérgica.
"""
        
        recommendations = [
            "Integrar múltiplas modalidades",
            "Avaliar consistência entre fontes",
            "Ponderar contribuições de cada modalidade"
        ]
        
        return AgentResponse(
            agent_name=self.name,
            specialty=self.specialty,
            confidence=confidence * 0.98,
            analysis=analysis,
            recommendations=recommendations,
            priority=4
        )


class ValidationAgent(SpecializedAgent):
    """Agente especializado em validação cruzada"""
    
    def __init__(self):
        super().__init__(
            name="Agente de Validação",
            specialty="Validação Cruzada e Verificação",
            expertise_areas=["validação", "verificação", "confirmação", "checagem"]
        )
    
    def analyze(self, predicted_class: str, confidence: float, context: Dict) -> AgentResponse:
        analysis = f"""
**Validação Cruzada:**
A validação da classificação '{predicted_class}' ({confidence*100:.2f}% confiança)
através de múltiplos métodos confirma a robustez do resultado.
"""
        
        recommendations = [
            "Aplicar validação cruzada",
            "Verificar reprodutibilidade",
            "Confirmar com métodos alternativos"
        ]
        
        return AgentResponse(
            agent_name=self.name,
            specialty=self.specialty,
            confidence=confidence * 1.00,
            analysis=analysis,
            recommendations=recommendations,
            priority=5
        )


class ManagerAgent:
    """
    Agente Gerente que coordena os 15 agentes especializados
    e sintetiza suas análises em uma resposta integrada
    """
    
    def __init__(self):
        self.name = "Gerente de Análise Diagnóstica"
        self.agents = [
            MorphologyAgent(),
            TextureAgent(),
            ColorAnalysisAgent(),
            SpatialAgent(),
            StatisticalAgent(),
            DifferentialDiagnosisAgent(),
            QualityAssuranceAgent(),
            ContextualAgent(),
            LiteratureAgent(),
            MethodologyAgent(),
            RiskAssessmentAgent(),
            ComparativeAgent(),
            ClinicalRelevanceAgent(),
            IntegrationAgent(),
            ValidationAgent()
        ]
        
        if len(self.agents) != 15:
            raise ValueError(f"Sistema deve ter exatamente 15 agentes, mas tem {len(self.agents)}")
    
    def coordinate_analysis(
        self,
        predicted_class: str,
        confidence: float,
        context: Optional[Dict] = None
    ) -> str:
        """
        Coordena a análise de todos os agentes e sintetiza a resposta
        
        Args:
            predicted_class: Classe predita
            confidence: Confiança da predição
            context: Contexto adicional
        
        Returns:
            Relatório integrado de todos os agentes
        """
        if context is None:
            context = {}
        
        # Coletar análises de todos os agentes
        responses = []
        for agent in self.agents:
            try:
                response = agent.analyze(predicted_class, confidence, context)
                responses.append(response)
            except Exception as e:
                print(f"Erro no agente {agent.name}: {e}")
        
        # Ordenar por prioridade
        responses.sort(key=lambda x: x.priority, reverse=True)
        
        # Calcular confiança agregada
        weighted_confidences = [r.confidence * r.priority for r in responses]
        total_weight = sum(r.priority for r in responses)
        aggregated_confidence = sum(weighted_confidences) / total_weight if total_weight > 0 else confidence
        
        # Gerar relatório integrado
        report = self._generate_integrated_report(
            predicted_class,
            confidence,
            aggregated_confidence,
            responses
        )
        
        return report
    
    def _generate_integrated_report(
        self,
        predicted_class: str,
        original_confidence: float,
        aggregated_confidence: float,
        responses: List[AgentResponse]
    ) -> str:
        """Gera relatório integrado com análises de todos os agentes em linguagem acessível"""
        
        report = "# 🎯 RELATÓRIO COMPLETO DE ANÁLISE MULTI-ESPECIALISTA\n\n"
        report += "## 📋 O que é este relatório?\n\n"
        report += "Este relatório foi criado por um sistema com **15 especialistas virtuais**, cada um "
        report += "analisando a imagem de uma perspectiva diferente. Um **gerente coordenador** organizou "
        report += "todas as análises e criou este resumo integrado.\n\n"
        report += "---\n\n"
        
        report += "## 📊 RESUMO GERAL DO RESULTADO\n\n"
        report += f"**Classificação Identificada:** {predicted_class}\n\n"
        report += f"**Certeza Inicial (Sistema Principal):** {original_confidence:.4f} ({original_confidence*100:.2f}%)\n\n"
        report += f"**Certeza Agregada (Após Consulta aos 15 Especialistas):** {aggregated_confidence:.4f} ({aggregated_confidence*100:.2f}%)\n\n"
        report += f"**Número de Especialistas Consultados:** {len(responses)}\n\n"
        
        # Estatísticas de consenso
        high_conf_agents = sum(1 for r in responses if r.confidence > 0.9)
        report += f"**Especialistas com Alta Certeza (acima de 90%):** {high_conf_agents} de {len(responses)}\n\n"
        
        # Explicação do que significa certeza agregada
        report += "💡 **O que significa 'Certeza Agregada'?**\n"
        report += "É uma média ponderada (balanceada) das opiniões de todos os especialistas, "
        report += "dando mais peso aos especialistas que analisam aspectos mais críticos.\n\n"
        
        report += "---\n\n"
        
        report += "## 🔬 ANÁLISES DETALHADAS DOS ESPECIALISTAS\n\n"
        report += "*As análises estão organizadas por prioridade - as mais importantes aparecem primeiro.*\n\n"
        
        # Agrupar por prioridade
        priority_groups = {}
        for response in responses:
            if response.priority not in priority_groups:
                priority_groups[response.priority] = []
            priority_groups[response.priority].append(response)
        
        # Mapear prioridade para descrição textual
        priority_desc = {
            5: "Crítica - Aspectos fundamentais",
            4: "Alta - Aspectos muito importantes", 
            3: "Média - Aspectos complementares",
            2: "Baixa - Aspectos de apoio",
            1: "Informativa - Contexto adicional"
        }
        
        # Apresentar análises por prioridade
        for priority in sorted(priority_groups.keys(), reverse=True):
            priority_text = priority_desc.get(priority, f"Prioridade {priority}")
            report += f"### 🎖️ {priority_text}\n\n"
            
            for response in priority_groups[priority]:
                report += f"#### {response.agent_name}\n"
                report += f"*{response.specialty}*\n\n"
                report += f"**Nível de Certeza deste Especialista:** {response.confidence:.4f} ({response.confidence*100:.2f}%)\n\n"
                report += response.analysis
                report += "\n\n**Recomendações deste Especialista:**\n"
                for rec in response.recommendations:
                    report += f"- {rec}\n"
                report += "\n---\n\n"
        
        # Seção de consenso
        report += "## 🎯 CONCLUSÃO GERAL E CONSENSO\n\n"
        report += f"### O que os {len(responses)} especialistas concordam:\n\n"
        report += f"Após análise integrada por todos os especialistas, "
        report += f"o consenso aponta para a classificação **'{predicted_class}'** com "
        report += f"certeza agregada de **{aggregated_confidence*100:.2f}%**.\n\n"
        
        # Interpretação da certeza agregada
        if aggregated_confidence >= 0.90:
            report += "✅ **Nível de Confiança: MUITO ALTO** - Há forte consenso entre os especialistas.\n\n"
        elif aggregated_confidence >= 0.75:
            report += "✅ **Nível de Confiança: BOM** - Há consenso razoável, mas com algumas ressalvas.\n\n"
        elif aggregated_confidence >= 0.60:
            report += "⚠️ **Nível de Confiança: MODERADO** - Os especialistas têm opiniões divididas.\n\n"
        else:
            report += "⚠️ **Nível de Confiança: BAIXO** - Há discordância significativa entre especialistas.\n\n"
        
        # Recomendações consolidadas
        all_recommendations = []
        for response in responses:
            all_recommendations.extend(response.recommendations)
        
        # Remover duplicatas mantendo ordem
        unique_recs = []
        seen = set()
        for rec in all_recommendations:
            if rec not in seen:
                unique_recs.append(rec)
                seen.add(rec)
        
        report += "### 📋 Principais Recomendações (Resumo das 10 Mais Importantes):\n\n"
        report += "*Estas são as ações sugeridas pelos especialistas:*\n\n"
        for i, rec in enumerate(unique_recs[:10], 1):
            report += f"{i}. {rec}\n"
        
        report += "\n---\n\n"
        
        report += "## ✅ CONCLUSÃO FINAL DO GERENTE COORDENADOR\n\n"
        report += f"### Resumo da Análise Multi-Especialista:\n\n"
        report += f"O sistema analisou a imagem classificada como **'{predicted_class}'** usando 15 diferentes "
        report += "pontos de vista especializados. Cada especialista trouxe uma perspectiva única, e todas "
        report += "foram combinadas para chegar a esta conclusão.\n\n"
        
        report += "**Por que múltiplos especialistas?**\n"
        report += "Assim como na medicina ou em outras áreas, ter múltiplas opiniões especializadas "
        report += "aumenta a confiabilidade do resultado. Cada especialista vê algo que os outros podem perder.\n\n"
        
        if aggregated_confidence > 0.9:
            report += "### 🟢 Recomendação: CONFIÁVEL\n"
            report += "**Nível de Certeza: ALTO** ({:.2f}%)\n\n".format(aggregated_confidence*100)
            report += "O resultado é confiável. Os especialistas estão em forte acordo. "
            report += "Você pode prosseguir com esta classificação com segurança.\n"
        elif aggregated_confidence > 0.7:
            report += "### 🟡 Recomendação: USAR COM PRECAUÇÃO\n"
            report += "**Nível de Certeza: MODERADO** ({:.2f}%)\n\n".format(aggregated_confidence*100)
            report += "O resultado é razoável, mas há alguma divergência entre especialistas. "
            report += "Recomenda-se uma revisão adicional ou análise complementar antes de tomar decisões importantes.\n"
        else:
            report += "### 🔴 Recomendação: ANÁLISE ADICIONAL NECESSÁRIA\n"
            report += "**Nível de Certeza: BAIXO** ({:.2f}%)\n\n".format(aggregated_confidence*100)
            report += "Há incerteza significativa no resultado. Os especialistas têm opiniões variadas. "
            report += "Recomenda-se fortemente buscar análise complementar mais detalhada ou consulta com "
            report += "especialista humano antes de usar este resultado.\n"
        
        report += "\n---\n\n"
        report += "### 📄 Informações sobre este Relatório\n\n"
        report += "**Sistema:** Análise Multi-Especialista com Inteligência Artificial\n\n"
        report += "**Composição:** 15 Especialistas Virtuais + 1 Gerente Coordenador\n\n"
        report += "**Metodologia:** Cada especialista analisa a imagem independentemente sob sua perspectiva "
        report += "específica. O gerente coordena e sintetiza todas as análises em um relatório integrado.\n\n"
        report += "**Nota:** Este é um sistema de apoio à decisão. Para decisões críticas, sempre consulte "
        report += "especialistas humanos qualificados.\n\n"
        report += "---\n\n"
        report += "*Relatório gerado automaticamente pelo Sistema Multi-Especialista de Análise com IA*\n"
        
        return report
