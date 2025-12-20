"""
AI Chat Module for Deep Diagnostic Analysis
Supports Gemini and Groq APIs for comprehensive image analysis
"""

import os
import requests
from typing import Dict, List, Optional, Tuple
import base64
from io import BytesIO
import numpy as np

try:
    import google.genai as genai
    GEMINI_AVAILABLE = True
    GEMINI_NEW_API = True  # New google-genai package
except ImportError:
    # Fallback to old package if new one not available
    try:
        import google.generativeai as genai
        GEMINI_AVAILABLE = True
        GEMINI_NEW_API = False  # Old google-generativeai package
    except ImportError:
        GEMINI_AVAILABLE = False
        GEMINI_NEW_API = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class AIAnalyzer:
    """
    AI-powered diagnostic analyzer that uses LLMs to provide
    PhD-level analysis of image classification results
    """
    
    def __init__(self, api_provider: str, api_key: str, model_name: str):
        """
        Initialize the AI analyzer
        
        Args:
            api_provider: 'gemini' or 'groq'
            api_key: API key for the selected provider
            model_name: Model name (e.g., 'gemini-1.0-pro', 'mixtral-8x7b-32768')
        """
        self.api_provider = api_provider.lower()
        self.api_key = api_key
        self.model_name = model_name
        
        if self.api_provider == 'gemini' and GEMINI_AVAILABLE:
            if GEMINI_NEW_API:
                # New google-genai package API
                self.client = genai.Client(api_key=api_key)
                self.model = None  # Will be set when generating content
            else:
                # Old google-generativeai package API
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(model_name)
        elif self.api_provider == 'groq' and GROQ_AVAILABLE:
            self.client = Groq(api_key=api_key)
        else:
            raise ValueError(f"Provider {api_provider} not available or not supported")
    
    def generate_diagnostic_prompt(
        self,
        predicted_class: str,
        confidence: float,
        training_stats: Dict,
        statistical_results: Dict,
        gradcam_analysis: str = "",
        academic_references: List[Dict] = None
    ) -> str:
        """
        Generate a comprehensive prompt for PhD-level diagnostic analysis
        
        Args:
            predicted_class: The predicted class name
            confidence: Confidence level of prediction
            training_stats: Training statistics (accuracy, loss, etc.)
            statistical_results: Statistical analysis results
            gradcam_analysis: Description of Grad-CAM visualization
            academic_references: List of academic references
        
        Returns:
            Formatted prompt for AI analysis
        """
        prompt = f"""
Como especialista em diagnóstico de imagens com nível de PhD, análise forense e residência médica,
realize uma interpretação METICULOSAMENTE DESCRITIVA e ANAMNÉSICA dos seguintes resultados de classificação:

## DADOS DO PACIENTE/AMOSTRA:
- **Classe Predita:** {predicted_class}
- **Nível de Confiança:** {confidence:.4f} ({confidence*100:.2f}%)

## INFORMAÇÕES DE TREINAMENTO:
"""
        
        # Add training statistics
        if training_stats:
            for key, value in training_stats.items():
                if isinstance(value, float):
                    prompt += f"- **{key}:** {value:.4f}\n"
                else:
                    prompt += f"- **{key}:** {value}\n"
        
        prompt += "\n## RESULTADOS ESTATÍSTICOS:\n"
        
        # Add statistical results
        if statistical_results:
            for key, value in statistical_results.items():
                if isinstance(value, (int, float)):
                    prompt += f"- **{key}:** {value:.4f}\n"
                else:
                    prompt += f"- **{key}:** {value}\n"
        
        # Add Grad-CAM analysis if available
        if gradcam_analysis:
            prompt += f"\n## ANÁLISE GRAD-CAM:\n{gradcam_analysis}\n"
        
        # Add academic references if available
        if academic_references:
            prompt += "\n## REFERÊNCIAS ACADÊMICAS RELEVANTES:\n"
            for i, ref in enumerate(academic_references[:5], 1):  # Limit to 5 references
                prompt += f"{i}. **{ref.get('title', 'N/A')}**\n"
                prompt += f"   - Autores: {ref.get('authors', 'N/A')}\n"
                prompt += f"   - Fonte: {ref.get('source', 'N/A')}\n"
                prompt += f"   - URL: {ref.get('url', 'N/A')}\n\n"
        
        prompt += """
## SOLICITAÇÃO DE ANÁLISE:

Por favor, forneça uma interpretação APROFUNDADA e DIAGNÓSTICA que inclua:

1. **📝 RESUMO EXECUTIVO (OBRIGATÓRIO):**
   - **Resumo Original (Inglês):** Breve resumo em inglês dos achados principais
   - **Resumo Traduzido (PT-BR):** Tradução completa do resumo para português brasileiro
   - **Resenha Crítica:** Análise crítica dos resultados, apontando forças, limitações e pontos de atenção

2. **ANÁLISE CLÍNICA/FORENSE DETALHADA:**
   - Interpretação minuciosa dos resultados
   - Significado clínico/científico da classificação
   - Fatores que podem ter influenciado a predição
   - Análise da confiança do modelo e suas implicações

3. **CORRELAÇÃO COM PADRÕES CONHECIDOS:**
   - Comparação com casos similares na literatura
   - Padrões característicos observados
   - Desvios ou peculiaridades notáveis

4. **INTERPRETAÇÃO MULTI-ANGULAR:**
   - Visão do ponto de vista morfológico
   - Análise de características texturais
   - Considerações contextuais
   - Implicações práticas

5. **DIAGNÓSTICO DIFERENCIAL:**
   - Classes alternativas consideradas
   - Razões para descarte de outras hipóteses
   - Casos limítrofes ou ambíguos

6. **RECOMENDAÇÕES E CONSIDERAÇÕES:**
   - Sugestões para confirmação diagnóstica
   - Limitações da análise atual
   - Necessidade de exames complementares
   - Considerações éticas e de boas práticas

7. **EMBASAMENTO CIENTÍFICO:**
   - Citações e referências relevantes
   - Metodologias estabelecidas
   - Evidências científicas de suporte

IMPORTANTE: 
- SEMPRE comece com o Resumo Executivo em três partes (Original em inglês, Traduzido em PT-BR, e Resenha Crítica)
- Mantenha um tom profissional, técnico e científico, como esperado em um diagnóstico de residência médica ou perícia forense de alto nível
- A resenha crítica deve ser imparcial e apontar tanto aspectos positivos quanto limitações
"""
        
        return prompt
    
    def analyze(self, prompt: str) -> str:
        """
        Send prompt to AI and get analysis
        
        Args:
            prompt: The diagnostic prompt
        
        Returns:
            AI-generated analysis
        """
        try:
            if self.api_provider == 'gemini':
                if GEMINI_NEW_API:
                    # New google-genai package API
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt
                    )
                    return response.text
                else:
                    # Old google-generativeai package API
                    response = self.model.generate_content(prompt)
                    return response.text
            elif self.api_provider == 'groq':
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a PhD-level specialist in image diagnostics with expertise in medical residency and forensic analysis."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model=self.model_name,
                    temperature=0.7,
                    max_tokens=4096,
                )
                return chat_completion.choices[0].message.content
            else:
                return "Provider not supported"
        except Exception as e:
            return f"Error generating analysis: {str(e)}"
    
    def generate_comprehensive_analysis(
        self,
        predicted_class: str,
        confidence: float,
        training_stats: Dict,
        statistical_results: Dict,
        gradcam_description: str = "",
        academic_references: List[Dict] = None
    ) -> str:
        """
        Generate comprehensive diagnostic analysis
        
        Returns:
            Complete AI-generated diagnostic report
        """
        prompt = self.generate_diagnostic_prompt(
            predicted_class=predicted_class,
            confidence=confidence,
            training_stats=training_stats,
            statistical_results=statistical_results,
            gradcam_analysis=gradcam_description,
            academic_references=academic_references
        )
        
        return self.analyze(prompt)


def describe_gradcam_regions(gradcam_array: np.ndarray, threshold: float = 0.5) -> str:
    """
    Analyze Grad-CAM heatmap and generate textual description
    
    Args:
        gradcam_array: Grad-CAM activation map (must be 2D numpy array)
        threshold: Threshold for significant activations
    
    Returns:
        Textual description of Grad-CAM regions
    """
    if gradcam_array is None:
        return "Grad-CAM não disponível"
    
    # Validate input
    if not isinstance(gradcam_array, np.ndarray):
        return "Grad-CAM: formato de dados inválido"
    
    if len(gradcam_array.shape) != 2:
        return "Grad-CAM: esperado array 2D, recebido formato diferente"
    
    # Normalize to 0-1
    if gradcam_array.max() > 0:
        gradcam_norm = gradcam_array / gradcam_array.max()
    else:
        return "Grad-CAM sem ativações significativas"
    
    # Calculate statistics
    high_activation_pct = (gradcam_norm > threshold).sum() / gradcam_norm.size * 100
    mean_activation = gradcam_norm.mean()
    max_activation = gradcam_norm.max()
    
    # Find regions
    height, width = gradcam_norm.shape
    high_regions = gradcam_norm > threshold
    
    # Describe location of high activations
    y_coords, x_coords = np.where(high_regions)
    if len(y_coords) > 0:
        center_y = y_coords.mean() / height
        center_x = x_coords.mean() / width
        
        # Determine quadrant
        if center_y < 0.33:
            v_position = "superior"
        elif center_y < 0.67:
            v_position = "central"
        else:
            v_position = "inferior"
        
        if center_x < 0.33:
            h_position = "esquerda"
        elif center_x < 0.67:
            h_position = "centro"
        else:
            h_position = "direita"
        
        position_desc = f"região {v_position} {h_position}"
    else:
        position_desc = "sem região específica destacada"
    
    description = f"""
Análise Grad-CAM:
- Porcentagem de ativação alta: {high_activation_pct:.2f}%
- Ativação média: {mean_activation:.4f}
- Ativação máxima: {max_activation:.4f}
- Localização principal: {position_desc}
- Interpretação: O modelo focou principalmente na {position_desc} da imagem, 
  com {high_activation_pct:.2f}% da área apresentando ativação significativa.
  Isso indica que as características decisivas para a classificação estão
  concentradas nesta região.
"""
    
    return description
