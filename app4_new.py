def analyze_image_with_gemini(image, api_key, model_name, class_name, confidence, gradcam_description="", gradcam_image=None, max_retries=2):
    """
    Analisa uma imagem usando Google Gemini com visão computacional.
    Inclui retry automático para erros de rate limit e otimização de imagens.
    
    Args:
        image: PIL Image (imagem original)
        api_key: Chave API do Gemini
        model_name: Nome do modelo Gemini (deve suportar visão)
        class_name: Classe predita pelo modelo
        confidence: Confiança da predição
        gradcam_description: Descrição textual do Grad-CAM
        gradcam_image: PIL Image com Grad-CAM sobreposto (opcional)
        max_retries: Número máximo de tentativas em caso de rate limit
    
    Returns:
        str: Análise técnica e forense da imagem
    """
    if not GEMINI_AVAILABLE:
        return "Google Generative AI não está disponível. Instale com: pip install google-generativeai"
    
    # Otimizar imagens antes de enviar (reduz custos e melhora performance)
    optimized_image = optimize_image_for_api(image, max_size=(1024, 1024))
    optimized_gradcam = optimize_image_for_api(gradcam_image, max_size=(1024, 1024)) if gradcam_image is not None else None
    
    # Construir prompt baseado na disponibilidade de Grad-CAM
    if optimized_gradcam is not None:
        prompt = f"""
Você é um especialista em análise de imagens e interpretação técnica e forense.

**Contexto da Classificação:**
- Classe Predita: {class_name}
- Confiança: {confidence:.4f} ({confidence*100:.2f}%)
- Análise Grad-CAM: {gradcam_description if gradcam_description else 'Veja a segunda imagem'}

**IMPORTANTE:** Você receberá DUAS imagens:
1. **Primeira imagem**: A imagem ORIGINAL classificada
2. **Segunda imagem**: A mesma imagem com sobreposição de Grad-CAM (mapa de calor vermelho-amarelo)

O Grad-CAM (Gradient-weighted Class Activation Mapping) mostra onde a rede neural focou sua "atenção" 
para fazer a classificação. Regiões em vermelho/amarelo indicam áreas de alta importância para a decisão.

Por favor, realize uma análise COMPLETA e DETALHADA das DUAS imagens, incluindo:

1. **Descrição Visual da Imagem Original:**
   - Descreva todos os elementos visuais presentes na imagem original
   - Identifique padrões, texturas, cores e formas relevantes
   - Analise a qualidade e características da imagem

2. **Análise do Grad-CAM (Segunda Imagem):**
   - Identifique quais regiões da imagem têm maior ativação (vermelho/amarelo intenso)
   - Descreva O QUE está presente nessas regiões de alta ativação
   - Avalie se essas regiões fazem sentido para a classificação como "{class_name}"
   - Compare: O modelo está focando nas características corretas?

3. **Interpretação Técnica Integrada:**
   - Avalie se a classificação como "{class_name}" é compatível com o que você observa
   - Relacione as características visuais da imagem original com as regiões de ativação
   - Analise se a confiança de {confidence*100:.2f}% é justificada pelas regiões focadas
   - Identifique se há características importantes ignoradas pelo modelo

4. **Análise Forense:**
   - Identifique possíveis artefatos ou anomalias nas imagens
   - Avalie a integridade e autenticidade da imagem
   - Verifique se o Grad-CAM está focando em artefatos em vez de características reais
   - Destaque áreas de interesse ou preocupação

5. **Recomendações:**
   - Sugira se a classificação deve ser aceita ou revista
   - Baseie-se na correlação entre características visuais e regiões de ativação
   - Recomende análises adicionais se necessário
   - Forneça orientações para melhorar a confiança na classificação

Seja detalhado, técnico e preciso na sua análise. Relacione SEMPRE os dois aspectos: 
o que você vê na imagem original e onde o modelo está focando no Grad-CAM.
"""
    else:
        prompt = f"""
Você é um especialista em análise de imagens e interpretação técnica e forense.

**Contexto da Classificação:**
- Classe Predita: {class_name}
- Confiança: {confidence:.4f} ({confidence*100:.2f}%)
- Análise Grad-CAM: {gradcam_description if gradcam_description else 'Não disponível'}

Por favor, realize uma análise COMPLETA e DETALHADA da imagem fornecida, incluindo:

1. **Descrição Visual Detalhada:**
   - Descreva todos os elementos visuais presentes na imagem
   - Identifique padrões, texturas, cores e formas relevantes
   - Analise a qualidade e características da imagem

2. **Interpretação Técnica:**
   - Avalie se a classificação como "{class_name}" é compatível com o que você observa
   - Identifique características específicas que suportam ou contradizem a classificação
   - Analise a confiança de {confidence*100:.2f}% em relação aos padrões visuais

3. **Análise Forense:**
   - Identifique possíveis artefatos ou anomalias na imagem
   - Avalie a integridade e autenticidade da imagem
   - Destaque áreas de interesse ou preocupação

4. **Recomendações:**
   - Sugira se a classificação deve ser aceita ou revista
   - Recomende análises adicionais se necessário
   - Forneça orientações para melhorar a confiança na classificação

Seja detalhado, técnico e preciso na sua análise.
"""
    
    # Função interna para fazer a chamada da API
    def make_api_call():
        if GEMINI_NEW_API:
            # New beta google-genai package API
            client = genai.Client(api_key=api_key)
            
            # Convert PIL images to bytes
            img_byte_arr = io.BytesIO()
            optimized_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Get correct model path for beta API
            model_path = get_gemini_model_path(model_name, use_new_api=True)
            
            # Build content list
            content_parts = [prompt, {"mime_type": "image/png", "data": img_byte_arr}]
            
            # Add Grad-CAM image if available
            if optimized_gradcam is not None:
                gradcam_byte_arr = io.BytesIO()
                optimized_gradcam.save(gradcam_byte_arr, format='PNG')
                gradcam_byte_arr = gradcam_byte_arr.getvalue()
                content_parts.append({"mime_type": "image/png", "data": gradcam_byte_arr})
            
            response = client.models.generate_content(
                model=model_path,
                contents=content_parts
            )
            return (True, response.text)
        else:
            # Stable google-generativeai package API (recommended)
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            
            # Build content list
            content_parts = [prompt, optimized_image]
            
            # Add Grad-CAM image if available
            if optimized_gradcam is not None:
                content_parts.append(optimized_gradcam)
            
            response = model.generate_content(content_parts)
            return (True, response.text)
    
    # Tentar fazer a chamada com retry
    try:
        return retry_api_call(make_api_call, max_retries=max_retries, initial_delay=2.0, backoff_factor=2.0)
    except Exception as e:
        error_msg = f"Erro ao analisar com Gemini: {str(e)}\n\n"
        error_type = str(e).lower()
        
        # Provide helpful guidance based on error type
        if "configure" in error_type:
            error_msg += (
                "💡 Dica: Parece que há um problema de configuração da API.\n"
                "   Certifique-se de usar: pip install google-generativeai\n"
            )
        elif "404" in str(e) and "not found" in error_type:
            error_msg += (
                "🔍 Modelo não encontrado. Use os modelos atuais do Gemini API.\n"
                "   📚 Baseado no cookbook oficial: https://github.com/google-gemini/cookbook\n"
                "   \n"
                "   Modelos recomendados (todos com suporte multimodal/visão):\n"
                "   - gemini-2.0-flash-exp ⭐ RECOMENDADO (última versão, grátis)\n"
                "   - gemini-1.5-flash (rápido e eficiente)\n"
                "   - gemini-1.5-pro (avançado com capacidade de raciocínio)\n"
                "   \n"
                "   ⚠️ Modelos legados (1.0) não são mais recomendados\n"
            )
        elif "api key" in error_type or "401" in str(e) or "403" in str(e):
            error_msg += (
                "🔑 Verifique se a API key está correta e ativa.\n"
                "   Obtenha sua API key em: https://ai.google.dev/\n"
            )
        elif "quota" in error_type or "rate limit" in error_type or "429" in str(e):
            error_msg += (
                "⏱️ Limite de requisições atingido. Aguarde alguns minutos.\n"
                f"   Tentativas realizadas: {max_retries}\n"
                "   💡 Sugestões:\n"
                "   - Aguarde 1-2 minutos antes de tentar novamente\n"
                "   - Verifique seu limite em: https://ai.google.dev/\n"
                "   - Use a análise Multi-Agente como alternativa (não requer API externa)\n"
            )
        elif "resource" in error_type and "exhausted" in error_type:
            error_msg += (
                "💳 Recursos/créditos esgotados. Verifique sua conta.\n"
            )
        else:
            error_msg += (
                "📖 Consulte o guia: API_SETUP_GUIDE.md para mais detalhes.\n"
            )
        
        return error_msg
