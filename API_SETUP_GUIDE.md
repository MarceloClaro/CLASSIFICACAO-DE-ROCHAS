# 🔑 Guia de Configuração de API para Análise com IA

Este guia explica como configurar e usar as APIs de IA (Gemini e Groq) no sistema de classificação de rochas.

## 📋 Visão Geral

O sistema suporta duas APIs para análise diagnóstica avançada com IA:
- **Google Gemini**: API de IA do Google com suporte a visão computacional
- **Groq**: API de inferência rápida com modelos de linguagem

## 🚀 Instalação dos Pacotes

### Opção 1: Google Generative AI (Recomendado)

```bash
pip install google-generativeai
```

Este é o pacote **estável e recomendado**. O sistema foi testado com esta versão.

### Opção 2: Google GenAI (Novo - Experimental)

```bash
pip install google-genai
```

Este é o **novo pacote** do Google que está em desenvolvimento. O sistema detecta automaticamente qual pacote está instalado e usa a API apropriada.

### Groq API

```bash
pip install groq
```

## 🔧 Compatibilidade

O código suporta **automaticamente** ambas as versões da API do Google:

| Pacote | Versão | Status | Método de Inicialização |
|--------|--------|--------|-------------------------|
| `google-generativeai` | < 1.0 | ✅ Estável | `genai.configure(api_key)` |
| `google-genai` | >= 0.2 | ⚠️ Experimental | `genai.Client(api_key)` |

### Como Funciona

O sistema detecta automaticamente qual pacote está instalado:

1. **Tenta importar** `google.genai` (novo pacote)
2. **Se falhar**, tenta importar `google.generativeai` (pacote antigo)
3. **Usa a API apropriada** baseado no pacote detectado

## 🎯 Obtendo as API Keys

### Google Gemini

1. Acesse: https://ai.google.dev/
2. Clique em "Get API Key"
3. Crie um novo projeto ou selecione um existente
4. Copie a API Key gerada

**Modelos Disponíveis (API v1beta):**
- `gemini-1.5-pro-latest` - ⭐ **RECOMENDADO** - Modelo mais avançado e recente
- `gemini-1.5-flash-latest` - Modelo rápido e eficiente
- `gemini-1.0-pro-latest` - Modelo estável
- `gemini-pro` - Modelo para uso geral
- `gemini-1.0-pro-vision-latest` - Visão mais recente

**Nota:** Os modelos com sufixo `-latest` são atualizados automaticamente para a versão mais recente disponível.

### Groq

1. Acesse: https://console.groq.com/
2. Faça login ou crie uma conta
3. Vá para "API Keys"
4. Gere uma nova API Key

**Modelos Disponíveis:**

**Modelos Multimodais (com suporte a visão):**
- `meta-llama/llama-4-scout-17b-16e-instruct` - ⭐ **RECOMENDADO** - Scout Llama 4 (multimodal, 128K contexto)
- `meta-llama/llama-4-maverick-17b-128e-instruct` - Llama 4 Maverick (multimodal, 128K contexto)

**Modelos Apenas Texto:**
- `mixtral-8x7b-32768` - Modelo Mixtral com 32K tokens de contexto
- `llama-3.1-70b-versatile` - LLama 3.1 70B
- `llama-3.1-8b-instant` - LLama 3.1 8B (mais rápido)

## 📱 Configuração na Interface

### Configuração na Barra Lateral

1. Na barra lateral, expanda "**Configurar API (Gemini/Groq)**"
2. Selecione o **Provedor de API** (Gemini ou Groq)
3. Escolha o **Modelo** desejado
4. Insira sua **API Key**
5. Aguarde a mensagem de confirmação "✅ API Key configurada!"

### Usando a Análise com IA

Após configurar a API:

1. Faça o upload de uma imagem para avaliação
2. Aguarde a classificação do modelo
3. Role até a seção "**🤖 Análise Diagnóstica Avançada com IA**"
4. Marque "**Ativar Análise Diagnóstica Completa com IA**"
5. Clique em "**🔬 Gerar Análise Diagnóstica Completa**"

## 📝 Formato da Análise

A análise com IA inclui **obrigatoriamente** três componentes no resumo:

### 1. 📋 Resumo Original (Inglês)
Breve resumo dos principais achados em inglês científico.

### 2. 🇧🇷 Resumo Traduzido (PT-BR)
Tradução completa e precisa do resumo para português brasileiro.

### 3. 🔍 Resenha Crítica
Análise crítica imparcial apontando:
- ✅ Aspectos positivos e forças da classificação
- ⚠️ Limitações e pontos de atenção
- 📊 Confiabilidade dos resultados
- 💡 Recomendações para melhorias

## 🐛 Resolução de Problemas

### Erro: "module 'google.genai' has no attribute 'configure'"

**Causa:** Você tem o pacote `google-genai` instalado, mas o código estava tentando usar a API antiga.

**Solução:** ✅ JÁ CORRIGIDO! O código agora detecta automaticamente qual pacote está instalado.

Se ainda tiver problemas:

```bash
# Desinstale ambos os pacotes
pip uninstall google-genai google-generativeai -y

# Instale o pacote recomendado
pip install google-generativeai
```

### Erro: "Google Generative AI não está disponível"

**Causa:** Nenhum dos pacotes está instalado.

**Solução:**

```bash
pip install google-generativeai
```

### Erro: "API key inválida" ou "401 Unauthorized"

**Possíveis causas:**
1. API Key incorreta ou expirada
2. Projeto sem créditos ou billing desabilitado
3. API não habilitada no projeto

**Solução:**
1. Verifique se copiou a API Key completa
2. Confirme que o billing está ativo (para Gemini)
3. Verifique se você tem créditos disponíveis

### Erro: Rate Limit ou Quota Exceeded

**Causa:** Você excedeu o limite de requisições por minuto/dia.

**Solução:**
- Aguarde alguns minutos
- Considere upgrade do plano
- Para Groq: Verifique seus limites em https://console.groq.com/

### Erro: "404 models/gemini-1.5-pro is not found for API version v1beta"

**Causa:** Uso de nomes de modelo incorretos ou sem o sufixo `-latest` recomendado.

**Solução:** ✅ JÁ CORRIGIDO! O código agora usa os modelos corretos disponíveis na API v1beta:
- `gemini-1.5-pro-latest` ⭐ (recomendado)
- `gemini-1.5-flash-latest`
- `gemini-1.0-pro-latest`
- `gemini-pro`
- `gemini-1.0-pro-vision-latest`

**Importante:** Use sempre os modelos com `-latest` quando disponível para garantir acesso à versão mais recente.

Se ainda tiver problemas:

```bash
# Atualize o pacote google-generativeai
pip install --upgrade google-generativeai

# Ou, se estiver usando o pacote beta, migre para o estável:
pip uninstall google-genai -y
pip install google-generativeai
```

**Nota:** O pacote `google-generativeai` é mais estável e recomendado para uso em produção.

## 💡 Dicas e Boas Práticas

### Escolha do Modelo

**Para análises detalhadas e raciocínio complexo:**
- Gemini: Use `gemini-1.5-pro-latest` ⭐ **RECOMENDADO**
- Groq: Use `mixtral-8x7b-32768` ou `llama-3.1-70b-versatile`

**Para análises rápidas e eficientes:**
- Gemini: Use `gemini-1.5-flash-latest` ⭐ **RECOMENDADO**
- Groq: Use `llama-3.1-8b-instant`

**Para análise de imagens:**
- Gemini: Use `gemini-1.5-pro-latest` ⭐ **RECOMENDADO** ou `gemini-1.0-pro-vision-latest`

### Segurança da API Key

⚠️ **IMPORTANTE:**
- Nunca compartilhe sua API Key
- Não commite API Keys no código
- Use variáveis de ambiente em produção
- Revogue keys comprometidas imediatamente

### Otimização de Custos

**Gemini:**
- Modelos 1.5-flash são mais baratos
- Verifique pricing em https://ai.google.dev/pricing

**Groq:**
- Serviço gratuito com limites
- Muito rápido para inferência

## 📚 Referências

- [Google AI for Developers](https://ai.google.dev/)
- [Google Generative AI Python SDK](https://github.com/google/generative-ai-python)
- [Groq Documentation](https://console.groq.com/docs)
- [Groq Python SDK](https://github.com/groq/groq-python)

## 🆘 Suporte

Se você encontrar problemas:

1. Verifique se seguiu todos os passos deste guia
2. Confirme que a API Key está correta
3. Verifique os logs de erro na interface
4. Consulte a documentação oficial das APIs

Para problemas específicos do sistema, entre em contato:
- Email: marceloclaro@gmail.com
- WhatsApp: (88) 981587145
- Instagram: [@marceloclaro.geomaker](https://www.instagram.com/marceloclaro.geomaker/)

---

**Projeto:** Geomaker + IA  
**DOI:** [10.5281/zenodo.13910277](https://doi.org/10.5281/zenodo.13910277)  
**Professor:** Marcelo Claro
