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

**Modelos Disponíveis (Recomendados - 2024):**

**Modelos de Nova Geração (Recomendados):**
- `gemini-2.5-flash` - ⭐ **RECOMENDADO** - Rápido e eficiente, multimodal
- `gemini-2.5-flash-lite` - Ultra rápido para tarefas simples
- `gemini-2.5-pro` - Avançado com capacidade de raciocínio superior
- `gemini-3-flash-preview` - Próxima geração (preview)
- `gemini-3-pro-preview` - Avançado próxima geração (preview)

**Modelos Legados (não recomendados):**
- `gemini-1.5-pro-latest` - Modelo mais antigo
- `gemini-1.5-flash-latest` - Modelo rápido legado
- `gemini-1.0-pro-latest` - Modelo estável legado

**Nota:** Os modelos 2.5 e 3.0 são os mais atuais e recomendados. Baseado no [Gemini API Cookbook](https://github.com/google-gemini/cookbook).

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

### Erro: Rate Limit ou Quota Exceeded (429)

**Causa:** Você excedeu o limite de requisições por minuto/dia, ou sua quota gratuita foi esgotada.

**Mensagem típica:**
```
429 You exceeded your current quota, please check your plan and billing details.
Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_*
```

**Solução:**

1. **Aguarde alguns minutos** - Os limites são reiniciados após um tempo
2. **Verifique sua quota** em https://ai.dev/usage?tab=rate-limit
3. **Considere upgrade do plano** se você precisa de mais requisições
4. **Use modelos mais eficientes:**
   - `gemini-2.5-flash-lite` (mais leve, usa menos quota)
   - `gemini-2.5-flash` (balanço entre eficiência e qualidade)
5. **Para Groq:** Verifique seus limites em https://console.groq.com/

**Limites do Free Tier (Gemini):**
- Requisições por minuto: Limitado
- Requisições por dia: Limitado
- Tokens de entrada por dia: Limitado
- Tokens de entrada por minuto: Limitado

**Dica:** Se você está desenvolvendo/testando, considere adicionar delays entre requisições ou usar o plano pago para limites maiores.

### Erro: "404 models/gemini-1.5-pro is not found for API version v1beta"

**Causa:** Uso de nomes de modelo incorretos ou modelos descontinuados.

**Solução:** ✅ JÁ CORRIGIDO! O código agora usa os modelos corretos disponíveis:

**Modelos Recomendados (2024):**
- `gemini-2.5-flash` ⭐ (recomendado)
- `gemini-2.5-flash-lite`
- `gemini-2.5-pro`
- `gemini-3-flash-preview`
- `gemini-3-pro-preview`

**Modelos Legados (ainda funcionam):**
- `gemini-1.5-pro-latest`
- `gemini-1.5-flash-latest`

**Importante:** Use sempre os modelos da série 2.5 ou 3.0 para melhor desempenho e recursos mais recentes.

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
- Gemini: Use `gemini-2.5-pro` ⭐ **RECOMENDADO**
- Groq: Use `mixtral-8x7b-32768` ou `llama-3.1-70b-versatile`

**Para análises rápidas e eficientes:**
- Gemini: Use `gemini-2.5-flash` ⭐ **RECOMENDADO**
- Groq: Use `llama-3.1-8b-instant`

**Para análise de imagens (multimodal):**
- Gemini: Use `gemini-2.5-flash` ⭐ **RECOMENDADO** ou `gemini-2.5-pro`
- Groq: Use `llama-4-scout-17b-16e-instruct` ⭐ (multimodal)

**Para economia de quota (free tier):**
- Gemini: Use `gemini-2.5-flash-lite` ⭐ **MAIS LEVE** - Consome menos tokens
- Groq: Use `llama-3.1-8b-instant`

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

## 🤖 Sistema Multi-Agente e CrewAI

### O que é o Sistema Multi-Agente?

O sistema inclui 15 agentes especializados + 1 gerente coordenador que analisam a imagem classificada de múltiplas perspectivas:

- **Agente Morfológico** - Análise de forma e estrutura
- **Agente Textural** - Análise de textura e padrões
- **Agente Cromático** - Análise de cor e tonalidade
- **Agente Espacial** - Análise de distribuição espacial
- **Agente Estatístico** - Análise estatística e métricas
- **Agente de Diagnóstico Diferencial** - Análise de alternativas
- **Agente de Qualidade** - Controle de qualidade
- **Agente Contextual** - Análise de contexto
- **Agente Bibliográfico** - Revisão de literatura
- **Agente Metodológico** - Avaliação metodológica
- **Agente de Risco** - Avaliação de risco e incertezas
- **Agente Comparativo** - Análise comparativa
- **Agente de Relevância Clínica** - Relevância prática
- **Agente de Integração** - Integração multi-modal
- **Agente de Validação** - Validação cruzada

**Importante:** O sistema multi-agente **funciona sem necessidade de configuração adicional** - não requer API keys extras.

### CrewAI (Opcional - EXPERIMENTAL)

O CrewAI é uma funcionalidade **opcional e experimental** que adiciona análise avançada usando inteligência artificial colaborativa.

**Requisitos para usar CrewAI:**
- ✅ Pacote `crewai` instalado: `pip install crewai crewai-tools`
- ✅ Variável de ambiente `OPENAI_API_KEY` configurada
- ✅ Conta OpenAI com créditos disponíveis

**Como configurar:**

```bash
# No terminal, antes de executar o app
export OPENAI_API_KEY='sua-chave-openai-aqui'

# Ou no Windows
set OPENAI_API_KEY=sua-chave-openai-aqui
```

**Nota:** Se você não tem uma API key da OpenAI, **não ative o CrewAI**. O sistema multi-agente funciona perfeitamente sem ele.

**Quando usar CrewAI:**
- ✅ Quando você precisa de análises ainda mais profundas
- ✅ Quando você tem uma API key da OpenAI disponível
- ✅ Quando você quer correlações avançadas com literatura científica

**Quando NÃO usar CrewAI:**
- ❌ Se você não tem API key da OpenAI
- ❌ Se você quer análise mais rápida
- ❌ Se você quer economizar créditos de API

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
