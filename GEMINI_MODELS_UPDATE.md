# Atualização dos Modelos Gemini (Baseado no Cookbook Oficial)

## 📚 Referência
Este documento baseia-se no [Cookbook Oficial do Google Gemini](https://github.com/google-gemini/cookbook).

## ✅ Modelos Atuais Recomendados

### Modelos Gemini 2.5 e 3.0 (Recomendados)

Todos os modelos Gemini 2.5+ possuem **suporte multimodal nativo**, incluindo:
- 📷 Análise de imagens
- 🎵 Análise de áudio
- 📄 Análise de PDFs
- 🎬 Análise de vídeos

| Modelo | Descrição | Uso Recomendado |
|--------|-----------|-----------------|
| **gemini-2.5-flash** ⭐ | Rápido e eficiente | **RECOMENDADO** para uso geral |
| **gemini-2.5-flash-lite** | Ainda mais rápido | Respostas ultra-rápidas |
| **gemini-2.5-pro** | Avançado com raciocínio | Tarefas complexas |
| **gemini-3-flash-preview** | Próxima geração | Preview - teste de recursos futuros |
| **gemini-3-pro-preview** | Avançado próxima geração | Preview - capacidades avançadas |

### Características dos Modelos 2.5+

1. **Suporte Multimodal Nativo**: Não é necessário um modelo separado para visão (`-vision`)
2. **Capacidade de Raciocínio (Thinking)**: Modelos 2.5+ incluem fase de análise antes da resposta
3. **Auto-atualização**: Modelos se atualizam automaticamente com melhorias

## ❌ Modelos Descontinuados/Não Recomendados

### Modelos Gemini 1.5 (Legados)
- ~~gemini-1.5-pro-latest~~ → Use `gemini-2.5-pro`
- ~~gemini-1.5-flash-latest~~ → Use `gemini-2.5-flash`
- ~~gemini-1.5-flash~~ → Use `gemini-2.5-flash`
- ~~gemini-1.5-pro~~ → Use `gemini-2.5-pro`

### Modelos Gemini 1.0 (Descontinuados)
- ~~gemini-1.0-pro-latest~~ → Use `gemini-2.5-flash`
- ~~gemini-1.0-pro~~ → Use `gemini-2.5-flash`
- ~~gemini-1.0-pro-vision-latest~~ → Use `gemini-2.5-flash` (visão nativa)
- ~~gemini-pro-vision~~ ❌ **NÃO EXISTE** na API v1beta → Use `gemini-2.5-flash`
- ~~gemini-pro~~ → Use `gemini-2.5-flash`

## 🔄 Migração

### De Modelos 1.0/1.5 para 2.5+

**Antes:**
```python
# Código antigo com modelo descontinuado
model = genai.GenerativeModel('gemini-1.5-pro-latest')
model = genai.GenerativeModel('gemini-pro-vision')  # ❌ Nunca existiu!
```

**Depois:**
```python
# Código atualizado com modelo recomendado
model = genai.GenerativeModel('gemini-2.5-flash')  # ⭐ Recomendado
```

### Não Há Mais Modelos Separados para Visão

**Antes:**
```python
# Modelo específico para análise de imagens
vision_model = genai.GenerativeModel('gemini-pro-vision')  # ❌ Erro 404
image_model = genai.GenerativeModel('gemini-1.0-pro-vision-latest')
```

**Depois:**
```python
# Todos os modelos 2.5+ têm suporte multimodal nativo
model = genai.GenerativeModel('gemini-2.5-flash')
# Funciona para texto, imagens, áudio, vídeo e PDFs!
```

## 📦 Pacote SDK

### Pacote Recomendado
```bash
pip install -U google-genai>=1.51.0
```

### Uso Básico
```python
from google import genai
from google.genai import types

# Configurar cliente
client = genai.Client(api_key="SUA_API_KEY")

# Usar modelo recomendado
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Sua pergunta aqui"
)

print(response.text)
```

### Análise Multimodal (Imagem + Texto)
```python
from PIL import Image

image = Image.open("imagem.jpg")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        image,
        "Descreva esta imagem em detalhes"
    ]
)

print(response.text)
```

## 🔗 Recursos Adicionais

- [Cookbook Oficial](https://github.com/google-gemini/cookbook)
- [Get Started Guide](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started.ipynb)
- [Documentação Oficial](https://ai.google.dev/gemini-api/docs)
- [Obter API Key](https://aistudio.google.com/app/apikey)

## 📊 Comparação de Performance

| Modelo | Velocidade | Capacidades | Contexto |
|--------|-----------|-------------|----------|
| gemini-2.5-flash-lite | ⚡⚡⚡⚡⚡ | Básicas | Médio |
| gemini-2.5-flash | ⚡⚡⚡⚡ | Completas | Grande |
| gemini-2.5-pro | ⚡⚡⚡ | Avançadas + Raciocínio | Muito Grande |

## ⚠️ Notas Importantes

1. **gemini-pro-vision nunca existiu na API v1beta** - Este modelo causava erro 404
2. Todos os modelos 2.5+ incluem suporte multimodal nativo
3. Modelos 1.0 e 1.5 não são mais recomendados
4. Use sempre `gemini-2.5-flash` como padrão a menos que precise de capacidades específicas

## 🚀 Recomendação

Para a maioria dos casos de uso, incluindo análise de imagens/visão:

```python
MODEL_ID = "gemini-2.5-flash"  # ⭐ RECOMENDADO
```

---

*Documento baseado no [Google Gemini Cookbook](https://github.com/google-gemini/cookbook)*
*Atualizado em: 2025-12-20*
