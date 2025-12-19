# 🚀 Guia Rápido de Início

## ⚡ Início Rápido em 5 Minutos

### 1. Instale as Dependências

```bash
pip install -r requirements.txt
```

### 2. Prepare seus Dados

Organize suas imagens em pastas por classe:

```
dataset.zip
├── classe1/
│   ├── imagem1.jpg
│   ├── imagem2.jpg
│   └── ...
├── classe2/
│   ├── imagem1.jpg
│   ├── imagem2.jpg
│   └── ...
└── classe3/
    ├── imagem1.jpg
    ├── imagem2.jpg
    └── ...
```

### 3. Execute a Aplicação

```bash
streamlit run app3.py
```

### 4. Configure e Treine

1. **Na barra lateral**, configure:
   - Número de Classes: 3 (neste exemplo)
   - Modelo: ResNet18 (para teste rápido)
   - Épocas: 10 (para teste rápido)
   
2. **Faça upload** do arquivo ZIP com suas imagens

3. **Aguarde o treinamento** - Você verá:
   - Progress bars
   - Métricas em tempo real
   - Gráficos de aprendizado

4. **Análise Automática** - Após o treinamento:
   - Métricas detalhadas de classificação
   - Análise de eficiência e desempenho
   - Score de eficiência geral
   - Opção de exportar relatório

## 📊 Exemplo de Resultado Esperado

```
✅ Treinamento Concluído!

Análise de Eficiência e Desempenho
────────────────────────────────────

Métricas de Classificação:
  Acurácia: 0.9200
  F1-Score: 0.9150
  
Métricas de Eficiência:
  Tempo de Inferência: 18.5 ms
  Amostras/seg: 54.0
  
Score de Eficiência: 0.8500 ✅ Excelente!

📥 [Download Relatório CSV]
```

## 🎯 Configurações Recomendadas por Cenário

### 🧪 Teste Rápido (5-10 minutos)
```
Modelo: ResNet18
Épocas: 10
Batch Size: 16
Fine-Tuning: Desabilitado
```

### 🎓 Uso Acadêmico/Qualis A1 (30-60 minutos)
```
Modelo: DenseNet121 ou ResNet50
Épocas: 100-200
Batch Size: 16
Fine-Tuning: Habilitado
L2 Regularization: 0.01
Early Stopping: 5
```

### 🏭 Produção/Alta Performance (1-2 horas)
```
Modelo: ResNet50
Épocas: 200
Batch Size: 32
Fine-Tuning: Habilitado
L2 Regularization: 0.01
Weighted Loss: Habilitar se desbalanceado
```

### 📱 Mobile/Edge (otimização)
```
Modelo: ResNet18
Épocas: 50-100
Batch Size: 8
Fine-Tuning: Desabilitado
```

## 🔧 Solução de Problemas Comuns

### Erro: "CUDA out of memory"
**Solução**: Reduza o batch size (ex: 8 ou 4)

### Overfitting (grande diferença treino/validação)
**Solução**: 
- Habilite L2 Regularization
- Reduza número de épocas
- Use Early Stopping
- Adicione mais dados aumentados

### Underfitting (baixa acurácia em ambos)
**Solução**:
- Aumente número de épocas
- Habilite Fine-Tuning
- Aumente taxa de aprendizagem
- Use modelo mais complexo

### Treinamento muito lento
**Solução**:
- Reduza número de épocas
- Aumente batch size (se memória permitir)
- Use modelo mais simples (ResNet18)
- Desabilite Fine-Tuning

## 📖 Próximos Passos

1. **Explore o Guia de Comparação**:
   ```bash
   streamlit run model_comparison_guide.py
   ```

2. **Leia a Documentação Completa**:
   - [PERFORMANCE_ANALYSIS.md](PERFORMANCE_ANALYSIS.md)
   - [README.md](README.md)

3. **Experimente Diferentes Modelos**:
   - Treine com ResNet18, ResNet50 e DenseNet121
   - Compare os resultados

4. **Otimize seu Modelo**:
   - Ajuste hiperparâmetros
   - Experimente diferentes técnicas de regularização
   - Analise as métricas detalhadas

## 💡 Dicas Importantes

✅ **Use Seed Fixo**: Garantido automaticamente (seed=42)
✅ **Salve seus Relatórios**: Use o botão de exportar CSV
✅ **Monitore o Score de Eficiência**: Objetivo > 0.80 para Qualis A1
✅ **Verifique GPU**: `torch.cuda.is_available()` - Acelera muito!
✅ **Documente seus Resultados**: Use os gráficos e métricas exportadas

## 🎓 Para Uso Acadêmico

### Checklist de Publicação Qualis A1

- [ ] Usar conjunto de teste independente
- [ ] Reportar múltiplas métricas (não só acurácia)
- [ ] Incluir matriz de confusão
- [ ] Calcular intervalos de confiança
- [ ] Comparar com baselines
- [ ] Documentar hiperparâmetros
- [ ] Garantir reprodutibilidade (seed)
- [ ] Incluir análise de erros
- [ ] Exportar resultados para tabelas
- [ ] Citar referências apropriadas

## 📧 Precisa de Ajuda?

- **Email**: marceloclaro@gmail.com
- **WhatsApp**: (88) 981587145
- **Issues**: GitHub Issues do projeto

## 🌟 Exemplos de Sucesso

### Caso 1: Classificação de Rochas
```
Dataset: 500 imagens, 3 classes
Modelo: ResNet50
Resultado: 95.2% acurácia, Score 0.87
Tempo: 45 minutos
```

### Caso 2: Diagnóstico de Lesões
```
Dataset: 1000 imagens, 7 classes
Modelo: DenseNet121
Resultado: 93.8% acurácia, Score 0.84
Tempo: 90 minutos
```

### Caso 3: Controle de Qualidade
```
Dataset: 300 imagens, 2 classes
Modelo: ResNet18
Resultado: 98.1% acurácia, Score 0.91
Tempo: 20 minutos
```

---

**Pronto para começar?** Execute `streamlit run app3.py` e comece a treinar!

🎯 **Objetivo**: Alcançar Score de Eficiência > 0.80 para qualidade Qualis A1!
