import os
import zipfile
import shutil
import tempfile
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models, datasets
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
import streamlit as st
import gc
import cv2
import logging
import base64
import torch.nn as nn
import torchvision.models as models

# Importações adicionais para Grad-CAM
from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms.functional import normalize, resize, to_pil_image

# Importação para modelos de segmentação
import segmentation_models_pytorch as smp

# Definir o dispositivo (CPU ou GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configurações para tornar os gráficos mais bonitos
sns.set_style('whitegrid')

def set_seed(seed):
    """
    Define a seed para garantir a reprodutibilidade.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Definir a seed para reprodutibilidade

# Definir as transformações para aumento de dados (aplicando transformações aleatórias)
train_transforms = transforms.Compose([
    transforms.RandomApply([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=90),
        transforms.ColorJitter(),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomAffine(degrees=0, shear=10),
    ], p=0.5),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Transformações para validação e teste
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Dataset personalizado
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def seed_worker(worker_id):
    """
    Função para definir a seed em cada worker do DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def visualize_data(dataset, classes):
    """
    Exibe algumas imagens do conjunto de dados com suas classes.
    """
    st.write("Visualização de algumas imagens do conjunto de dados:")
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        idx = np.random.randint(len(dataset))
        image, label = dataset[idx]
        image = np.array(image)  # Converter a imagem PIL em array NumPy
        axes[i].imshow(image)
        axes[i].set_title(classes[label])
        axes[i].axis('off')
    st.pyplot(fig)

def plot_class_distribution(dataset, classes):
    """
    Exibe a distribuição das classes no conjunto de dados e mostra os valores quantitativos.
    """
    # Extrair os rótulos das classes para todas as imagens no dataset
    labels = [label for _, label in dataset]
    
    # Contagem de cada classe
    class_counts = np.bincount(labels)
    
    # Plotar o gráfico com as contagens
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x=labels, ax=ax, palette="Set2")
    
    # Adicionar os nomes das classes no eixo X
    ax.set_xticklabels(classes, rotation=45)
    
    # Adicionar as contagens acima das barras
    for i, count in enumerate(class_counts):
        ax.text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
    
    ax.set_title("Distribuição das Classes (Quantidade de Imagens)")
    ax.set_xlabel("Classes")
    ax.set_ylabel("Número de Imagens")
    
    st.pyplot(fig)

def get_model(model_name, num_classes, dropout_p=0.5, fine_tune=False):
    """
    Retorna o modelo pré-treinado selecionado.
    """
    if model_name == 'ResNet18':
        model = models.resnet18(pretrained=True)
    elif model_name == 'ResNet50':
        model = models.resnet50(pretrained=True)
    elif model_name == 'DenseNet121':
        model = models.densenet121(pretrained=True)
    else:
        st.error("Modelo não suportado.")
        return None

    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False

    if model_name.startswith('ResNet'):
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(num_ftrs, num_classes)
        )
    elif model_name.startswith('DenseNet'):
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(num_ftrs, num_classes)
        )
    else:
        st.error("Modelo não suportado.")
        return None

    model = model.to(device)
    return model

def train_model(data_dir, num_classes, model_name, fine_tune, epochs, learning_rate, batch_size, train_split, valid_split, use_weighted_loss, l2_lambda, patience):
    """
    Função principal para treinamento do modelo.
    """
    set_seed(42)

    # Carregar o dataset original sem transformações
    full_dataset = datasets.ImageFolder(root=data_dir)

    # Exibir algumas imagens do dataset
    visualize_data(full_dataset, full_dataset.classes)
    plot_class_distribution(full_dataset, full_dataset.classes)

    # Criar o dataset personalizado com aumento de dados
    train_dataset = CustomDataset(full_dataset, transform=train_transforms)
    valid_dataset = CustomDataset(full_dataset, transform=test_transforms)
    test_dataset = CustomDataset(full_dataset, transform=test_transforms)

    # Dividir os índices para treino, validação e teste
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    train_end = int(train_split * dataset_size)
    valid_end = int((train_split + valid_split) * dataset_size)

    train_indices = indices[:train_end]
    valid_indices = indices[train_end:valid_end]
    test_indices = indices[valid_end:]

    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(valid_dataset, valid_indices)
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

    # Dataloaders
    g = torch.Generator()
    g.manual_seed(42)

    if use_weighted_loss:
        targets = [full_dataset.targets[i] for i in train_indices]
        class_counts = np.bincount(targets)
        class_counts = class_counts + 1e-6  # Para evitar divisão por zero
        class_weights = 1.0 / class_counts
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)

    # Carregar o modelo
    model = get_model(model_name, num_classes, dropout_p=0.5, fine_tune=fine_tune)
    if model is None:
        return None

    # Definir o otimizador com L2 regularization (weight_decay)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=l2_lambda)

    # Listas para armazenar as perdas e acurácias
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    # Early Stopping
    best_valid_loss = float('inf')
    epochs_no_improve = 0

    # Treinamento
    for epoch in range(epochs):
        set_seed(42 + epoch)
        running_loss = 0.0
        running_corrects = 0
        model.train()

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            try:
                outputs = model(inputs)
            except Exception as e:
                st.error(f"Erro durante o treinamento: {e}")
                return None

            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())

        # Validação
        model.eval()
        valid_running_loss = 0.0
        valid_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                valid_running_loss += loss.item() * inputs.size(0)
                valid_running_corrects += torch.sum(preds == labels.data)

        valid_epoch_loss = valid_running_loss / len(valid_dataset)
        valid_epoch_acc = valid_running_corrects.double() / len(valid_dataset)
        valid_losses.append(valid_epoch_loss)
        valid_accuracies.append(valid_epoch_acc.item())

        st.write(f'**Época {epoch+1}/{epochs}**')
        st.write(f'Perda de Treino: {epoch_loss:.4f} | Acurácia de Treino: {epoch_acc:.4f}')
        st.write(f'Perda de Validação: {valid_epoch_loss:.4f} | Acurácia de Validação: {valid_epoch_acc:.4f}')

        # Early Stopping
        if valid_epoch_loss < best_valid_loss:
            best_valid_loss = valid_epoch_loss
            epochs_no_improve = 0
            best_model_wts = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                st.write('Early stopping!')
                model.load_state_dict(best_model_wts)
                break

    # Carregar os melhores pesos do modelo
    model.load_state_dict(best_model_wts)

    # Gráficos de Perda e Acurácia
    plot_metrics(epochs, train_losses, valid_losses, train_accuracies, valid_accuracies)

    # Avaliação Final no Conjunto de Teste
    st.write("**Avaliação no Conjunto de Teste**")
    compute_metrics(model, test_loader, full_dataset.classes)

    # Análise de Erros
    st.write("**Análise de Erros**")
    error_analysis(model, test_loader, full_dataset.classes)

    # Liberar memória
    del train_loader, valid_loader
    gc.collect()

    return model, full_dataset.classes

def plot_metrics(epochs, train_losses, valid_losses, train_accuracies, valid_accuracies):
    """
    Plota os gráficos de perda e acurácia.
    """
    epochs_range = range(1, len(train_losses)+1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico de Perda
    ax[0].plot(epochs_range, train_losses, label='Treino')
    ax[0].plot(epochs_range, valid_losses, label='Validação')
    ax[0].set_title('Perda por Época')
    ax[0].set_xlabel('Épocas')
    ax[0].set_ylabel('Perda')
    ax[0].legend()

    # Gráfico de Acurácia
    ax[1].plot(epochs_range, train_accuracies, label='Treino')
    ax[1].plot(epochs_range, valid_accuracies, label='Validação')
    ax[1].set_title('Acurácia por Época')
    ax[1].set_xlabel('Épocas')
    ax[1].set_ylabel('Acurácia')
    ax[1].legend()

    st.pyplot(fig)

def compute_metrics(model, dataloader, classes):
    """
    Calcula métricas detalhadas e exibe matriz de confusão e relatório de classificação.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    # Relatório de Classificação
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    st.text("Relatório de Classificação:")
    st.write(pd.DataFrame(report).transpose())

    # Matriz de Confusão Normalizada
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predito')
    ax.set_ylabel('Verdadeiro')
    ax.set_title('Matriz de Confusão Normalizada')
    st.pyplot(fig)

    # Curva ROC
    if len(classes) == 2:
        fpr, tpr, thresholds = roc_curve(all_labels, [p[1] for p in all_probs])
        roc_auc = roc_auc_score(all_labels, [p[1] for p in all_probs])
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('Taxa de Falsos Positivos')
        ax.set_ylabel('Taxa de Verdadeiros Positivos')
        ax.set_title('Curva ROC')
        ax.legend(loc='lower right')
        st.pyplot(fig)
    else:
        # Multiclasse
        binarized_labels = label_binarize(all_labels, classes=range(len(classes)))
        roc_auc = roc_auc_score(binarized_labels, np.array(all_probs), average='weighted', multi_class='ovr')
        st.write(f"AUC-ROC Média Ponderada: {roc_auc:.4f}")

def error_analysis(model, dataloader, classes):
    """
    Realiza análise de erros mostrando algumas imagens mal classificadas.
    """
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            incorrect = preds != labels
            if incorrect.any():
                misclassified_images.extend(inputs[incorrect].cpu())
                misclassified_labels.extend(labels[incorrect].cpu())
                misclassified_preds.extend(preds[incorrect].cpu())
                if len(misclassified_images) >= 5:
                    break

    if misclassified_images:
        st.write("Algumas imagens mal classificadas:")
        fig, axes = plt.subplots(1, min(5, len(misclassified_images)), figsize=(15, 3))
        for i in range(min(5, len(misclassified_images))):
            image = misclassified_images[i]
            image = image.permute(1, 2, 0).numpy()
            axes[i].imshow(image)
            axes[i].set_title(f"V: {classes[misclassified_labels[i]]}\nP: {classes[misclassified_preds[i]]}")
            axes[i].axis('off')
        st.pyplot(fig)
    else:
        st.write("Nenhuma imagem mal classificada encontrada.")

def extract_features(dataset, model, batch_size):
    """
    Extrai características de um conjunto de dados usando um modelo pré-treinado.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker)

    features = []
    labels = []

    model.eval()
    with torch.no_grad():
        for inputs, lbls in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1)  # Flatten
            features.append(outputs.cpu().numpy())
            labels.extend(lbls.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.array(labels)
    return features, labels

def perform_clustering(features, num_clusters):
    """
    Aplica algoritmos de clustering às características.
    """
    # Clustering Hierárquico
    hierarchical = AgglomerativeClustering(n_clusters=num_clusters)
    hierarchical_labels = hierarchical.fit_predict(features)

    # K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(features)

    return hierarchical_labels, kmeans_labels

def evaluate_clustering(true_labels, cluster_labels, method_name):
    """
    Avalia os resultados do clustering comparando com as classes reais.
    """
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    st.write(f"**Métricas para {method_name}:**")
    st.write(f"Adjusted Rand Index: {ari:.4f}")
    st.write(f"Normalized Mutual Information Score: {nmi:.4f}")

def visualize_clusters(features, true_labels, hierarchical_labels, kmeans_labels, classes):
    """
    Visualiza os clusters usando redução de dimensionalidade e inclui as classes verdadeiras com nomes de rótulos.
    """
    # Redução de dimensionalidade com PCA para visualizar os clusters em 2D
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    # Mapear os rótulos verdadeiros para os nomes das classes
    true_labels_named = [classes[label] for label in true_labels]
    
    # Usar as cores distintas e visíveis para garantir que os clusters sejam claramente separados
    color_palette = sns.color_palette("tab10", len(set(true_labels)))

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))  # Agora temos 3 gráficos: Hierarchical, K-Means e classes verdadeiras

    # Clustering Hierárquico
    sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=hierarchical_labels, palette="deep", ax=axes[0], legend='full')
    axes[0].set_title('Clustering Hierárquico')
    ari_hierarchical = adjusted_rand_score(true_labels, hierarchical_labels)
    nmi_hierarchical = normalized_mutual_info_score(true_labels, hierarchical_labels)
    axes[0].text(0.1, 0.9, f"ARI: {ari_hierarchical:.2f}\nNMI: {nmi_hierarchical:.2f}", horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes, bbox=dict(facecolor='white', alpha=0.5))

    # K-Means Clustering
    sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=kmeans_labels, palette="deep", ax=axes[1], legend='full')
    axes[1].set_title('K-Means Clustering')
    ari_kmeans = adjusted_rand_score(true_labels, kmeans_labels)
    nmi_kmeans = normalized_mutual_info_score(true_labels, kmeans_labels)
    axes[1].text(0.1, 0.9, f"ARI: {ari_kmeans:.2f}\nNMI: {nmi_kmeans:.2f}", horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes, bbox=dict(facecolor='white', alpha=0.5))

    # Classes verdadeiras
    sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=true_labels_named, palette=color_palette, ax=axes[2], legend='full')
    axes[2].set_title('Classes Verdadeiras')

    # Exibir os gráficos
    st.pyplot(fig)

def evaluate_image(model, image, classes):
    """
    Avalia uma única imagem e retorna a classe predita e a confiança.
    """
    model.eval()
    image_tensor = test_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        class_idx = predicted.item()
        class_name = classes[class_idx]
        return class_name, confidence.item()

# Função para carregar o modelo de segmentação em PyTorch
@st.cache(allow_output_mutation=True)
def carregar_modelo_segmentacao():
    # Verificar se o arquivo existe
    if not os.path.exists('modelo_segmentacao.pth'):
        st.error("O arquivo 'modelo_segmentacao.pth' não foi encontrado. Certifique-se de que o modelo de segmentação foi treinado e o arquivo está no diretório correto.")
        return None
    # Substitua 'resnet18' pelo encoder que você usou durante o treinamento
    model = smp.Unet(
        encoder_name="resnet18",        # Escolha o encoder que você usou
        encoder_weights=None,           # Pesos pré-treinados não são necessários, pois o modelo já foi treinado
        in_channels=3,                  # Número de canais da imagem de entrada (RGB)
        classes=1,                      # Número de classes para segmentação (1 para segmentação binária)
    )
    # Carregar os pesos do modelo treinado
    model.load_state_dict(torch.load('modelo_segmentacao.pth', map_location=device))
    model.to(device)
    model.eval()
    return model

# Carregar o modelo de segmentação
modelo_segmentacao = carregar_modelo_segmentacao()
if modelo_segmentacao is None:
    st.stop()  # Interrompe a execução se o modelo não foi carregado

# Função para aplicar a máscara na imagem
def aplicar_mascara(image):
    # Pré-processar a imagem
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Obter a predição do modelo
    with torch.no_grad():
        output = modelo_segmentacao(img_tensor)
        # Aplicar função de ativação sigmoid para segmentação binária
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        # Converter a máscara predita para formato binário
        mask = (pred_mask > 0.5).astype(np.uint8)

    # Redimensionar a máscara para o tamanho original da imagem
    mask_resized = cv2.resize(mask, (image.width, image.height), interpolation=cv2.INTER_NEAREST)

    # Converter a imagem para array NumPy
    image_np = np.array(image)

    # Aplicar a máscara na imagem original
    res = cv2.bitwise_and(image_np, image_np, mask=mask_resized)

    return res

# Função para visualizar as ativações e exibir a imagem segmentada
def visualize_activations(model, image, class_names):
    """
    Visualiza as ativações na imagem usando Grad-CAM e exibe a imagem segmentada.
    """
    model.eval()  # Coloca o modelo em modo de avaliação
    input_tensor = test_transforms(image).unsqueeze(0).to(device)
    
    # Verificar se o modelo é suportado
    if isinstance(model, models.ResNet):
        target_layer = model.layer4[-1]
    elif isinstance(model, models.DenseNet):
        target_layer = model.features.denseblock4.denselayer16
    else:
        st.error("Modelo não suportado para Grad-CAM.")
        return
    
    # Criar o objeto CAM usando torchcam
    cam_extractor = SmoothGradCAMpp(model, target_layer=target_layer)
    
    # Habilitar gradientes explicitamente
    with torch.set_grad_enabled(True):
        out = model(input_tensor)  # Faz a previsão
        _, pred = torch.max(out, 1)  # Obtém a classe predita
        pred_class = pred.item()
    
    # Gerar o mapa de ativação
    activation_map = cam_extractor(pred_class, out)
    
    # Obter o mapa de ativação da primeira imagem no lote
    activation_map = activation_map[0].cpu().numpy()
    
    # Redimensionar o mapa de ativação para coincidir com o tamanho da imagem original
    activation_map_resized = cv2.resize(activation_map, (image.width, image.height))
    
    # Normalizar o mapa de ativação para o intervalo [0, 1]
    activation_map_resized = (activation_map_resized - activation_map_resized.min()) / (activation_map_resized.max() - activation_map_resized.min() + 1e-8)
    
    # Converter a imagem para array NumPy
    image_np = np.array(image)
    
    # Converter o mapa de ativação em uma imagem RGB
    heatmap = cv2.applyColorMap(np.uint8(activation_map_resized * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Sobrepor o mapa de ativação na imagem original
    superimposed_img = heatmap * 0.4 + image_np * 0.6
    superimposed_img = np.uint8(superimposed_img)
    
    # Aplicar a segmentação à imagem
    segmented_img = aplicar_mascara(image)
    
    # Exibir a imagem original, o Grad-CAM e a imagem segmentada
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # Imagem original
    ax[0].imshow(image_np)
    ax[0].set_title('Imagem Original')
    ax[0].axis('off')
    
    # Imagem com Grad-CAM
    ax[1].imshow(superimposed_img)
    ax[1].set_title('Grad-CAM')
    ax[1].axis('off')
    
    # Imagem segmentada
    ax[2].imshow(segmented_img)
    ax[2].set_title('Imagem Segmentada')
    ax[2].axis('off')
    
    # Exibir as imagens com o Streamlit
    st.pyplot(fig)

def main():

    # Definir o caminho do ícone
    icon_path = "logo.png"  # Verifique se o arquivo logo.png está no diretório correto
    
    # Verificar se o arquivo de ícone existe antes de configurá-lo
    if os.path.exists(icon_path):
        st.set_page_config(page_title="Geomaker +IA", page_icon=icon_path, layout="wide")
        logging.info(f"Ícone {icon_path} carregado com sucesso.")
    else:
        # Se o ícone não for encontrado, carrega sem favicon
        st.set_page_config(page_title="Geomaker +IA", layout="wide")
        logging.warning(f"Ícone {icon_path} não encontrado, carregando sem favicon.")
    
    # Layout da página
    if os.path.exists('capa.png'):
        st.image('capa.png', width=100, caption='Laboratório de Educação e Inteligência Artificial - Geomaker. "A melhor forma de prever o futuro é inventá-lo." - Alan Kay', use_column_width='always')
    else:
        st.warning("Imagem 'capa.png' não encontrada.")
    
    if os.path.exists("logo.png"):
        st.sidebar.image("logo.png", width=200)
    else:
        st.sidebar.text("Imagem do logotipo não encontrada.")
    
    # O restante do código da função main permanece inalterado
    # Aqui você deve incluir o restante do código da função main que você já possui,
    # incluindo as configurações da barra lateral, uploads de arquivos, etc.

    # Verificar se a soma dos splits é válida
    if train_split + valid_split > 0.95:
        st.sidebar.error("A soma dos splits de treinamento e validação deve ser menor ou igual a 0.95.")

    # Upload do arquivo ZIP
    zip_file = st.file_uploader("Upload do arquivo ZIP com as imagens", type=["zip"])

    if zip_file is not None and num_classes > 0 and train_split + valid_split <= 0.95:
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.read())
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        data_dir = temp_dir

        st.write("Iniciando o treinamento supervisionado...")
        model_data = train_model(data_dir, num_classes, model_name, fine_tune, epochs, learning_rate, batch_size, train_split, valid_split, use_weighted_loss, l2_lambda, patience)

        if model_data is None:
            st.error("Erro no treinamento do modelo.")
            shutil.rmtree(temp_dir)
            return

        model, classes = model_data
        st.success("Treinamento concluído!")

        # Extrair características usando o modelo pré-treinado (sem a camada final)
        st.write("Extraindo características para clustering...")
        # Remover a última camada do modelo para obter embeddings
        if model_name.startswith('ResNet'):
            feature_extractor = nn.Sequential(*list(model.children())[:-1])
        elif model_name.startswith('DenseNet'):
            feature_extractor = nn.Sequential(*list(model.features))
            feature_extractor.add_module('global_pool', nn.AdaptiveAvgPool2d((1,1)))
        else:
            st.error("Modelo não suportado para extração de características.")
            return

        feature_extractor = feature_extractor.to(device)
        feature_extractor.eval()

        # Carregar o dataset completo para extração de características
        full_dataset = datasets.ImageFolder(root=data_dir, transform=test_transforms)
        features, labels = extract_features(full_dataset, feature_extractor, batch_size)

        # Aplicar algoritmos de clustering
        st.write("Aplicando algoritmos de clustering...")
        features_reshaped = features.reshape(len(features), -1)
        hierarchical_labels, kmeans_labels = perform_clustering(features_reshaped, num_classes)

        # Avaliar e exibir os resultados
        st.write("Avaliando os resultados do clustering...")
        evaluate_clustering(labels, hierarchical_labels, "Clustering Hierárquico")
        evaluate_clustering(labels, kmeans_labels, "K-Means Clustering")

        # Visualizar clusters
        visualize_clusters(features_reshaped, labels, hierarchical_labels, kmeans_labels, classes)

        # Avaliação de uma imagem individual
        evaluate = st.radio("Deseja avaliar uma imagem?", ("Sim", "Não"))
        if evaluate == "Sim":
            eval_image_file = st.file_uploader("Faça upload da imagem para avaliação", type=["png", "jpg", "jpeg", "bmp", "gif"])
            if eval_image_file is not None:
                eval_image_file.seek(0)
                try:
                    eval_image = Image.open(eval_image_file).convert("RGB")
                except Exception as e:
                    st.error(f"Erro ao abrir a imagem: {e}")
                    return

                st.image(eval_image, caption='Imagem para avaliação', use_column_width=True)
                class_name, confidence = evaluate_image(model, eval_image, classes)
                st.write(f"**Classe Predita:** {class_name}")
                st.write(f"**Confiança:** {confidence:.4f}")

                # Visualizar ativações
                visualize_activations(model, eval_image, classes)

        # Limpar o diretório temporário
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
