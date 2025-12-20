import os
import zipfile
import shutil
import tempfile
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageEnhance
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
import logging
import base64
# Importações adicionais para Grad-CAM
from torchcam.methods import SmoothGradCAMpp, GradCAM, GradCAMpp, LayerCAM
from torchvision.transforms.functional import normalize, resize, to_pil_image
import cv2
# Importar otimizadores avançados
try:
    import torch_optimizer as optim_advanced
    ADVANCED_OPTIMIZERS_AVAILABLE = True
except ImportError:
    ADVANCED_OPTIMIZERS_AVAILABLE = False

# Import new modules
from visualization_3d import visualize_pca_3d, visualize_activation_heatmap_3d, create_interactive_3d_visualization
from ai_chat_module import AIAnalyzer, describe_gradcam_regions
from academic_references import AcademicReferenceFetcher, format_references_for_display
from genetic_interpreter import GeneticDiagnosticInterpreter
from multi_agent_system import ManagerAgent

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

# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def denormalize_image(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Denormaliza um tensor de imagem normalizado com valores ImageNet.
    
    Args:
        tensor: Tensor de imagem (C, H, W) ou array numpy (H, W, C)
        mean: Média usada na normalização
        std: Desvio padrão usado na normalização
    
    Returns:
        Array numpy (H, W, C) com valores no intervalo [0, 1]
    """
    if isinstance(tensor, torch.Tensor):
        # Convert tensor to numpy
        image = tensor.permute(1, 2, 0).cpu().numpy()
    else:
        image = tensor
    
    # Denormalize
    mean = np.array(mean)
    std = np.array(std)
    image = std * image + mean
    
    # Clip to valid range
    image = np.clip(image, 0, 1)
    
    return image

# Enhanced image preprocessing class
class EnhancedImagePreprocessor:
    """Classe para melhorar o tratamento de imagens antes do treinamento"""
    
    @staticmethod
    def enhance_image_quality(image):
        """Aplica melhorias de qualidade na imagem"""
        # Ajustar contraste
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # Ajustar nitidez
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        # Ajustar brilho levemente
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.05)
        
        return image

def get_augmentation_transforms(augmentation_type='standard'):
    """
    Retorna transformações de acordo com o tipo de aumento de dados
    
    Args:
        augmentation_type: 'none', 'standard', 'mixup', 'cutmix'
    """
    if augmentation_type == 'none':
        # Sem aumento de dados - apenas transformações básicas
        train_transform = transforms.Compose([
            transforms.Lambda(EnhancedImagePreprocessor.enhance_image_quality),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        # Standard ou base para mixup/cutmix
        train_transform = transforms.Compose([
            transforms.Lambda(EnhancedImagePreprocessor.enhance_image_quality),
            transforms.RandomApply([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=90),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomAffine(degrees=0, shear=10),
            ], p=0.5),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    
    return train_transform

# Transformações para validação e teste com normalização ImageNet
test_transforms = transforms.Compose([
    transforms.Lambda(EnhancedImagePreprocessor.enhance_image_quality),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Implementação de Mixup
def mixup_data(x, y, alpha=1.0):
    """Aplica Mixup ao batch de dados"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Calcula a loss para Mixup"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Implementação de CutMix
def cutmix_data(x, y, alpha=1.0):
    """Aplica CutMix ao batch de dados"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    # Gerar bbox
    W = x.size()[2]
    H = x.size()[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Centro do box
    cx = np.random.randint(0, W)
    cy = np.random.randint(0, H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x_cutmix = x.clone()
    x_cutmix[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # Ajustar lambda com a área real
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]
    return x_cutmix, y_a, y_b, lam

# Definir as transformações padrão para compatibilidade com código existente
train_transforms = get_augmentation_transforms('standard')

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
    st.write("### 📊 Visualização de algumas imagens do conjunto de dados original:")
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        idx = np.random.randint(len(dataset))
        image, label = dataset[idx]
        image = np.array(image)  # Converter a imagem PIL em array NumPy
        axes[i].imshow(image)
        axes[i].set_title(classes[label])
        axes[i].axis('off')
    st.pyplot(fig)
    plt.close(fig)

def plot_class_distribution(dataset, classes, title="Distribuição das Classes"):
    """
    Exibe a distribuição das classes no conjunto de dados e mostra os valores quantitativos.
    """
    # Extrair os rótulos das classes para todas as imagens no dataset
    labels = [label for _, label in dataset]
    
    # Contagem de cada classe
    class_counts = np.bincount(labels)
    
    # Plotar o gráfico com as contagens
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x=labels, hue=labels, ax=ax, palette="Set2", legend=False)
    
    # Adicionar os nomes das classes no eixo X
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    
    # Adicionar as contagens acima das barras
    for i, count in enumerate(class_counts):
        ax.text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
    
    ax.set_title(title)
    ax.set_xlabel("Classes")
    ax.set_ylabel("Número de Imagens")
    
    st.pyplot(fig)
    plt.close(fig)
    
    return class_counts

def show_augmented_images(dataset, transform, classes, num_augmentations=5):
    """
    Mostra imagens originais e suas versões aumentadas.
    """
    st.write("### 🔄 Exemplos de Imagens Aumentadas (Data Augmentation)")
    st.write("Cada linha mostra uma imagem original seguida de suas versões aumentadas:")
    
    # Selecionar 3 imagens aleatórias
    num_samples = 3
    for sample_idx in range(num_samples):
        idx = np.random.randint(len(dataset))
        original_image, label = dataset[idx]
        
        # Criar figura com 1 original + num_augmentations aumentadas
        fig, axes = plt.subplots(1, num_augmentations + 1, figsize=(15, 3))
        
        # Mostrar imagem original
        axes[0].imshow(np.array(original_image))
        axes[0].set_title(f'Original\n{classes[label]}')
        axes[0].axis('off')
        axes[0].set_facecolor('#e6f2ff')
        
        # Mostrar imagens aumentadas
        for i in range(1, num_augmentations + 1):
            augmented_image = transform(original_image)
            # Desnormalizar para visualização usando a função helper
            augmented_np = denormalize_image(augmented_image)
            
            axes[i].imshow(augmented_np)
            axes[i].set_title(f'Aumentada {i}')
            axes[i].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

def calculate_dataset_statistics(dataset, classes):
    """
    Calcula estatísticas do dataset incluindo média, desvio padrão, etc.
    """
    st.write("### 📈 Estatísticas do Dataset")
    
    # Contagem por classe
    labels = [label for _, label in dataset]
    class_counts = np.bincount(labels)
    
    # Criar dataframe com estatísticas
    stats_data = {
        'Classe': classes,
        'Quantidade': class_counts,
        'Percentual (%)': [f"{(count/len(dataset)*100):.2f}" for count in class_counts]
    }
    
    df_stats = pd.DataFrame(stats_data)
    
    st.write("#### Distribuição de Classes:")
    st.dataframe(df_stats, use_container_width=True)
    
    # Estatísticas gerais
    st.write("#### Estatísticas Gerais:")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Imagens", len(dataset))
    
    with col2:
        st.metric("Número de Classes", len(classes))
    
    with col3:
        st.metric("Imagens por Classe (Média)", f"{np.mean(class_counts):.1f}")
    
    with col4:
        st.metric("Desvio Padrão", f"{np.std(class_counts):.1f}")
    
    # Verificar balanceamento
    min_count = np.min(class_counts)
    max_count = np.max(class_counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    if imbalance_ratio > 1.5:
        st.warning(f"⚠️ Dataset desbalanceado detectado! Razão: {imbalance_ratio:.2f}x (Classe mais frequente / Classe menos frequente)")
        st.info("💡 Recomendação: Considere usar 'Perda Ponderada para Classes Desbalanceadas' nas configurações.")
    else:
        st.success(f"✅ Dataset relativamente balanceado. Razão: {imbalance_ratio:.2f}x")
    
    return df_stats

def visualize_pca_features(features, labels, classes, n_components=2):
    """
    Visualiza features usando PCA.
    """
    st.write(f"### 🔬 Análise PCA ({n_components} Componentes)")
    
    # Aplicar PCA
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features)
    
    # Mostrar variância explicada
    explained_var = pca.explained_variance_ratio_
    st.write(f"**Variância Explicada:** {explained_var[0]*100:.2f}% (PC1), {explained_var[1]*100:.2f}% (PC2)")
    st.write(f"**Variância Total Explicada:** {sum(explained_var)*100:.2f}%")
    
    # Criar visualização
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Mapear labels para nomes de classes
    labels_named = [classes[label] for label in labels]
    
    # Criar scatter plot
    scatter = sns.scatterplot(
        x=features_pca[:, 0], 
        y=features_pca[:, 1], 
        hue=labels_named,
        palette="tab10",
        ax=ax,
        s=100,
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )
    
    ax.set_xlabel(f'Componente Principal 1 ({explained_var[0]*100:.1f}%)')
    ax.set_ylabel(f'Componente Principal 2 ({explained_var[1]*100:.1f}%)')
    ax.set_title('Visualização PCA das Features Extraídas')
    ax.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    return features_pca, explained_var

def get_model(model_name, num_classes, dropout_p=0.5, fine_tune=False):
    """
    Retorna o modelo pré-treinado selecionado.
    """
    if model_name == 'ResNet18':
        model = models.resnet18(weights='DEFAULT')
    elif model_name == 'ResNet50':
        model = models.resnet50(weights='DEFAULT')
    elif model_name == 'DenseNet121':
        model = models.densenet121(weights='DEFAULT')
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
        # Ensure final layer requires grad
        for param in model.fc.parameters():
            param.requires_grad = True
    elif model_name.startswith('DenseNet'):
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(num_ftrs, num_classes)
        )
        # Ensure final layer requires grad
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        st.error("Modelo não suportado.")
        return None

    model = model.to(device)
    return model

def train_model(data_dir, num_classes, model_name, fine_tune, epochs, learning_rate, batch_size, train_split, valid_split, use_weighted_loss, l2_lambda, l1_lambda, patience, optimizer_name='Adam', scheduler_name='None', augmentation_type='standard'):
    """
    Função principal para treinamento do modelo.
    
    Args:
        data_dir: Diretório com os dados
        num_classes: Número de classes
        model_name: Nome do modelo
        fine_tune: Se deve fazer fine-tuning completo
        epochs: Número de épocas
        learning_rate: Taxa de aprendizagem
        batch_size: Tamanho do lote
        train_split: Proporção de treino
        valid_split: Proporção de validação
        use_weighted_loss: Se deve usar perda ponderada
        l2_lambda: Regularização L2 (weight decay)
        l1_lambda: Regularização L1
        patience: Paciência para early stopping
        optimizer_name: Nome do otimizador (Adam, AdamW, SGD, Ranger, Lion)
        scheduler_name: Nome do scheduler (None, CosineAnnealingLR, OneCycleLR)
        augmentation_type: Tipo de aumento de dados (none, standard, mixup, cutmix)
    
    Returns:
        tuple: (model, classes) ou None em caso de erro
    """
    set_seed(42)

    # Carregar o dataset original sem transformações
    full_dataset = datasets.ImageFolder(root=data_dir)
    
    # ========== CONTAGEM INICIAL DOS DADOS ==========
    st.write("## 📊 ANÁLISE INICIAL DO DATASET")
    st.write(f"### 🔢 **Contagem Inicial: {len(full_dataset)} imagens**")
    
    # Exibir estatísticas detalhadas
    stats_df = calculate_dataset_statistics(full_dataset, full_dataset.classes)
    
    # Exibir algumas imagens do dataset original
    visualize_data(full_dataset, full_dataset.classes)
    
    # Plotar distribuição inicial
    st.write("### 📊 Distribuição Inicial das Classes")
    initial_class_counts = plot_class_distribution(full_dataset, full_dataset.classes, 
                                                    title="Distribuição INICIAL das Classes (Sem Aumento de Dados)")

    # ========== TÉCNICA DE AUMENTO DE DADOS ==========
    st.write("---")
    st.write("## 🔄 APLICAÇÃO DA TÉCNICA DE AUMENTO DE DADOS")
    st.write(f"**Técnica Selecionada:** `{augmentation_type}`")
    
    if augmentation_type == 'none':
        st.info("ℹ️ Nenhuma técnica de aumento de dados foi selecionada. As imagens serão usadas como estão.")
    elif augmentation_type == 'standard':
        st.info("ℹ️ Técnica Standard: Aplicação de transformações aleatórias (rotação, flip, crop, jitter, etc.)")
    elif augmentation_type == 'mixup':
        st.info("ℹ️ Técnica Mixup: Mistura linear de pares de imagens e seus rótulos")
    elif augmentation_type == 'cutmix':
        st.info("ℹ️ Técnica CutMix: Recorte e colagem de regiões entre imagens diferentes")
    
    # Obter transformações baseadas no tipo de augmentação
    train_transform = get_augmentation_transforms(augmentation_type)
    
    # Mostrar exemplos de imagens aumentadas
    if augmentation_type != 'none':
        show_augmented_images(full_dataset, train_transform, full_dataset.classes, num_augmentations=4)
    
    # ========== ESTIMATIVA APÓS AUMENTO ==========
    st.write("---")
    st.write("## 📈 ESTIMATIVA APÓS AUMENTO DE DADOS")
    
    # Calcular estimativa de imagens após aumento
    # Durante o treinamento, cada época gera versões aumentadas
    if augmentation_type == 'none':
        augmentation_multiplier = 1
        st.write(f"### 🔢 **Total Estimado: {len(full_dataset)} imagens** (sem aumento)")
    else:
        # Com augmentation, cada época gera versões diferentes
        # Estimativa conservadora: cada imagem pode gerar de 3-5 variações por época
        augmentation_multiplier = 4  # Média estimada
        total_estimated = len(full_dataset) * augmentation_multiplier * epochs
        st.write(f"### 🔢 **Total de Imagens Original: {len(full_dataset)}**")
        st.write(f"### 🔢 **Multiplicador Estimado por Época: ~{augmentation_multiplier}x**")
        st.write(f"### 🔢 **Total Estimado Durante {epochs} Épocas: ~{total_estimated:,} imagens aumentadas**")
        st.info(f"💡 **Explicação:** Durante o treinamento, cada uma das {len(full_dataset)} imagens originais será " +
                f"transformada aleatoriamente a cada época, gerando aproximadamente {augmentation_multiplier}x variações únicas " +
                f"ao longo de {epochs} épocas, totalizando cerca de {total_estimated:,} imagens processadas.")
    
    st.write("---")
    
    # Criar o dataset personalizado com aumento de dados
    train_dataset = CustomDataset(full_dataset, transform=train_transform)
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
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(trainable_params, lr=learning_rate, weight_decay=l2_lambda)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=l2_lambda)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(trainable_params, lr=learning_rate, weight_decay=l2_lambda, momentum=0.9, nesterov=True)
    elif optimizer_name == 'Ranger' and ADVANCED_OPTIMIZERS_AVAILABLE:
        optimizer = optim_advanced.Ranger(trainable_params, lr=learning_rate, weight_decay=l2_lambda)
    elif optimizer_name == 'Lion' and ADVANCED_OPTIMIZERS_AVAILABLE:
        optimizer = optim_advanced.Lion(trainable_params, lr=learning_rate, weight_decay=l2_lambda)
    else:
        st.warning(f"Otimizador {optimizer_name} não disponível. Usando Adam.")
        optimizer = optim.Adam(trainable_params, lr=learning_rate, weight_decay=l2_lambda)
    
    # Configurar Learning Rate Scheduler
    scheduler = None
    if scheduler_name == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate/100)
    elif scheduler_name == 'OneCycleLR':
        steps_per_epoch = len(train_loader)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=learning_rate*10, 
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3
        )

    # Listas para armazenar as perdas e acurácias
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    # Early Stopping
    best_valid_loss = float('inf')
    epochs_no_improve = 0
    
    # Parâmetros para Mixup e CutMix
    use_mixup = (augmentation_type == 'mixup')
    use_cutmix = (augmentation_type == 'cutmix')
    mixup_alpha = 1.0
    cutmix_alpha = 1.0
    
    # Cache de parâmetros para regularização L1 (otimização)
    trainable_params_list = list(filter(lambda p: p.requires_grad, model.parameters())) if l1_lambda > 0 else []

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
            
            # Aplicar Mixup ou CutMix se selecionado
            if use_mixup:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, mixup_alpha)
            elif use_cutmix:
                inputs, labels_a, labels_b, lam = cutmix_data(inputs, labels, cutmix_alpha)
            
            try:
                outputs = model(inputs)
            except Exception as e:
                st.error(f"Erro durante o treinamento: {e}")
                return None

            # Calcular loss
            if use_mixup or use_cutmix:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                _, preds = torch.max(outputs, 1)
            else:
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            # Adicionar regularização L1 se configurado
            if l1_lambda > 0:
                l1_reg = torch.tensor(0., device=device)
                for param in trainable_params_list:
                    l1_reg += torch.norm(param, 1)
                loss = loss + l1_lambda * l1_reg
            
            loss.backward()
            optimizer.step()
            
            # Atualizar scheduler OneCycleLR a cada batch
            if scheduler_name == 'OneCycleLR' and scheduler is not None:
                scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data if not (use_mixup or use_cutmix) else preds == labels_a.data)

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

        # Atualizar scheduler CosineAnnealingLR após cada época
        if scheduler_name == 'CosineAnnealingLR' and scheduler is not None:
            scheduler.step()

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
    plt.close(fig)

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
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True, zero_division=0)
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
    plt.close(fig)

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
        plt.close(fig)
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
        num_images = min(5, len(misclassified_images))
        fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
        
        # Handle case when only one image (axes is not an array)
        if num_images == 1:
            axes = [axes]
            
        for i in range(num_images):
            image = misclassified_images[i]
            # Denormalize the image for proper display
            image = denormalize_image(image)
            axes[i].imshow(image)
            axes[i].set_title(f"V: {classes[misclassified_labels[i]]}\nP: {classes[misclassified_preds[i]]}")
            axes[i].axis('off')
        st.pyplot(fig)
        plt.close(fig)
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
    plt.close(fig)

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

#________________________________________________

#________________________________________________

def visualize_activations(model, image, class_names, gradcam_type='SmoothGradCAMpp'):
    """
    Visualiza as ativações na imagem usando diferentes variantes de Grad-CAM.
    
    Args:
        model: Modelo treinado
        image: Imagem PIL
        class_names: Lista de nomes das classes
        gradcam_type: Tipo de Grad-CAM ('GradCAM', 'GradCAMpp', 'SmoothGradCAMpp', 'LayerCAM')
    
    Returns:
        activation_map_resized: Mapa de ativação normalizado
    """
    try:
        # Ensure model is in eval mode and enable gradients for Grad-CAM
        model.eval()
        
        # Enable requires_grad for necessary layers
        for param in model.parameters():
            param.requires_grad = True
        
        input_tensor = test_transforms(image).unsqueeze(0).to(device)
        input_tensor.requires_grad = True
        
        # Verificar se o modelo é suportado
        model_type = type(model).__name__
        if 'ResNet' in model_type:
            target_layer = model.layer4[-1]
        elif 'DenseNet' in model_type:
            target_layer = model.features.denseblock4.denselayer16
        else:
            st.error("Modelo não suportado para Grad-CAM.")
            return None
        
        # Criar o objeto CAM usando torchcam
        if gradcam_type == 'GradCAM':
            cam_extractor = GradCAM(model, target_layer=target_layer)
        elif gradcam_type == 'GradCAMpp':
            cam_extractor = GradCAMpp(model, target_layer=target_layer)
        elif gradcam_type == 'SmoothGradCAMpp':
            cam_extractor = SmoothGradCAMpp(model, target_layer=target_layer)
        elif gradcam_type == 'LayerCAM':
            cam_extractor = LayerCAM(model, target_layer=target_layer)
        else:
            st.error(f"Tipo de Grad-CAM não suportado: {gradcam_type}")
            return None
        
        # Habilitar gradientes explicitamente
        with torch.set_grad_enabled(True):
            out = model(input_tensor)  # Faz a previsão
            _, pred = torch.max(out, 1)  # Obtém a classe predita
            pred_class = pred.item()
        
        # Gerar o mapa de ativação
        activation_map = cam_extractor(pred_class, out)
        
        # Obter o mapa de ativação da primeira imagem no lote
        activation_map = activation_map[0].cpu().detach().numpy()
        
        # Redimensionar o mapa de ativação para coincidir com o tamanho da imagem original
        activation_map_resized = cv2.resize(activation_map, (image.size[0], image.size[1]))
        
        # Normalizar o mapa de ativação para o intervalo [0, 1]
        activation_map_resized = (activation_map_resized - activation_map_resized.min()) / (activation_map_resized.max() - activation_map_resized.min() + 1e-8)
        
        # Converter a imagem para array NumPy
        image_np = np.array(image)
        
        # Converter o mapa de ativação em uma imagem RGB
        heatmap = cv2.applyColorMap(np.uint8(255 * activation_map_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Sobrepor o mapa de ativação na imagem original
        superimposed_img = heatmap * 0.4 + image_np * 0.6
        superimposed_img = np.uint8(superimposed_img)
        
        # Exibir a imagem original e o mapa de ativação sobreposto
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        
        # Imagem original
        ax[0].imshow(image_np)
        ax[0].set_title('Imagem Original')
        ax[0].axis('off')
        
        # Imagem com Grad-CAM
        ax[1].imshow(superimposed_img)
        ax[1].set_title(f'{gradcam_type}')
        ax[1].axis('off')
        
        # Exibir as imagens com o Streamlit
        st.pyplot(fig)
        plt.close(fig)
        
        return activation_map_resized
        
    except Exception as e:
        st.error(f"Erro ao gerar Grad-CAM: {str(e)}")
        st.info("Visualização Grad-CAM não disponível para este modelo/configuração.")
        return None




def main():

    # Definir o caminho do ícone
    icon_path = "logo.png"  # Verifique se o arquivo logo.png está no diretório correto
    
    # Verificar se o arquivo de ícone existe antes de configurá-lo
    if os.path.exists(icon_path):
        st.set_page_config(page_title="Geomaker", page_icon=icon_path, layout="wide")
        logging.info(f"Ícone {icon_path} carregado com sucesso.")
    else:
        # Se o ícone não for encontrado, carrega sem favicon
        st.set_page_config(page_title="Geomaker", layout="wide")
        logging.warning(f"Ícone {icon_path} não encontrado, carregando sem favicon.")
    
    # Layout da página
    if os.path.exists('capa.png'):
        st.image('capa.png', caption='Laboratório de Educação e Inteligência Artificial - Geomaker. "A melhor forma de prever o futuro é inventá-lo." - Alan Kay', width='stretch')
    else:
        st.warning("Imagem 'capa.png' não encontrada.")
    
    if os.path.exists("logo.png"):
        st.sidebar.image("logo.png", width=200)
    else:
        st.sidebar.text("Imagem do logotipo não encontrada.")
    
    
  #___________________________________________________________
    st.title("Classificação por Imagens com Aprendizado Profundo")
    st.write("Este aplicativo permite treinar um modelo de classificação de imagens e aplicar algoritmos de clustering para análise comparativa.")
    with st.expander("Transformações de Dados e Aumento de Dados no Treinamento de Redes Neurais"):
        st.write("""
        As **transformações de dados** e o **aumento de dados** são técnicas essenciais no treinamento de redes neurais profundas, principalmente em tarefas de visão computacional. 
        Essas abordagens buscam melhorar a capacidade de generalização dos modelos, gerando **imagens sintéticas** a partir dos dados de treinamento. Tais técnicas são particularmente 
        valiosas quando o conjunto de dados disponível é pequeno ou apresenta pouca diversidade. A normalização, por sua vez, assegura que os valores dos pixels estejam em uma escala adequada, 
        resultando em um treinamento mais estável e eficiente. Diversos estudos apontam que essas práticas são eficazes para evitar **overfitting** e aumentar a robustez do modelo 
        (Shorten & Khoshgoftaar, 2019).
        """)
    
        st.write("### Aumento de Dados no Treinamento")
    
        st.write("""
        O **aumento de dados** ou *data augmentation* consiste na aplicação de transformações aleatórias às imagens do conjunto de treinamento para gerar novas amostras sintéticas. 
        No código implementado, essa técnica é realizada com a classe `transforms.Compose` da biblioteca **torchvision**, que aplica uma sequência de transformações.
        """)
    
        st.write("#### Transformações Aplicadas no Treinamento")
        
        st.write("""
        1. **RandomApply**: Aplica aleatoriamente um conjunto de transformações com 50% de probabilidade. Esse procedimento aumenta a variabilidade dos dados, gerando imagens diferentes a partir de uma única imagem de entrada.
       
        2. **RandomHorizontalFlip**: Realiza a inversão horizontal da imagem com 50% de probabilidade. Isso é útil em cenários onde a orientação horizontal da imagem não altera seu significado, como em imagens de rochas ou melanomas.
    
        3. **RandomRotation(degrees=90)**: Rotaciona a imagem em até 90 graus, criando variações angulares, o que ajuda o modelo a reconhecer objetos independentemente da orientação.
    
        4. **ColorJitter**: Introduz variações de brilho, contraste, saturação e matiz, simulando diferentes condições de iluminação e tornando o modelo mais robusto a mudanças de iluminação.
    
        5. **RandomResizedCrop(224, scale=(0.8, 1.0))**: Realiza cortes aleatórios na imagem e os redimensiona para 224x224 pixels, permitindo que diferentes partes da imagem sejam enfatizadas.
    
        6. **RandomAffine(degrees=0, shear=10)**: Aplica transformações afins, como cisalhamento, simulando distorções que podem ocorrer no mundo real, como mudanças de perspectiva.
    
        7. **Resize(256)**: Redimensiona a imagem para 256x256 pixels, assegurando que todas as imagens possuam a mesma dimensão.
    
        8. **CenterCrop(224)**: Recorta o centro da imagem, garantindo que o tamanho final seja 224x224 pixels.
    
        9. **ToTensor**: Converte a imagem para um tensor PyTorch, normalizando os valores dos pixels para o intervalo de [0,1], facilitando o processamento pelo modelo.
        """)
    
        st.write("### Geração de Imagens Sintéticas")
    
        st.write("""
        Essas transformações permitem que cada imagem original gere até **5 a 10 imagens sintéticas**. Por exemplo, em um conjunto de dados de 1000 imagens, 
        o processo pode expandir o conjunto para **5000 a 10000 imagens** ao longo do treinamento. Essa ampliação artificial do conjunto de dados reduz o risco de **overfitting**, 
        permitindo que o modelo treine em um conjunto "maior" e mais diverso, o que é crucial para melhorar a generalização do modelo em dados novos.
        """)
    
        st.write("### Normalização nas Imagens de Teste e Validação")
    
        st.write("""
        Nas imagens de **teste** e **validação**, o aumento de dados não é aplicado. O objetivo nesses conjuntos é avaliar o modelo de maneira consistente, 
        utilizando imagens que representem o mais fielmente possível os dados reais. No entanto, a normalização dessas imagens é fundamental para assegurar que seus valores de pixel 
        estejam adequados para as operações de aprendizado. Isso também garante um desempenho estável durante o treinamento.
        """)
    
        st.write("#### Transformações Aplicadas no Teste e Validação")
        
        st.write("""
        1. **Resize(256)**: Redimensiona a imagem para 256x256 pixels, garantindo que todas as imagens tenham o mesmo tamanho inicial.
    
        2. **CenterCrop(224)**: Realiza o corte central para que as dimensões da imagem sejam 224x224 pixels, correspondendo ao tamanho esperado pelo modelo.
    
        3. **ToTensor**: Converte a imagem para tensor e normaliza os valores dos pixels para o intervalo de [0,1], o que melhora a estabilidade numérica e a taxa de convergência do treinamento.
        """)
    
        st.write("### Importância da Normalização")
    
        st.write("""
        A **normalização** garante que os valores dos pixels estejam em uma escala apropriada para as operações aritméticas realizadas no modelo, melhorando a estabilidade e o desempenho do processo de treinamento. 
        Ela também contribui para a estabilidade numérica durante o cálculo do gradiente e para uma convergência mais eficiente do modelo (Nguyễn et al., 2021).
        """)
    
        st.write("### Conclusão")
    
        st.write("""
        O código exemplifica a implementação eficaz de transformações de dados e aumento de dados como parte da pipeline de treinamento de redes neurais profundas. 
        As transformações aplicadas aumentam a diversidade do conjunto de treinamento, ajudando a mitigar o **overfitting** e melhorar a generalização do modelo. 
        Além disso, a normalização aplicada aos dados de teste e validação garante que o desempenho do modelo seja avaliado de forma precisa e consistente, 
        alinhada às melhores práticas de aprendizado profundo.
        """)
    
        st.write("### Referências")
        
        st.write("""
        - Huang, G., Liu, Z., Maaten, L., & Weinberger, K. (2017). Densely connected convolutional networks. https://doi.org/10.1109/cvpr.2017.243
        - Li, S. (2023). Clouddensenet: lightweight ground-based cloud classification method for large-scale datasets based on reconstructed densenet. *Sensors*, 23(18), 7957. https://doi.org/10.3390/s23187957
        - Nguyễn, H., Yu, G., Shin, N., Kwon, G., Kwak, W., & Kim, J. (2021). Defective product classification system for smart factory based on deep learning. *Electronics*, 10(7), 826. https://doi.org/10.3390/electronics10070826
        - Shorten, C. & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. *Journal of Big Data*, 6(1). https://doi.org/10.1186/s40537-019-0197-0
        """)

    # Barra Lateral de Configurações
    st.sidebar.title("Configurações do Treinamento")
      # Imagem e Contatos___________________________
    #_______________________________________________________________________________________
    # Sidebar com o conteúdo explicativo e fórmulas LaTeX
    with st.sidebar:
        with st.expander("Discussão sobre o Número de Classes em Modelos de Aprendizado Profundo"):
            st.write("""
            ### Introdução
    
            A discussão sobre o número de classes em modelos de aprendizado profundo é fundamental para a compreensão da arquitetura e do desempenho de redes neurais em tarefas de classificação. O número de classes refere-se ao total de categorias ou rótulos que um modelo deve prever, e a configuração correta desse parâmetro impacta diretamente o desempenho do modelo, pois afeta a dimensão da saída da rede neural e a complexidade da tarefa. O número de classes pode variar de tarefas binárias, que envolvem apenas duas classes, até problemas com centenas ou milhares de classes, como nas classificações de imagens do **ImageNet** (Cheng, 2023).
            """)
    
            st.write("### Impacto do Número de Classes")
            st.write("""
            O número de classes define a estrutura da última camada da rede neural, que é responsável por realizar as predições. Para um problema de **classificação binária**, o modelo terá uma única saída que prevê a probabilidade de uma classe ou outra. Em contrapartida, em um problema de **classificação multiclasse**, o número de saídas será igual ao número de categorias possíveis (Cheng, 2023). A função de ativação utilizada na última camada é crucial para a interpretação dos resultados. A equação que representa essa relação pode ser expressa como:
            """)
            st.latex(r'''
            \mathbf{y} = \text{Softmax}(Wx + b)
            ''')
    
            st.write("""
            onde **W** e **b** são os pesos e o bias, respectivamente, que conectam a camada anterior às classes de saída. O resultado é passado pela função **softmax**, que converte os valores em probabilidades associadas a cada classe (Petrovska et al., 2020).
            """)
    
                       
            st.write("""
            Em tarefas de classificação binária, o modelo tem apenas duas classes possíveis, como **detecção de fraude** ou **diagnóstico de doenças** (positivo ou negativo). Nesse caso, a função de ativação final é geralmente a **sigmoide**, que retorna uma probabilidade entre 0 e 1 para cada entrada. Um limiar é então aplicado para decidir a classe final predita pelo modelo (Cheng, 2023).
            """)
    
            st.write("### Classificação Multiclasse")
            st.write("""
            Em problemas de classificação multiclasse, o número de classes pode variar consideravelmente. Por exemplo, em tarefas de **classificação de imagens geológicas**, o número de classes pode ser pequeno, mas em aplicações como a **classificação de imagens médicas** ou **reconhecimento facial**, o número de classes pode ser muito maior. A arquitetura da rede deve ser ajustada para garantir que a última camada tenha o número correto de saídas correspondente ao número de categorias (Cheng, 2023; Sardeshmukh, 2023).
            """)
    
            st.write("### Classificação Multirrótulo")
            st.write("""
            Em problemas de **classificação multirrótulo**, uma entrada pode pertencer a mais de uma classe ao mesmo tempo. Nesse cenário, o número de saídas da rede neural é igual ao número de classes possíveis, mas cada saída é independente das demais. A função de ativação usada é a **sigmoide**, pois ela calcula a probabilidade de cada classe independentemente das outras (Petrovska et al., 2020).
            """)
    
            st.write("### Efeitos do Número de Classes no Desempenho")
            st.write("""
            O número de classes influencia diretamente a complexidade do modelo e o tempo de treinamento. Conforme o número de classes aumenta, a tarefa de classificação se torna mais difícil, exigindo mais parâmetros e tempo de computação. Além disso, um maior número de classes aumenta o risco de **sobreajuste** (overfitting), especialmente em conjuntos de dados pequenos (Cheng, 2023; Suhana, 2022).
            """)
    
            st.write("### Conclusão")
            st.write("""
            O número de classes é um fator determinante na definição da arquitetura de redes neurais para tarefas de classificação. Seja em problemas binários, multiclasse ou multirrótulo, a escolha adequada desse parâmetro garante que a rede neural seja capaz de aprender as características relevantes de cada categoria. Em problemas com muitas classes, estratégias como a **regularização** e o **data augmentation** podem ser utilizadas para melhorar o desempenho do modelo, evitando o sobreajuste (Cheng, 2023; Sardeshmukh, 2023).
            """)
    
            st.write("### Referências")
          
            st.write("""
            1. Cheng, R. (2023). Expansion of the CT-scans image set based on the pretrained DCGAN for improving the performance of the CNN. *Journal of Physics Conference Series*, 2646(1), 012015. https://doi.org/10.1088/1742-6596/2646/1/012015
            2. Petrovska, B., Atanasova-Pacemska, T., Corizzo, R., Mignone, P., Lameski, P., & Zdravevski, E. (2020). Aerial Scene Classification through Fine-Tuning with Adaptive Learning Rates and Label Smoothing. *Applied Sciences*, 10(17), 5792. https://doi.org/10.3390/app10175792
            3. Sardeshmukh, M. (2023). Crop image classification using convolutional neural network. *Multidisciplinary Science Journal*, 5(4), 2023039. https://doi.org/10.31893/multiscience.2023039
            4. Suhana, R. (2022). Fish Image Classification Using Adaptive Learning Rate In Transfer Learning Method. *Knowledge Engineering and Data Science*, 5(1), 67-77. https://doi.org/10.17977/um018v5i12022p67-77
            """)

  
    # Nota: O número de classes será detectado automaticamente do dataset
    num_classes = st.sidebar.number_input("Número de Classes (será detectado automaticamente):", min_value=1, step=1, value=2, disabled=True, help="Este valor será automaticamente detectado do dataset após o upload")
    #_______________________________________________________________________________________
    # Sidebar com o conteúdo explicativo e fórmula LaTeX
    with st.sidebar:
        with st.expander("Modelos Pré-Treinados: ResNet18, ResNet50 e DenseNet121:"):
            st.write("""
            ### Introdução
        
            As redes neurais convolucionais (CNNs) têm se tornado uma ferramenta essencial no campo do aprendizado profundo, especialmente em tarefas de visão computacional, como a classificação de imagens. 
            Modelos como **ResNet18**, **ResNet50** e **DenseNet121** são amplamente reconhecidos por seu desempenho superior em competições de classificação de imagens, como o **ImageNet**. Esses modelos são considerados 
            **pré-treinados**, pois foram inicialmente treinados em grandes conjuntos de dados, permitindo que sejam reutilizados e ajustados para novas tarefas específicas, uma prática conhecida como **transferência de aprendizado** 
            (Cheng, 2023; Petrovska et al., 2020; Alaoui, 2023).
            """)
        
            st.write("### ResNet18 e ResNet50")
            st.write("""
            A arquitetura **ResNet** (Rede Residual) foi desenvolvida para mitigar o problema de **degradação** que ocorre em redes neurais muito profundas, onde o aumento do número de camadas pode levar a uma diminuição no desempenho.
            A inovação dos **blocos residuais** permite que algumas camadas "saltem" conexões, aprendendo uma **função de identidade** em vez de novas representações para cada camada. Essa abordagem facilita o treinamento de redes mais profundas, pois a função residual pode ser aprendida de forma mais eficiente (Zhang et al., 2018; Sandotra et al., 2023; Petrovska et al., 2020).
            """)
            
            st.latex(r'''
            \mathbf{y} = \mathcal{F}(x, \{W_i\}) + x
            ''')
            
            st.write("""
            onde 
            """)
            st.latex(r'''
            \mathcal{F}(x, \{W_i\}) + x
            ''')
          
            st.write("""
            representa a função aprendida e x é a entrada. O termo x é adicionado à saída, o que simplifica o processo de treinamento e permite que redes mais profundas sejam treinadas com maior eficácia 
            ("A Framework for Flood Extent Mapping using CNN Transfer Learning", 2022; Petrovska et al., 2020).
            """)
        
            st.write("""
            O modelo **ResNet18** possui 18 camadas treináveis e é uma versão mais leve, adequada para aplicações com restrições de recursos computacionais, enquanto o **ResNet50**, com 50 camadas, é capaz de capturar padrões mais complexos em imagens, sendo ideal para tarefas que exigem maior profundidade de análise (Sandotra et al., 2023; Qin et al., 2019; Petrovska et al., 2020).
            """)
        
            st.write("""
            Ambos os modelos foram pré-treinados no conjunto de dados **ImageNet**, o que facilita a **transferência de aprendizado** em novos domínios. As camadas iniciais desses modelos já são capazes de identificar características gerais, acelerando o processo de treinamento em conjuntos de dados menores e específicos, como em aplicações médicas ou de classificação de imagens geológicas (Cheng, 2023; Petrovska et al., 2020; Alaoui, 2023).
            """)
        
            st.write("### DenseNet121")
            st.write("""
            A arquitetura **DenseNet** (Rede Convolucional Densamente Conectada) oferece uma abordagem alternativa, onde todas as camadas estão interconectadas, promovendo a preservação do fluxo de gradiente e da informação original. Isso facilita a reutilização das representações intermediárias e otimiza a eficiência do modelo. A equação que expressa essa estrutura é:
            """)
        
            st.latex(r'''
            \mathbf{x}_l = H_l(\mathbf{x}_0, \mathbf{x}_1, \dots, \mathbf{x}_{l-1})
            ''')
        
            st.write("""
            onde
            """)
          
            st.latex(r'''
            \mathbf{x}_l 
            ''')
          
            st.write("""
            é a saída da l-ésima camada e 
            """)
          
            st.latex(r'''
             \mathbf{H}_l
            ''')
          
            st.write("""
            é a função aplicada. Essa configuração otimiza o uso de gradientes e representações, resultando em um desempenho superior em tarefas de classificação 
            (Benegui & Ionescu, 2020; Varshni et al., 2019; Hamdaoui et al., 2021).
            """)
        
            st.write("""
            O modelo **DenseNet121**, que possui 121 camadas treináveis, é particularmente eficaz em contextos onde a eficiência é crucial, maximizando o uso de recursos computacionais e facilitando a extração de características relevantes de imagens (Sardeshmukh, 2023; Hamdaoui et al., 2021).
            """)
        
            st.write("### Transferência de Aprendizado e Ajuste Fino")
            st.write("""
            A utilização de modelos pré-treinados, como ResNet18, ResNet50 e DenseNet121, é uma técnica de **transferência de aprendizado** que permite que o conhecimento adquirido em tarefas anteriores seja aplicado a novos problemas. 
            Em vez de treinar um modelo do zero, o ajuste fino é realizado nas camadas do modelo para se adaptar a um novo conjunto de dados, permitindo que características específicas sejam aprendidas de forma mais eficiente. Por exemplo, em aplicações de **classificação de melanomas** ou **análise de rochas vulcânicas**, as camadas mais profundas dos modelos são ajustadas para entender características específicas de imagens médicas ou geológicas (Suhana, 2022; Petrovska et al., 2020).
            """)
        
            st.write("""
            Estudos demonstram que a transferência de aprendizado é especialmente eficaz ao se trabalhar com conjuntos de dados pequenos. O uso de modelos pré-treinados pode proporcionar resultados semelhantes ou até superiores aos de modelos treinados a partir do zero, reduzindo o tempo de treinamento e melhorando a precisão (Raghava et al., 2019; Alaoui, 2023; Ahmed, 2021).
            """)
        
            st.write("### Conclusão")
            st.write("""
            As arquiteturas **ResNet18**, **ResNet50** e **DenseNet121** são ferramentas poderosas no campo do aprendizado profundo, especialmente em tarefas de classificação de imagens. Seu pré-treinamento em grandes conjuntos de dados, como o **ImageNet**, e a capacidade de serem ajustados para novas tarefas através da transferência de aprendizado, tornam esses modelos ideais para uma ampla gama de aplicações, incluindo a classificação de imagens médicas e geológicas. O uso dessas arquiteturas não apenas reduz o tempo de treinamento, mas também melhora a precisão e a eficácia em diversas áreas de pesquisa e aplicação prática (Zeimarani et al., 2020; "Dog Breed Identification with Fine Tuning of Pre-trained Models", 2019; Awais et al., 2020).
            """)
        
            st.write("### Referências")
        
            st.write("""
            - (2019). Dog breed identification with fine tuning of pre-trained models. *International Journal of Recent Technology and Engineering*, 8(2S11), 3677-3680. https://doi.org/10.35940/ijrte.b1464.0982s1119
            - (2022). A framework for flood extent mapping using cnn transfer learning. https://doi.org/10.17762/ijisae.v10i3s.2426
            - Ahmed, A. (2021). Pre-trained cnns models for content based image retrieval. *International Journal of Advanced Computer Science and Applications*, 12(7). https://doi.org/10.14569/ijacsa.2021.0120723
            - Alaoui, A. (2023). Pre-trained cnns: evaluating emergency vehicle image classification. *Data & Metadata*, 2, 153. https://doi.org/10.56294/dm2023153
            - Benegui, C. and Ionescu, R. (2020). Convolutional neural networks for user identification based on motion sensors represented as images. *IEEE Access*, 8, 61255-61266. https://doi.org/10.1109/access.2020.2984214
            - Cheng, R. (2023). Expansion of the ct-scans image set based on the pretrained dcgan for improving the performance of the cnn. *Journal of Physics Conference Series*, 2646(1), 012015. https://doi.org/10.1088/1742-6596/2646/1/012015
            - Hamdaoui, H., Ben-fares, A., Boujraf, S., Chaoui, N., Alami, B., Maâroufi, M., … & Qjidaa, H. (2021). High precision brain tumor classification model based on deep transfer learning and stacking concepts. *Indonesian Journal of Electrical Engineering and Computer Science*, 24(1), 167. https://doi.org/10.11591/ijeecs.v24.i1.pp167-177
            - Petrovska, B., Atanasova-Pacemska, T., Corizzo, R., Mignone, P., Lameski, P., & Zdravevski, E. (2020). Aerial scene classification through fine-tuning with adaptive learning rates and label smoothing. *Applied Sciences*, 10(17), 5792. https://doi.org/10.3390/app10175792
            - Raghava, Y., Kuthadi, V., & Rajalakshmi, S. (2019). Enhanced deep learning with featured transfer learning in identifying disguised faces. *International Journal of Innovative Technology and Exploring Engineering*, 8(10), 1257-1260. https://doi.org/10.35940/ijitee.h7286.0881019
            - Sandotra, N., Mahajan, P., Abrol, P., & Lehana, P. (2023). Analyzing performance of deep learning models under the presence of distortions in identifying plant leaf disease. *International Journal of Informatics and Communication Technology (IJ-ICT)*, 12(2), 115. https://doi.org/10.11591/ijict.v12i2.pp115-126
            - Sardeshmukh, M. (2023). Crop image classification using convolutional neural network. *Multidisciplinary Science Journal*, 5(4), 2023039. https://doi.org/10.31893/multiscience.2023039
            - Suhana, R. (2022). Fish image classification using adaptive learning rate in transfer learning method. *Knowledge Engineering and Data Science*, 5(1), 67. https://doi.org/10.17977/um018v5i12022p67-77
            - Varshni, D., Thakral, K., Agarwal, L., Nijhawan, R., & Mittal, A. (2019). Pneumonia detection using cnn based feature extraction. https://doi.org/10.1109/icecct.2019.8869364
            - Zeimarani, B., Costa, M., Nurani, N., Bianco, S., Pereira, W., & Filho, C. (2020). Breast lesion classification in ultrasound images using deep convolutional neural network. *IEEE Access*, 8, 133349-133359. https://doi.org/10.1109/access.2020.3010863
            - Zhang, B., Wang, C., Shen, Y., & Liu, Y. (2018). Fully connected conditional random fields for high-resolution remote sensing land use/land cover classification with convolutional neural networks. *Remote Sensing*, 10(12), 1889. https://doi.org/10.3390/rs10121889
            """)

    model_name = st.sidebar.selectbox("Modelo Pré-treinado:", options=['ResNet18', 'ResNet50', 'DenseNet121'])

    #________________________________________________________________________________________
    # Fine-Tuning Completo em Redes Neurais Profundas
    with st.sidebar:
        with st.expander("Fine-Tuning Completo em Redes Neurais Profundas:"):
            st.write("""
            ### Introdução
        
            O **fine-tuning** (ajuste fino) é uma técnica poderosa utilizada para ajustar redes neurais pré-treinadas em novos conjuntos de dados. No contexto de redes como a **ResNet18**, **ResNet50** ou **DenseNet121**, que foram inicialmente treinadas em grandes bases de dados (como o **ImageNet**), o fine-tuning permite que essas redes sejam adaptadas a novos problemas, como a **classificação de melanomas** ou de **rochas vulcânicas e plutônicas**. Ao realizar o fine-tuning, todas as camadas do modelo são atualizadas para refletir as características do novo conjunto de dados, ao invés de congelar as camadas iniciais, o que permite uma adaptação mais profunda e precisa ao novo problema (Piotrowski & Napiorkowski, 2013; Friedrich et al., 2022).
            """)
        
            st.write("""
            ### Fundamentação Teórica
        
            O conceito de fine-tuning é baseado no princípio de **transferência de aprendizado**, no qual um modelo pré-treinado em um grande conjunto de dados genéricos é reaproveitado para um novo problema específico. Essa abordagem é particularmente útil quando o novo conjunto de dados é relativamente pequeno, pois o modelo já foi treinado para capturar padrões gerais em dados visuais (como bordas, texturas e formas), o que pode acelerar o treinamento e melhorar a precisão final (Al‐rimy et al., 2023; Sakizadeh et al., 2015).
            """)
        
            st.write("""
            Ao utilizar o fine-tuning completo, todas as camadas do modelo são ajustadas com base nos novos dados. Isso significa que os pesos das camadas profundas do modelo, que foram aprendidos durante o treinamento inicial, são atualizados para se adequar às características específicas do novo conjunto de dados. Matematicamente, essa abordagem pode ser descrita como a otimização da seguinte função de perda:
            """)
        
            st.latex(r'''
            L_{\text{fine-tuning}} = L_{\text{original}} + \lambda \sum_{i} w_i^2
            ''')
        
            st.write("""
            Onde:
            """)
          
            st.latex(r'''
            L_{\text{fine-tuning}}
            ''')
          
            st.write("""
            é a função de perda durante o fine-tuning;
            """)
          
            st.latex(r'''
            L_{\text{original}}
            ''')
          
            st.write("""
            representa a função de perda original do modelo pré-treinado;
            """)
          
            st.latex(r'''
            \lambda
            ''')
          
            st.write("""
            é o coeficiente de regularização (no caso de utilizar a regularização L2);
            """)
          
            st.latex(r'''
            w_i
            ''')
            st.write("""
            são os pesos individuais que serão atualizados durante o processo de fine-tuning (Friedrich et al., 2022; Al‐rimy et al., 2023).
            """)
        
            st.write("""
            ### Benefícios do Fine-Tuning Completo
        
            O fine-tuning completo oferece vários benefícios, especialmente quando o novo conjunto de dados difere substancialmente do conjunto no qual o modelo foi originalmente treinado. No caso da **classificação de melanomas** ou **rochas**, por exemplo, as características visuais dos dados podem ser muito diferentes das imagens do **ImageNet**, que incluem uma ampla variedade de objetos, animais e cenários (Piotrowski & Napiorkowski, 2013; Sakizadeh et al., 2015).
            """)
        
            st.write("""
            Os principais benefícios incluem:
            1. **Adaptação Profunda**: Ao ajustar todas as camadas, o modelo consegue adaptar não apenas as características genéricas (como bordas e texturas), mas também padrões mais complexos e específicos do novo problema.
            2. **Melhoria da Precisão**: O fine-tuning completo geralmente resulta em melhorias significativas na precisão, especialmente quando os dados de treinamento são limitados ou possuem características visuais únicas (Friedrich et al., 2022; Al‐rimy et al., 2023).
            3. **Generalização Melhorada**: O processo de fine-tuning permite que o modelo generalize melhor para novos dados, uma vez que ele é treinado para capturar padrões mais específicos do novo domínio (Piotrowski & Napiorkowski, 2013; Sakizadeh et al., 2015).
            """)
        
            st.write("""
            ### Comparação com o Fine-Tuning Parcial
        
            Em contraste com o fine-tuning completo, no qual todas as camadas são atualizadas, o **fine-tuning parcial** mantém algumas das camadas iniciais congeladas, atualizando apenas as camadas finais. Essa abordagem pode ser útil quando o novo conjunto de dados é semelhante ao conjunto de dados original no qual o modelo foi treinado. No entanto, quando os dados diferem substancialmente, o fine-tuning completo tende a ser mais eficaz, pois permite uma adaptação mais profunda e personalizada (Al‐rimy et al., 2023; Sakizadeh et al., 2015).
            """)
        
            st.write("""
            ### Efeitos do Fine-Tuning em Problemas Específicos
        
            #### Classificação de Melanomas
        
            No caso da **classificação de melanomas**, o fine-tuning completo permite que o modelo identifique padrões visuais sutis na pele que podem ser indicativos de câncer. Essas características visuais podem incluir variações de textura, cor e bordas, que são específicas de imagens médicas e diferem dos objetos comuns presentes em bases de dados genéricas, como o **ImageNet** (Piotrowski & Napiorkowski, 2013; Friedrich et al., 2022).
            """)
        
            st.write("""
            #### Classificação de Rochas
        
            Para a **classificação de rochas vulcânicas e plutônicas**, o fine-tuning completo permite que o modelo capture padrões geológicos e estruturais específicos, como variações de granulação e texturas minerais. Novamente, esses padrões são significativamente diferentes dos dados de objetos comuns, tornando o fine-tuning completo uma abordagem valiosa para melhorar a precisão da classificação (Friedrich et al., 2022; Al‐rimy et al., 2023).
            """)
        
            st.write("""
            ### Considerações Práticas
        
            Durante o processo de fine-tuning, é importante monitorar o desempenho do modelo em um conjunto de validação para evitar o **overfitting**. Uma técnica comum é utilizar a **regularização L2** ou o **dropout** para garantir que o modelo não se ajuste excessivamente aos dados de treinamento (Piotrowski & Napiorkowski, 2013; Sakizadeh et al., 2015). Além disso, a taxa de aprendizado deve ser cuidadosamente ajustada. Em muitos casos, utiliza-se uma taxa de aprendizado menor durante o fine-tuning para garantir que as atualizações dos pesos não sejam muito drásticas, preservando parte das informações aprendidas anteriormente.
            """)
        
            st.write("""
            ### Conclusão
        
            O fine-tuning completo é uma técnica eficaz para ajustar modelos pré-treinados, como a **ResNet18**, **ResNet50** ou **DenseNet121**, a novos conjuntos de dados. Ao permitir que todas as camadas do modelo sejam atualizadas, o fine-tuning completo oferece maior flexibilidade e precisão em problemas que diferem substancialmente dos dados originais. Quando combinado com outras técnicas de regularização, como a L2, o fine-tuning pode levar a modelos robustos e capazes de generalizar para novos dados, sendo uma ferramenta essencial no arsenal de técnicas de aprendizado profundo.
            """)
        
            st.write("""
            ### Referências
        
            - Al‐RIMY, B.; SAEED, F.; AL-SAREM, M.; ALBARRAK, A.; QASEM, S. An adaptive early stopping technique for densenet169-based knee osteoarthritis detection model. *Diagnostics*, 13(11), 1903, 2023. https://doi.org/10.3390/diagnostics13111903
            - FRIEDRICH, S. et al. Regularization approaches in clinical biostatistics: a review of methods and their applications. *Statistical Methods in Medical Research*, 32(2), 425-440, 2022. https://doi.org/10.1177/09622802221133557
            - PIOTROWSKI, A.; NAPIORKOWSKI, J. A comparison of methods to avoid overfitting in neural networks training in the case of catchment runoff modelling. *Journal of Hydrology*, 476, 97-111, 2013. https://doi.org/10.1016/j.jhydrol.2012.10.019
            - REZAEEZADE, A.; BATINA, L. Regularizers to the rescue: fighting overfitting in deeplearning-based side-channel analysis. 2022. https://doi.org/10.21203/rs.3.rs-2386625/v1
            - SAKIZADEH, M.; MALIAN, A.; AHMADPOUR, E. Groundwater quality modeling with a small data set. *Ground Water*, 54(1), 115-120, 2015. https://doi.org/10.1111/gwat.12317
            """)

    fine_tune = st.sidebar.checkbox("Fine-Tuning Completo", value=False)
    epochs = st.sidebar.slider("Número de Épocas:", min_value=1, max_value=500, value=200, step=1)
    learning_rate = st.sidebar.select_slider("Taxa de Aprendizagem:", options=[0.1, 0.01, 0.001, 0.0001], value=0.0001)
    batch_size = st.sidebar.selectbox("Tamanho de Lote:", options=[4, 8, 16, 32, 64], index=2)
    train_split = st.sidebar.slider("Percentual de Treinamento:", min_value=0.5, max_value=0.9, value=0.7, step=0.05)
    valid_split = st.sidebar.slider("Percentual de Validação:", min_value=0.05, max_value=0.4, value=0.15, step=0.05)
    #________________________________________________________________________________________
    # Sidebar com o conteúdo explicativo e fórmula LaTeX
    with st.sidebar:
        with st.expander("Implementação da Técnica de Regularização L2 (Weight Decay):"):
            st.write("""
            ### Introdução
            A regularização L2, frequentemente referida como *weight decay*, é uma técnica amplamente utilizada para mitigar o **overfitting** 
            em modelos de aprendizado de máquina, especialmente em redes neurais profundas. O *overfitting* ocorre quando o modelo se ajusta não apenas 
            aos padrões dos dados de treinamento, mas também ao ruído presente, o que compromete sua capacidade de generalização para novos dados 
            (Piotrowski & Napiorkowski, 2013). A regularização L2 adiciona um termo de penalização à função de perda do modelo, o que resulta em uma 
            redução dos valores absolutos dos pesos, promovendo, assim, modelos mais simples e generalizáveis (Friedrich et al., 2022).
            Esta revisão visa fornecer uma visão clara e técnica da aplicação da regularização L2, discutindo seus efeitos, a interpretação do coeficiente de regularização 
            """)
          
            st.latex(r'''
            \lambda
            ''')
          
            st.write("""
            e as implicações da escolha desse parâmetro.
            """)
          
            st.latex(r'''
            L_{\text{total}} = L_{\text{original}} + \lambda \sum_{i} w_i^2
            ''')
          
            st.write("""
            Onde:
            """) 
            
            st.latex(r'''
            L_{\text{total}}
            ''')
          
            st.write("""
            é a perda total que o modelo busca minimizar;
            """)
            
            st.latex(r'''
            L_{\text{original}}
            ''')
          
            st.write("""
            é a função de perda original (como a perda de entropia cruzada); λ é o coeficiente de regularização, que controla a penalidade aplicada aos pesos;
            """)
          
            st.latex(r'''
            w_i
            ''')
          
            st.write(""" 
            são os pesos individuais do modelo (Al‐Rimy et al., 2023).
            """)
          
            st.write("""
            Este termo adicional penaliza pesos grandes, forçando o modelo a priorizar soluções que utilizam pesos menores, o que é crucial para evitar 
            que o modelo memorize os dados de treinamento, promovendo maior capacidade de generalização (Sakizadeh et al., 2015).
            """)
          
            st.write("""
            ### Fundamentação Teórica
            A regularização L2 tem uma base teórica sólida, sendo amplamente aplicada para controlar a complexidade do modelo. Ao adicionar o termo de penalização, 
            a regularização L2 ajuda a evitar o overfitting e melhora a estabilidade numérica do modelo (Friedrich et al., 2022). Isso é particularmente importante 
            em redes neurais profundas, onde o número de parâmetros pode ser grande e a complexidade do modelo alta.
            """)
          
            st.write("""
            ### Efeitos da Regularização L2
            A regularização L2 controla a complexidade do modelo ao penalizar pesos grandes, o que é particularmente útil em cenários com muitos parâmetros 
            ou dados ruidosos (Piotrowski & Napiorkowski, 2013). Além de reduzir o overfitting, a L2 promove a estabilidade no treinamento, melhorando a consistência do desempenho 
            em dados de teste (Friedrich et al., 2022).
            """)
    
            st.write("""
            ### Interpretação e Efeitos Práticos de λ
            """)
          
            st.write("""        
            A escolha do valor de λ
            """)
      
            st.write("""
            influencia diretamente o comportamento do modelo:
            """)
    
            st.write("""
            #### λ = 0
            """)
            st.write("""
            Quando λ = 0, a regularização L2 está desativada. Isso permite que o modelo ajuste-se livremente aos dados de treinamento, 
            aumentando o risco de overfitting, especialmente em conjuntos de dados pequenos ou ruidosos (Friedrich et al., 2022).
            """)
    
            st.write("""
            #### λ = 0,01
            """)
            st.write("""
            Este é um valor moderado, que penaliza de forma equilibrada os pesos do modelo. Essa configuração ajuda a evitar o overfitting sem comprometer a capacidade do modelo de 
            aprender padrões relevantes (Al‐Rimy et al., 2023).
            """)
    
            st.write("""
            #### λ = 0,02 ou λ = 0,03
            Esses valores aumentam a intensidade da penalização, sendo úteis em cenários com dados ruidosos ou em que o número de parâmetros é alto em relação à quantidade de dados 
            disponíveis (Piotrowski & Napiorkowski, 2013). Contudo, deve-se monitorar o desempenho do modelo, pois valores elevados de λ podem resultar em **underfitting**, 
            comprometendo a capacidade do modelo de capturar padrões complexos (Friedrich et al., 2022).
            """)
    
            st.write("""
            ### Conclusão
            A regularização L2 é uma técnica poderosa no treinamento de redes neurais profundas, ajudando a mitigar o overfitting e a melhorar a capacidade de generalização do modelo. 
            Ao penalizar pesos grandes, a L2 incentiva soluções mais simples e robustas. No entanto, a escolha do valor de λ é crucial para garantir que o modelo consiga capturar 
            padrões complexos sem se ajustar excessivamente aos dados de treinamento.
            """)
    
            st.write("""
            ### Referências
            - AL‐RIMY, B.; SAEED, F.; AL-SAREM, M.; ALBARRAK, A.; QASEM, S. An adaptive early stopping technique for densenet169-based knee osteoarthritis detection model. *Diagnostics*, 13(11), 1903, 2023. https://doi.org/10.3390/diagnostics13111903
            - FRIEDRICH, S. et al. Regularization approaches in clinical biostatistics: a review of methods and their applications. *Statistical Methods in Medical Research*, 32(2), 425-440, 2022. https://doi.org/10.1177/09622802221133557
            - PIOTROWSKI, A.; NAPIORKOWSKI, J. A comparison of methods to avoid overfitting in neural networks training in the case of catchment runoff modelling. *Journal of Hydrology*, 476, 97-111, 2013. https://doi.org/10.1016/j.jhydrol.2012.10.019
            - SAKIZADEH, M.; MALIAN, A.; AHMADPOUR, E. Groundwater quality modeling with a small data set. *Ground Water*, 54(1), 115-120, 2015. https://doi.org/10.1111/gwat.12317
            """)
    

  
    l2_lambda = st.sidebar.number_input("L2 Regularization (Weight Decay):", min_value=0.0, max_value=0.1, value=0.01, step=0.01)
    l1_lambda = st.sidebar.number_input("L1 Regularization:", min_value=0.0, max_value=0.01, value=0.0, step=0.001, 
                                        help="Adiciona regularização L1 (Lasso) ao treinamento. Promove esparsidade nos pesos.")
    
    #________________________________________________________________________________________
    # Novos parâmetros de treinamento
    st.sidebar.write("---")
    st.sidebar.subheader("⚙️ Configurações Avançadas")
    
    # Tipo de Aumento de Dados
    augmentation_type = st.sidebar.selectbox(
        "Técnica de Aumento de Dados:",
        options=['none', 'standard', 'mixup', 'cutmix'],
        index=1,
        help="None: Sem aumento | Standard: Transformações básicas | Mixup: Mistura imagens | Cutmix: Recorta e cola regiões"
    )
    
    # Otimizador
    optimizer_options = ['Adam', 'AdamW', 'SGD']
    if ADVANCED_OPTIMIZERS_AVAILABLE:
        optimizer_options.extend(['Ranger', 'Lion'])
    
    optimizer_name = st.sidebar.selectbox(
        "Otimizador:",
        options=optimizer_options,
        index=0,
        help="Adam: Adaptativo padrão | AdamW: Adam com weight decay melhorado | SGD: Gradiente descendente com momento | Ranger: Lookahead + RAdam | Lion: Otimizador eficiente recente"
    )
    
    # Learning Rate Scheduler
    scheduler_name = st.sidebar.selectbox(
        "Agendador de Learning Rate:",
        options=['None', 'CosineAnnealingLR', 'OneCycleLR'],
        index=0,
        help="None: LR constante | CosineAnnealingLR: Reduz LR com coseno | OneCycleLR: Aumenta e depois reduz LR"
    )
    
    # Tipo de Grad-CAM
    gradcam_type = st.sidebar.selectbox(
        "Tipo de Grad-CAM:",
        options=['GradCAM', 'GradCAMpp', 'SmoothGradCAMpp', 'LayerCAM'],
        index=2,
        help="GradCAM: Básico | GradCAMpp: Melhorado | SmoothGradCAMpp: Suavizado | LayerCAM: Por camada"
    )
    
    st.sidebar.write("---")
    
    #________________________________________________________________________________________
    # Sidebar com o conteúdo explicativo e fórmula LaTeX
    with st.sidebar:
        with st.expander("Implementação da Técnica de Parada Precoce - Early Stopping:"):
            st.write("""
            #### Introdução
            A técnica de **parada precoce** (ou *early stopping*) é amplamente utilizada para mitigar o **overfitting** no treinamento de redes neurais profundas. 
            O overfitting ocorre quando o modelo se ajusta tão bem aos dados de treinamento que sua capacidade de generalização para novos dados é prejudicada. 
            O princípio da parada precoce é interromper o treinamento quando o desempenho do modelo em um conjunto de validação não apresenta melhorias significativas 
            após um número predefinido de épocas. Essa abordagem baseia-se na observação de que, após certo ponto, melhorias no desempenho do modelo em dados de treinamento 
            não resultam em melhorias em dados que o modelo ainda não viu (Piotrowski & Napiorkowski, 2013; Al‐Rimy et al., 2023).
            """)
      
            st.write("Matematicamente, a parada precoce pode ser descrita pela seguinte condição de interrupção:")
            # Fórmulas matemáticas
            st.latex(r'''
            \text{Se } L_{\text{val}}(t) \geq L_{\text{val}}(t-1)
            ''')
            st.write("""
            por (p) épocas consecutivas, então interrompa o treinamento. Aqui,
            """)
            st.latex(r'''
            L_{\text{val}}(t)
            ''')
    
            st.write("""
            representa o valor da **função de perda** no conjunto de validação na época (t), e (p) é o **parâmetro de paciência**. 
            A paciência (p) define quanto tempo o treinamento deve continuar mesmo que não haja melhorias imediatas. Se a perda não melhorar por (p) épocas consecutivas, 
            o treinamento é interrompido.
            """)
      
            st.write("""
            #### A Importância da Paciência
            O parâmetro de **paciência** define o número de épocas consecutivas sem melhoria na métrica de validação que o modelo pode suportar antes de o treinamento ser interrompido. 
            A escolha do valor de paciência tem impacto direto no equilíbrio entre **evitar o overfitting** e **permitir que o modelo continue aprendendo**. 
            """)
      
            st.write("##### Paciência = 0")
            st.write("""
            Um valor de paciência igual a zero implica que o treinamento será interrompido imediatamente após a primeira ocorrência de estagnação na métrica de validação. 
            Isso pode ser útil em cenários onde se deseja evitar qualquer risco de *overfitting*.
            """)
      
            st.write("##### Paciência ≥ 1")
            st.write("""
            Uma paciência maior (como 1 ou 2) permite que o modelo continue sendo treinado mesmo após pequenas flutuações no desempenho, 
            o que pode ser benéfico em conjuntos de dados ruidosos (Sakizadeh et al., 2015).
            """)
      
            st.write("""
            #### Impacto do *Early Stopping* e da Paciência
            A configuração do parâmetro de paciência influencia diretamente a eficiência do treinamento. Com uma paciência muito baixa, o treinamento pode ser interrompido de forma prematura, 
            mesmo que o modelo ainda tenha potencial de melhoria. Por outro lado, uma paciência muito alta pode permitir que o modelo se ajuste excessivamente aos dados de treinamento, 
            levando ao *overfitting* (Sakizadeh et al., 2015).
            """)
      
            st.write("""
            #### Exemplos de Aplicação
            Um exemplo prático de uso da parada precoce é em tarefas de **classificação de imagens**. Durante o treinamento de um modelo para detecção de melanoma, se a acurácia no conjunto de validação 
            não melhorar após um determinado número de épocas, o early stopping é acionado.
            """)
      
            st.write("""
            #### Integração com Outras Técnicas de Regularização
            A parada precoce pode ser usada em conjunto com outras técnicas de regularização, como a **injeção de ruído** e a regularização **L1/L2**, 
            para melhorar a robustez do modelo e sua capacidade de generalização (Friedrich et al., 2022). 
            A combinação dessas técnicas ajuda a evitar que o modelo se ajuste excessivamente aos dados de treinamento, principalmente em cenários com volumes limitados de dados.
            """)
      
            st.write("""
            #### Conclusão
            A **parada precoce** é uma técnica eficaz para evitar o *overfitting* no treinamento de redes neurais profundas. O valor da paciência desempenha um papel crítico, 
            permitindo o equilíbrio entre **eficiência computacional** e **capacidade de aprendizado**. Além disso, a combinação da parada precoce com outras técnicas de regularização 
            pode melhorar ainda mais o desempenho do modelo.
            """)
      
            st.write("""
            #### Referências
            - PIOTROWSKI, A.; NAPIORKOWSKI, J. A comparison of methods to avoid overfitting in neural networks training in the case of catchment runoff modelling. *Journal of Hydrology*, v. 476, p. 97-111, 2013. https://doi.org/10.1016/j.jhydrol.2012.10.019.
            - AL‐RIMY, B. et al. An adaptive early stopping technique for densenet169-based knee osteoarthritis detection model. *Diagnostics*, v. 13, n. 11, p. 1903, 2023. https://doi.org/10.3390/diagnostics13111903.
            - SAKIZADEH, M.; MALIAN, A.; AHMADPOUR, E. Groundwater quality modeling with a small data set. *Ground Water*, v. 54, n. 1, p. 115-120, 2015. https://doi.org/10.1111/gwat.12317.
            - FRIEDRICH, S. et al. Regularization approaches in clinical biostatistics: a review of methods and their applications. *Statistical Methods in Medical Research*, v. 32, n. 2, p. 425-440, 2022. https://doi.org/10.1177/09622802221133557.
            """)


    #________________________________________________________________________________________
    patience = st.sidebar.number_input("Paciência para Early Stopping:", min_value=1, max_value=10, value=3, step=1)

    #____________________________________________________________________________________________
    with st.sidebar:
        with st.expander("Perda Ponderada para Classes Desbalanceadas:"):
            st.write("""
            ### Perda Ponderada para Classes Desbalanceadas
        
            A técnica de **perda ponderada** para lidar com **classes desbalanceadas** é amplamente utilizada em **aprendizado de máquina**, especialmente em redes neurais, para tratar problemas onde o número de amostras entre as classes de um conjunto de dados não é equilibrado. O desbalanceamento ocorre em diversos domínios, como detecção de fraudes, diagnóstico de doenças e classificação de imagens. O principal objetivo da perda ponderada é ajustar a função de perda, atribuindo diferentes pesos às classes, de forma que o impacto das classes minoritárias (menos representadas) seja ampliado e o impacto das classes majoritárias seja reduzido. Isso ajuda o modelo a aprender de forma mais eficaz em cenários onde o desequilíbrio entre as classes pode levar ao **overfitting** nas classes majoritárias e à **sub-representação** das classes minoritárias (Buda et al., 2018).
        
            ### Motivação e Justificativa Científica
        
            Em um cenário de classificação de imagens, se o modelo for treinado com uma quantidade muito maior de amostras de uma classe (classe majoritária) em relação a outra (classe minoritária), o modelo tende a ser enviesado para a classe majoritária. Isso ocorre porque o objetivo padrão da maioria das funções de perda, como a **entropia cruzada**, é minimizar a soma dos erros. Em um conjunto de dados desbalanceado, essa minimização pode ser alcançada simplesmente classificando todas as amostras como pertencentes à classe majoritária, resultando em alta acurácia geral, mas com desempenho ruim na classe minoritária. Para resolver esse problema, atribui-se um peso maior à classe minoritária, forçando a função de perda a penalizar mais fortemente os erros cometidos nessa classe (Buda et al., 2018).
        
            ### Implementação no Código
        
            No código, a implementação da perda ponderada é feita utilizando a função de perda **CrossEntropyLoss** do PyTorch, que suporta a aplicação de pesos às classes. Esses pesos são calculados com base na **frequência das classes** no conjunto de treinamento. Classes com menos amostras recebem pesos maiores, enquanto classes com mais amostras recebem pesos menores, balanceando o impacto de ambas durante o treinamento do modelo.
        
            """)
            
            st.write("**criterion = nn.CrossEntropyLoss(weight=class_weights)**")
            
            st.write("""
            No trecho de código acima, o vetor `targets` coleta os rótulos das amostras no conjunto de treino e a função `np.bincount(targets)` conta quantas vezes cada classe aparece, resultando em um vetor `class_counts`, onde cada índice corresponde à quantidade de amostras de uma classe específica (Buda et al., 2018).
        
            ### Etapas do Processo
        
            1. **Cálculo das Frequências das Classes**: As frequências de cada classe são calculadas usando `np.bincount`. Classes menos representadas recebem pesos maiores.
            2. **Ajuste para Evitar Divisão por Zero**: Um pequeno valor (1e-6) é adicionado para evitar divisão por zero quando uma classe não tem nenhuma amostra.
            3. **Cálculo dos Pesos Inversos**: A partir da frequência, os pesos são calculados tomando o inverso da frequência de cada classe. Isso aumenta a penalização dos erros nas classes minoritárias.
            4. **Função de Perda Ponderada**: A função de perda `nn.CrossEntropyLoss(weight=class_weights)` usa os pesos calculados, penalizando mais fortemente os erros das classes minoritárias.
        
            ### Impacto e Eficácia da Perda Ponderada
        
            A **perda ponderada** ajusta o aprendizado do modelo, incentivando a penalização dos erros cometidos nas classes minoritárias. Estudos demonstram que essa técnica é eficaz em aumentar a **recall** das classes minoritárias, sem comprometer drasticamente a precisão das classes majoritárias (Buda et al., 2018). No entanto, a aplicação da perda ponderada pode tornar o treinamento mais **sensível à escolha dos hiperparâmetros**, como a **taxa de aprendizado**, pois o modelo passa a ser fortemente influenciado pelas amostras menos representativas.
        
            ### Conclusão
        
            A implementação da **perda ponderada** no código é uma abordagem robusta para lidar com **classes desbalanceadas**. Ao ajustar os pesos da função de perda com base nas frequências das classes, o modelo consegue equilibrar melhor o aprendizado entre as classes majoritárias e minoritárias, evitando vieses que favorecem a classe mais representada no conjunto de dados (Buda et al., 2018).
        
            ### Referências
        
            - Buda, M., Maki, A., & Mazurowski, M. (2018). A systematic study of the class imbalance problem in convolutional neural networks. *Neural Networks*, 106, 249-259. https://doi.org/10.1016/j.neunet.2018.07.011
            """)

    use_weighted_loss = st.sidebar.checkbox("Usar Perda Ponderada para Classes Desbalanceadas", value=False)
    st.sidebar.image("eu.ico", width=80)
   
    st.sidebar.write("""
    Produzido pelo:
    
    Projeto Geomaker + IA 
    
    https://doi.org/10.5281/zenodo.13910277
    
    - Professor: Marcelo Claro.
    
    Contatos: marceloclaro@gmail.com
    
    Whatsapp: (88)981587145
    
    Instagram: [marceloclaro.geomaker](https://www.instagram.com/marceloclaro.geomaker/)
    
    """)
     # _____________________________________________
    # Controle de Áudio
    st.sidebar.title("Controle de Áudio")
    
    # Dicionário de arquivos de áudio, com nomes amigáveis mapeando para o caminho do arquivo
    mp3_files = {
        "Áudio explicativo para Leigos": "leigo.mp3",
        "Áudio explicativo para treinamentos de poucos dados": "bucal.mp3",
    }
    
    # Lista de arquivos MP3 para seleção
    selected_mp3 = st.sidebar.radio("Escolha um áudio explicativo:", options=list(mp3_files.keys()))
    
    # Controle de opção de repetição
    loop = st.sidebar.checkbox("Repetir áudio")
    
    # Botão de Play para iniciar o áudio
    play_button = st.sidebar.button("Play")
    
    # Placeholder para o player de áudio
    audio_placeholder = st.sidebar.empty()
    
    # Função para verificar se o arquivo existe
    def check_file_exists(mp3_path):
        if not os.path.exists(mp3_path):
            st.sidebar.error(f"Arquivo {mp3_path} não encontrado.")
            return False
        return True
    
    # Se o botão Play for pressionado e um arquivo de áudio estiver selecionado
    if play_button and selected_mp3:
        mp3_path = mp3_files[selected_mp3]
        
        # Verificação da existência do arquivo
        if check_file_exists(mp3_path):
            try:
                # Abrindo o arquivo de áudio no modo binário
                with open(mp3_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                    
                    # Codificando o arquivo em base64 para embutir no HTML
                    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                    
                    # Controle de loop (repetição)
                    loop_attr = "loop" if loop else ""
                    
                    # Gerando o player de áudio em HTML
                    audio_html = f"""
                    <audio id="audio-player" controls autoplay {loop_attr}>
                      <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                      Seu navegador não suporta o elemento de áudio.
                    </audio>
                    """
                    
                    # Inserindo o player de áudio na interface
                    audio_placeholder.markdown(audio_html, unsafe_allow_html=True)
            
            except FileNotFoundError:
                st.sidebar.error(f"Arquivo {mp3_path} não encontrado.")
            except Exception as e:
                st.sidebar.error(f"Erro ao carregar o arquivo: {str(e)}")
    #______________________________________________________________________________________-


    # Verificar se a soma dos splits é válida
    if train_split + valid_split > 0.95:
        st.sidebar.error("A soma dos splits de treinamento e validação deve ser menor ou igual a 0.95.")

    # Upload do arquivo ZIP
    
    zip_file = st.file_uploader("Upload do arquivo ZIP com as imagens", type=["zip"])

    if zip_file is not None and train_split + valid_split <= 0.95:
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.read())
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        data_dir = temp_dir

        # Detectar automaticamente o número de classes do dataset
        try:
            temp_dataset = datasets.ImageFolder(root=data_dir)
            detected_num_classes = len(temp_dataset.classes)
            st.success(f"✅ Número de classes detectado automaticamente: **{detected_num_classes}**")
            st.write(f"Classes encontradas: {', '.join(temp_dataset.classes)}")
            num_classes = detected_num_classes
        except Exception as e:
            st.error(f"Erro ao detectar classes: {e}")
            st.error("Certifique-se de que o ZIP contém pastas com nomes de classes e imagens dentro delas.")
            shutil.rmtree(temp_dir)
            return

        st.write("Iniciando o treinamento supervisionado...")
        model_data = train_model(data_dir, num_classes, model_name, fine_tune, epochs, learning_rate, 
                                batch_size, train_split, valid_split, use_weighted_loss, l2_lambda, l1_lambda, 
                                patience, optimizer_name, scheduler_name, augmentation_type)

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
        
        # ========== OPÇÃO DE VISUALIZAÇÃO PCA ==========
        st.write("---")
        st.write("## 🔬 Análise PCA das Features")
        
        show_pca = st.checkbox("📊 Mostrar Análise PCA das Features Extraídas", value=True)
        
        if show_pca:
            # Opção de escolher número de componentes
            n_components = st.selectbox(
                "Escolha o número de componentes principais para visualização:",
                options=[2, 3],
                index=0,
                help="2 componentes: Visualização 2D | 3 componentes: Visualização 3D interativa com Plotly"
            )
            
            if n_components == 2:
                visualize_pca_features(features_reshaped, labels, classes, n_components=2)
            else:
                # 3D Visualization with Plotly
                st.write("### 📊 Visualização PCA 3D Interativa")
                try:
                    fig_3d = visualize_pca_3d(features_reshaped, labels, classes)
                    st.plotly_chart(fig_3d, use_container_width=True)
                except Exception as e:
                    st.error(f"Erro ao gerar visualização 3D: {str(e)}")
                    st.info("Mostrando visualização 2D como alternativa")
                    visualize_pca_features(features_reshaped, labels, classes, n_components=2)

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

                st.image(eval_image, caption='Imagem para avaliação', width='stretch')
                class_name, confidence = evaluate_image(model, eval_image, classes)
                st.write(f"**Classe Predita:** {class_name}")
                st.write(f"**Confiança:** {confidence:.4f}")

                # Visualizar ativações com o tipo de Grad-CAM selecionado
                activation_map = visualize_activations(model, eval_image, classes, gradcam_type)
                
                # ========== 3D GRAD-CAM VISUALIZATION ==========
                if activation_map is not None:
                    st.write("---")
                    st.write("### 🌐 Visualização 3D do Grad-CAM")
                    show_3d_gradcam = st.checkbox("Mostrar Grad-CAM em 3D", value=False)
                    if show_3d_gradcam:
                        try:
                            fig_gradcam_3d = visualize_activation_heatmap_3d(activation_map)
                            st.plotly_chart(fig_gradcam_3d, use_container_width=True)
                        except Exception as e:
                            st.error(f"Erro ao gerar visualização 3D do Grad-CAM: {str(e)}")
                
                # ========== AI CHAT DIAGNOSTIC ANALYSIS ==========
                st.write("---")
                st.write("## 🤖 Análise Diagnóstica com IA")
                
                enable_ai_analysis = st.checkbox("Ativar Análise Diagnóstica Avançada com IA", value=False)
                
                if enable_ai_analysis:
                    st.write("### Configuração da API")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        api_provider = st.selectbox(
                            "Provedor de API:",
                            options=['gemini', 'groq'],
                            help="Escolha entre Google Gemini ou Groq"
                        )
                    
                    with col2:
                        if api_provider == 'gemini':
                            model_options = ['gemini-1.0-pro', 'gemini-1.5-pro', 'gemini-1.5-flash']
                        else:
                            model_options = ['mixtral-8x7b-32768', 'llama-3.1-70b-versatile', 'llama-3.1-8b-instant']
                        
                        ai_model = st.selectbox(
                            "Modelo:",
                            options=model_options
                        )
                    
                    api_key = st.text_input(
                        "API Key:",
                        type="password",
                        help="Insira sua chave API"
                    )
                    
                    if api_key:
                        if st.button("🔬 Gerar Análise Diagnóstica Completa"):
                            with st.spinner("Gerando análise diagnóstica aprofundada..."):
                                try:
                                    # Fetch academic references
                                    st.write("📚 Buscando referências acadêmicas...")
                                    ref_fetcher = AcademicReferenceFetcher()
                                    references = ref_fetcher.get_references_for_classification(
                                        class_name=class_name,
                                        domain="image classification",
                                        max_per_source=3
                                    )
                                    
                                    if references:
                                        with st.expander("📚 Referências Acadêmicas Encontradas"):
                                            st.markdown(format_references_for_display(references))
                                    
                                    # Generate Grad-CAM description
                                    gradcam_desc = ""
                                    if activation_map is not None:
                                        gradcam_desc = describe_gradcam_regions(activation_map)
                                    
                                    # Collect training statistics
                                    training_stats = {
                                        "Épocas Treinadas": epochs,
                                        "Taxa de Aprendizagem": learning_rate,
                                        "Batch Size": batch_size,
                                        "Modelo": model_name,
                                        "Tipo de Augmentação": augmentation_type,
                                        "Otimizador": optimizer_name
                                    }
                                    
                                    # Collect statistical results
                                    # Note: These are placeholder values. In a production system,
                                    # these should be computed from actual test set evaluation.
                                    # TODO: Integrate with actual test metrics from trained model
                                    statistical_results = {
                                        "Informação": "Métricas baseadas no treinamento realizado",
                                        "Nota": "Para análise completa, avalie em conjunto de teste separado"
                                    }
                                    
                                    # Initialize AI analyzer
                                    ai_analyzer = AIAnalyzer(
                                        api_provider=api_provider,
                                        api_key=api_key,
                                        model_name=ai_model
                                    )
                                    
                                    # Generate comprehensive analysis
                                    st.write("🧠 Gerando interpretação diagnóstica...")
                                    analysis = ai_analyzer.generate_comprehensive_analysis(
                                        predicted_class=class_name,
                                        confidence=confidence,
                                        training_stats=training_stats,
                                        statistical_results=statistical_results,
                                        gradcam_description=gradcam_desc,
                                        academic_references=references
                                    )
                                    
                                    # Display analysis
                                    st.write("---")
                                    st.write("## 📋 Análise Diagnóstica Completa")
                                    st.markdown(analysis)
                                    
                                    # ========== MULTI-AGENT SYSTEM ANALYSIS (15 AGENTS + MANAGER) ==========
                                    st.write("---")
                                    st.write("## 🤖 Sistema Multi-Agente (15 Agentes + 1 Gerente)")
                                    
                                    use_multiagent = st.checkbox("Ativar Análise com Sistema Multi-Agente", value=True)
                                    
                                    if use_multiagent:
                                        with st.spinner("Coordenando análise de 15 agentes especializados..."):
                                            try:
                                                manager = ManagerAgent()
                                                
                                                # Preparar contexto com informações disponíveis
                                                agent_context = {
                                                    'training_stats': training_stats,
                                                    'statistical_results': statistical_results,
                                                    'gradcam_description': gradcam_desc,
                                                    'references': references
                                                }
                                                
                                                multi_agent_report = manager.coordinate_analysis(
                                                    predicted_class=class_name,
                                                    confidence=confidence,
                                                    context=agent_context
                                                )
                                                
                                                st.markdown(multi_agent_report)
                                                st.success("✅ Análise Multi-Agente Concluída! 15 especialistas + 1 gerente coordenador")
                                                
                                            except Exception as e:
                                                st.error(f"Erro ao gerar análise multi-agente: {str(e)}")
                                                import traceback
                                                st.code(traceback.format_exc())
                                    
                                    # ========== GENETIC ALGORITHM MULTI-ANGLE INTERPRETATION ==========
                                    st.write("---")
                                    st.write("## 🧬 Interpretação Multi-Angular com Algoritmos Genéticos")
                                    
                                    use_genetic = st.checkbox("Gerar Análise Multi-Perspectiva", value=True)
                                    
                                    if use_genetic:
                                        with st.spinner("Aplicando algoritmos genéticos para interpretação multi-angular..."):
                                            try:
                                                genetic_interpreter = GeneticDiagnosticInterpreter(
                                                    population_size=20,
                                                    generations=10
                                                )
                                                
                                                multi_angle_report = genetic_interpreter.generate_multi_angle_report(
                                                    predicted_class=class_name,
                                                    confidence=confidence
                                                )
                                                
                                                st.markdown(multi_angle_report)
                                                
                                            except Exception as e:
                                                st.error(f"Erro ao gerar análise multi-angular: {str(e)}")
                                    
                                except Exception as e:
                                    st.error(f"Erro ao gerar análise: {str(e)}")
                                    st.info("Verifique se a API key está correta e se você tem créditos disponíveis.")
                    else:
                        st.warning("⚠️ Por favor, insira sua API key para gerar a análise.")

        # Limpar o diretório temporário
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
