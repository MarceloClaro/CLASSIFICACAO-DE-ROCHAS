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
import logging
import base64
import time
import json

# Adicionar esta linha para contornar o RuntimeError com torch.classes e Streamlit
torch.classes.__path__ = []

# Importações adicionais para Grad-CAM
from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms.functional import normalize, resize, to_pil_image
import cv2
# Importações para otimizadores avançados (pode ser necessário instalar torch_optimizer)
# pip install torch_optimizer
import torch_optimizer as optim_adv
# Importações para agendadores de taxa de aprendizado
from torch.optim import lr_scheduler
# Importações para aumento de dados avançado
# Pode ser necessário instalar bibliotecas como albumentations ou implementar manualmente
# Importações para métodos CAM avançados
from torchcam.methods import ScoreCAM, LayerCAM
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
    sns.countplot(x=labels, ax=ax, palette="Set2", hue=labels, legend=False)
    
    # Adicionar os nomes das classes no eixo X
    # Set the tick locations explicitly before setting the labels to avoid the warning
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right") # Use ha="right" for better label alignment
    
    # Adicionar as contagens acima das barras
    for i, count in enumerate(class_counts):
        ax.text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
    
    ax.set_title("Distribuição das Classes (Quantidade de Imagens)")
    ax.set_xlabel("Classes")
    ax.set_ylabel("Número de Imagens")
    
    st.pyplot(fig)

def to_csv_string(data):
    """
    Converts a list of dictionaries or lists into a CSV string.
    Handles simple cases.
    """
    if not data:
        return ""

    # If data is a dictionary (like classification report), convert to a list of dicts
    if isinstance(data, dict):
        # Assuming classification report format: {class_name: {metric: value, ...}, ...}
        # Ensure there's at least one item to get keys from
        if not data:
            return ""
        header = ["Class"] + list(next(iter(data.values())).keys())
        rows = [header]
        for class_name, metrics in data.items():
            row = [class_name] + list(metrics.values())
            rows.append(row)
        data_to_process = rows

    # If data is a list of dictionaries
    elif isinstance(data, list) and data and isinstance(data[0], dict):
        header = list(data[0].keys())
        rows = [header]
        for row_dict in data:
            rows.append([row_dict.get(col, '') for col in header])
        data_to_process = rows

    # If data is a list of lists or tuples
    elif isinstance(data, list) and data and isinstance(data[0], (list, tuple)):
         data_to_process = data

    # If data is a simple list (like labels or single column data)
    # Check if it's a list of non-list/tuple/dict items
    elif isinstance(data, list) and data and not isinstance(data[0], (list, tuple, dict)):
        header = ["Value"]
        rows = [header]
        for item in data:
            rows.append([item])
        data_to_process = rows

    else:
        # Handle other potential data types or structures if necessary
        # For now, raise an error or return empty
        st.warning(f"Unsupported data type for CSV conversion: {type(data)}")
        return ""

    # Simple CSV formatting (handling commas in values might be needed for robustness)
    # Escape double quotes and enclose fields with commas or newlines in double quotes
    def escape_csv_field(field):
        field_str = str(field)
        # Check for comma, newline, or double quote
        if "," in field_str or "\n" in field_str or '"' in field_str:
            # Escape existing double quotes by doubling them
            field_str = field_str.replace('"', '""')
            # Enclose the field in double quotes
            return f'"{field_str}"'
        return field_str

    csv_string = "\n".join([",".join(escape_csv_field(col) for col in row) for row in data_to_process])
    return csv_string

def get_model(model_name, num_classes, dropout_p=0.5, fine_tune=False):
    """
    Retorna o modelo pré-treinado selecionado.
    """
    if model_name == 'ResNet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif model_name == 'ResNet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif model_name == 'DenseNet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
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

def train_model(data_dir, num_classes, model_name, fine_tune, epochs, learning_rate, batch_size, train_split, valid_split, use_weighted_loss, l2_lambda, patience, optimizer_name, lr_scheduler_name, data_augmentation_method):
    """
    Função principal para treinamento do modelo com opções avançadas.
    """
    set_seed(42)

    # Add a final check for data_dir right before ImageFolder
    if not os.path.exists(data_dir):
        st.error(f"Erro crítico dentro de train_model: O diretório de dados especificado \'{data_dir}\' não foi encontrado.")
        return None, None, None, None, None
    if not os.path.isdir(data_dir):
         st.error(f"Erro crítico dentro de train_model: O caminho especificado \'{data_dir}\' não é um diretório válido.")
         return None, None, None, None, None

    # Explicitly list directory contents for debugging
    st.write(f"Verificando o conteúdo do diretório: {data_dir}")
    try:
        dir_contents = [item.name for item in os.scandir(data_dir) if item.is_dir()]
        st.write(f"Subdiretórios encontrados: {dir_contents}")
        if not dir_contents:
            st.error("Erro: Nenhuma pasta de classe encontrada no diretório de dados. Certifique-se de que o arquivo ZIP contenha subdiretórios (classes).")
            return None, None, None, None, None
    except FileNotFoundError:
         st.error(f"Erro: O diretório {data_dir} não foi encontrado ao tentar listar seu conteúdo.")
         return None, None, None, None, None
    except Exception as e:
         st.error(f"Ocorreu um erro ao listar o conteúdo do diretório {data_dir}: {e}")
         return None, None, None, None, None

    # Carregar o conjunto de dados principal usando data_dir
    try:
        st.write(f"Tentando carregar ImageFolder de: {data_dir}")
        # Este full_dataset será usado para obter classes, dividir dados, etc.
        loaded_full_dataset = datasets.ImageFolder(root=data_dir)
        st.success(f"ImageFolder carregado com sucesso para configuração. Classes: {loaded_full_dataset.classes}")
    except Exception as e:
        st.error(f"Erro crítico ao carregar o conjunto de dados de '{data_dir}' dentro de train_model: {e}")
        st.error("Verifique se o diretório de dados está estruturado corretamente com subpastas para cada classe.")
        return None, None, None, None, None

    # Validate num_classes
    actual_num_classes = len(loaded_full_dataset.classes)
    if num_classes != actual_num_classes:
        st.error(f"Erro: O número de classes especificado ({num_classes}) não corresponde ao número de classes encontradas no conjunto de dados ({actual_num_classes}). Por favor, ajuste o Número de Classes na barra lateral.")
        return None, None, None, None, None # Indicate failure

    # Exibir algumas imagens do dataset e distribuição de classes
    st.write("Visualização de algumas imagens do conjunto de dados:")
    st.write("Exemplos do Conjunto de Dados")
    visualize_data(loaded_full_dataset, loaded_full_dataset.classes)
    st.write("\n") # Add a newline for spacing

    st.write("Distribuição das Classes")
    plot_class_distribution(loaded_full_dataset, loaded_full_dataset.classes)
    st.write("\n") # Add a newline for spacing

    # Definir as transformações com base no método de aumento de dados selecionado
    if data_augmentation_method == 'Padrão':
        current_train_transforms = train_transforms # Usar as transformações padrão já definidas
    elif data_augmentation_method == 'Mixup':
        # A implementação de Mixup/Cutmix é feita no loop de treinamento, não nas transformações do dataset.
        # As transformações do dataset serão as transformações de teste.
        current_train_transforms = test_transforms
    elif data_augmentation_method == 'Cutmix':
        # Similar ao Mixup, a lógica Cutmix é aplicada no loop de treinamento.
        current_train_transforms = test_transforms

    # Criar os datasets com as transformações apropriadas
    st.write("Processando os conjuntos de dados...")
    train_dataset_transformed = CustomDataset(loaded_full_dataset, transform=current_train_transforms)
    valid_dataset_transformed = CustomDataset(loaded_full_dataset, transform=test_transforms) # Usado para criar subsets
    # test_dataset_transformed = CustomDataset(loaded_full_dataset, transform=test_transforms) # O subset de teste usará valid_dataset_transformed

    # Dividir os índices para treino, validação e teste
    dataset_size = len(loaded_full_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    train_end = int(train_split * dataset_size)
    valid_end = int((train_split + valid_split) * dataset_size)

    train_indices = indices[:train_end]
    valid_indices = indices[train_end:valid_end]
    test_indices = indices[valid_end:]

    st.write(f"Criando subsets com divisões: Treino ({len(train_indices)}), Validação ({len(valid_indices)}), Teste ({len(test_indices)})")
    train_subset = torch.utils.data.Subset(train_dataset_transformed, train_indices)
    valid_subset = torch.utils.data.Subset(valid_dataset_transformed, valid_indices)
    test_subset = torch.utils.data.Subset(valid_dataset_transformed, test_indices) # Usa a mesma base transformada

    # Dataloaders
    g = torch.Generator()
    g.manual_seed(42)

    if use_weighted_loss:
        targets = [loaded_full_dataset.targets[i] for i in train_indices]
        class_counts = np.bincount(targets)
        class_counts = class_counts + 1e-6  # Para evitar divisão por zero
        class_weights = 1.0 / class_counts
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)

    st.write("Criando DataLoaders...")

    # Display augmented images sample
    if data_augmentation_method == 'Padrão':
        st.write("\n") # Add a newline for spacing
        st.subheader("Visualização de Imagens Após Aumento de Dados (Conjunto de Treinamento)")
        num_augmented_samples_to_show = 10 # Increase the number of samples shown
        fig, axes = plt.subplots(1, num_augmented_samples_to_show, figsize=(20, 3))
        # Get a batch from the training loader
        try:
            inputs, labels = next(iter(train_loader))
            inputs = inputs.cpu().numpy().transpose(0, 2, 3, 1) # Convert to numpy and change channel order for matplotlib
            # Denormalize if necessary for display (assuming ImageNet normalization for example)
            # For simplicity, assuming ToTensor normalization [0, 1]
            inputs = (inputs * 255).astype(np.uint8) # Simple denormalization assumption

            for i in range(min(num_augmented_samples_to_show, inputs.shape[0])):
                axes[i].imshow(inputs[i])
                axes[i].set_title(loaded_full_dataset.classes[labels[i].item()])
                axes[i].axis('off')
            st.pyplot(fig)
        except Exception as e:
             st.warning(f"Could not display augmented image sample: {e}")

    # Carregar o modelo
    model = get_model(model_name, actual_num_classes, dropout_p=0.5, fine_tune=fine_tune) # Pass actual_num_classes to get_model
    if model is None:
        return None, None, None, None, None

    # Definir o otimizador com base na seleção do usuário
    # Certifique-se de que `optimizer_name` é passado para esta função
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=l2_lambda)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=l2_lambda)
    elif optimizer_name == 'Ranger':
        # Ranger está em torch_optimizer
        optimizer = optim_adv.Ranger(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=l2_lambda)
    elif optimizer_name == 'Lion':
        # Lion está em torch_optimizer
        optimizer = optim_adv.Lion(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=l2_lambda)
    elif optimizer_name == 'Sophia':
        # Sophia está em torch_optimizer
        # Sophia usa um agendador interno para o aquecimento (warmup) do Hessian
        # Pode precisar de ajustes dependendo da implementação específica em torch_optimizer
        optimizer = optim_adv.SophiaH(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=l2_lambda)
    else:
        # Fallback para Adam ou erro
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=l2_lambda)
        st.warning(f"Otimizador {optimizer_name} não reconhecido. Usando Adam como padrão.")

    # Configurar o agendador de taxa de aprendizado
    scheduler = None # Inicializa como None
    if lr_scheduler_name == 'Recozimento por Cosseno':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif lr_scheduler_name == 'Política de Um Ciclo':
        # Precisa do número total de passos de treinamento para OneCycleLR
        # Isso geralmente é len(train_loader) * epochs
        max_lr = learning_rate # O LR máximo para OneCycle policy
        steps_per_epoch = len(train_loader)
        total_steps = epochs * steps_per_epoch
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps)
    # Se lr_scheduler_name for 'Nenhum', scheduler permanece None

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
        set_seed(42 + epoch) # Definir seed por época para reprodutibilidade do dataloader
        running_loss = 0.0
        running_corrects = 0
        model.train()

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Implementação de Mixup/Cutmix no loop de treinamento
            if data_augmentation_method == 'Mixup':
                # Implementar Mixup aqui (requer gerar lambda e misturar inputs/labels)
                alpha = 1.0 # Parâmetro para a distribuição Beta. Geralmente 1.0 ou 0.2
                if alpha > 0:
                    lam = np.random.beta(alpha, alpha)
                else:
                    lam = 1

                # Misturar os dados e os rótulos
                index = torch.randperm(inputs.size(0)).to(device)
                mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
                target_a, target_b = labels, labels[index]

                optimizer.zero_grad()
                try:
                    outputs = model(mixed_inputs)
                except Exception as e:
                    st.error(f"Erro durante a passagem forward com Mixup na época {epoch+1}: {e}")
                    return None, None, None, None, None

                # Calcular a perda com Mixup
                loss = lam * criterion(outputs, target_a) + (1 - lam) * criterion(outputs, target_b)

                # Nota sobre Acurácia com Mixup: O cálculo da acurácia diretamente com preds == labels.data
                # não é apropriado para Mixup, pois os rótulos são misturados. Uma abordagem seria calcular
                # uma 'soft' acurácia ou apenas reportar a acurácia no conjunto de validação/teste.
                # Por simplicidade, manteremos o cálculo da acurácia padrão, mas esteja ciente dessa limitação.
                _, preds = torch.max(outputs, 1) # Definir preds para cálculo de acurácia
                # _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

            elif data_augmentation_method == 'Cutmix':
                # Implementar Cutmix aqui (requer gerar caixa de corte e misturar inputs/labels)
                beta = 1.0 # Parâmetro para a distribuição Beta
                if beta > 0:
                    lam = np.random.beta(beta, beta)
                else:
                    lam = 1

                # Gerar caixa de corte
                # Funções auxiliares para gerar caixa de corte (bounding box)
                def rand_bbox(size, lam):
                    W = size[2]
                    H = size[3]
                    cut_rat = np.sqrt(1. - lam)
                    cut_w = int(W * cut_rat)
                    cut_h = int(H * cut_rat)

                    # Coordenada central aleatória
                    cx = np.random.randint(W)
                    cy = np.random.randint(H)

                    bbx1 = np.clip(cx - cut_w // 2, 0, W)
                    bby1 = np.clip(cy - cut_h // 2, 0, H)
                    bbx2 = np.clip(cx + cut_w // 2, 0, W)
                    bby2 = np.clip(cy + cut_h // 2, 0, H)

                    return bbx1, bby1, bbx2, bby2

                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                # Ajustar lambda com base na área da caixa de corte
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))

                # Misturar os dados e os rótulos
                index = torch.randperm(inputs.size(0)).to(device)
                mixed_inputs = inputs.clone() # Começa com a imagem original
                mixed_inputs[:, :, bby1:bby2, bbx1:bbx2] = inputs[index, :, bby1:bby2, bbx1:bbx2] # Cola o patch

                target_a, target_b = labels, labels[index]

                optimizer.zero_grad()
                try:
                    outputs = model(mixed_inputs)
                except Exception as e:
                    st.error(f"Erro durante a passagem forward com Cutmix na época {epoch+1}: {e}")
                    return None, None, None, None, None

                # Calcular a perda com Cutmix
                loss = lam * criterion(outputs, target_a) + (1 - lam) * criterion(outputs, target_b)

                # Nota sobre Acurácia com Cutmix: Similar ao Mixup, o cálculo direto da acurácia não é apropriado.
                # Manteremos o cálculo padrão, mas ciente da limitação.
                _, preds = torch.max(outputs, 1) # Definir preds para cálculo de acurácia
                # _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

            else:
                # Sem Mixup/Cutmix, usar a loss padrão
                optimizer.zero_grad()
                try:
                    outputs = model(inputs)
                except Exception as e:
                    st.error(f"Erro durante a passagem forward na época {epoch+1}: {e}")
                    return None, None, None, None, None

                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

            loss.backward()
            optimizer.step()

            # Aplicar o passo do agendador de taxa de aprendizado (se houver)
            if scheduler is not None:
                # Para CosineAnnealingLR e a maioria dos agendadores, step() é chamado por época
                # Para OneCycleLR, step() é geralmente chamado por batch
                # Ajuste conforme o agendador específico e sua preferência
                if lr_scheduler_name == 'Política de Um Ciclo':
                     scheduler.step()
            
            running_loss += loss.item() * inputs.size(0)
            # Cálculo da acurácia também pode precisar de ajuste para Mixup/Cutmix (ver notas acima)
            # _, preds = torch.max(outputs, 1) # Já calculado acima dentro dos blocos if/else
            # running_corrects += torch.sum(preds == labels.data) # Já calculado acima dentro dos blocos if/else

        # Fim do loop de batches (i, (inputs, labels))

        # Operações a serem feitas ao final de cada época
        epoch_loss = running_loss / len(train_subset)
        epoch_acc = running_corrects.double() / len(train_subset)
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

        valid_epoch_loss = valid_running_loss / len(valid_subset)
        valid_epoch_acc = valid_running_corrects.double() / len(valid_subset)
        valid_losses.append(valid_epoch_loss)
        valid_accuracies.append(valid_epoch_acc.item())

        st.write(f'**Época {epoch + 1}/{epochs}**')
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
        
        # Aplicar o passo do agendador de taxa de aprendizado (se houver e for por época)
        if scheduler is not None and lr_scheduler_name != 'Política de Um Ciclo':
            scheduler.step()
    # Fim do loop de épocas (epoch)

    # Carregar os melhores pesos do modelo após o treinamento (se early stopping ocorreu)
    # Se não ocorreu early stopping, best_model_wts pode não ser os da última época se a última época foi a melhor
    if 'best_model_wts' in locals() and epochs_no_improve >= patience: # Garante que best_model_wts existe
         model.load_state_dict(best_model_wts)

    # Gráficos de Perda e Acurácia
    # Note: Para Mixup/Cutmix, a perda de treino plotada é a perda mista, que pode ser menor que a perda 'real'
    # A acurácia de treino também pode ser calculada de forma diferente com Mixup/Cutmix
    # Considere adicionar opções para plotar métricas 'reais' no conjunto de treino sem aumento
    plot_metrics(epochs, train_losses, valid_losses, train_accuracies, valid_accuracies)

    # Avaliação Final no Conjunto de Teste
    st.write("**Avaliação no Conjunto de Teste**")
    # A avaliação no conjunto de teste não usa Mixup/Cutmix
    classification_report_dict, confusion_matrix_array, test_true_labels, test_predicted_labels = compute_metrics(model, test_loader, loaded_full_dataset.classes)

    # Análise de Erros
    st.write("**Análise de Erros**")
    # Pass test_indices to error_analysis to link misclassified samples back to the original dataset
    misclassified_info = error_analysis(model, test_loader, loaded_full_dataset.classes, test_indices)

    # Liberar memória
    del train_loader, valid_loader
    gc.collect()
    torch.cuda.empty_cache() # Limpar cache da GPU se estiver usando CUDA
    
    # Return the model and classes
    return model, loaded_full_dataset.classes, train_losses, valid_losses, train_accuracies, valid_accuracies

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

    # Return data for download
    return report, cm, all_labels, all_preds

def error_analysis(model, dataloader, classes, test_indices):
    """
    Realiza análise de erros mostrando algumas imagens mal classificadas.
    """
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    misclassified_original_indices = []

    with torch.no_grad():
        # It's better to iterate through the test_subset directly with its original indices
        # and access the full_dataset for class names and original indices
        # This requires passing the test_indices and the original dataset to this function
        # Let's modify the function signature to accept test_indices and the original dataset

        # We will iterate through the original test_indices and load samples from the full_dataset
        # This ensures we get the original image and can link back to the original index

        # First, let's create a DataLoader from the subset using original indices
        # This is slightly redundant with the test_loader, but ensures we can get original indices easily
        # An alternative is to modify the DataLoader to yield original indices
        # For simplicity now, let's iterate through indices and load from the original full_dataset
        # This might be less efficient but clearer for linking to original indices

        # Let's revert to iterating through the test_loader as it's more efficient
        # We will need to find a way to map batch indices back to original dataset indices.
        # This is tricky with shuffling.

        # A simpler approach: iterate through the original test_indices and the original dataset
        # and manually apply the test_transforms.
        # This avoids DataLoader complexities with indexing.

        # Reverting to iterating through the test_loader is more aligned with the existing structure.
        # We need to store the original indices when creating the test_subset or modify the DataLoader.
        # Modifying train_model to create test_subset that also tracks original indices.

        # Given the current structure, the simplest way to get original indices for misclassified images
        # is to iterate through the test_loader, record predictions, and then match them back
        # to the test_subset's original indices.

        # Let's iterate through the test_loader and collect predictions and original indices
        # We need to access the original indices from the test_subset which the test_loader is based on.
        # The test_loader does not directly provide original indices per sample in a batch with shuffling.

        # A better approach is to create a custom DataLoader or dataset that includes original indices.
        # For now, let's iterate through the test_subset directly to get original indices easily.
        # This function was called with test_loader. Let's stick to using the dataloader but try to map back.

        # Let's modify the train_model to pass a list of tuples (image, label, original_index) to the dataloader.
        # Or, create a custom dataset that wraps ImageFolder and includes original index.

        # Given the current state, the easiest way to get original indices for the misclassified samples
        # displayed (up to 5) is to find their corresponding original indices within the test_subset.
        # This requires iterating through the test_subset and comparing images or finding matches based on order.

        # Let's assume for now that the order of samples in the test_loader without shuffling corresponds
        # to the order in the test_subset, which in turn corresponds to the order of test_indices.
        # This assumption might be incorrect if shuffle=True is used in the test_loader.
        # Looking at the train_model, shuffle=False is used for the test_loader. Good.
        # So, the order in test_loader should match the order in test_subset, which is based on test_indices.

        # So, we can iterate through the test_loader, keep track of the batch index and sample index within the batch.
        # Then calculate the overall index within the test_subset (batch_index * batch_size + sample_index_in_batch).
        # This overall index corresponds to the index in the test_indices list.

        # Let's pass the test_indices list to error_analysis.
        # The function signature was already modified to accept test_indices in a previous step.
        # Need to capture the original indices within the loop.

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            incorrect = preds != labels
            if incorrect.any():
                # Get the indices of incorrect predictions within the current batch
                incorrect_indices_in_batch = torch.where(incorrect)[0]

                for i in incorrect_indices_in_batch:
                    # Calculate the overall index in the test_subset
                    overall_index_in_subset = batch_idx * dataloader.batch_size + i

                    # Get the original index from the test_indices list
                    original_index = test_indices[overall_index_in_subset]

                    misclassified_images.append(inputs[i].cpu())
                    misclassified_labels.append(labels[i].cpu())
                    misclassified_preds.append(preds[i].cpu())
                    misclassified_original_indices.append(original_index)

                    # Stop if we have enough misclassified images for visualization
                    if len(misclassified_images) >= 5:
                        break # Break from inner loop

                if len(misclassified_images) >= 5:
                     break # Break from outer loop (batches)

    if misclassified_images:
        st.write("Algumas imagens mal classificadas:")
        fig, axes = plt.subplots(1, min(5, len(misclassified_images)), figsize=(15, 3))
        for i in range(min(5, len(misclassified_images))):
            image = misclassified_images[i]
            image = image.permute(1, 2, 0).numpy()
            # Denormalize for display if necessary (assuming ToTensor normalization [0, 1])
            image = (image * 255).astype(np.uint8) # Simple denormalization assumption
            axes[i].imshow(image)
            # Add original index to the title
            original_idx = misclassified_original_indices[i]
            axes[i].set_title(f"Idx: {original_idx}\nV: {classes[misclassified_labels[i]]}\nP: {classes[misclassified_preds[i]]}")
            axes[i].axis('off')
        st.pyplot(fig)
    else:
        st.write("Nenhuma imagem mal classificada encontrada.")

    # Return data for download including original indices
    # We will return the original indices, true labels, and predicted labels for the misclassified samples found.
    # We are not returning the image tensors themselves in the download data to avoid large file sizes.
    return misclassified_original_indices, misclassified_labels, misclassified_preds

def extract_features(dataset, model, batch_size):
    """
        # Se o ícone não for encontrado, carrega sem favicon
        st.set_page_config(page_title="Geomaker", layout="wide")
        logging.warning(f"Ícone {icon_path} não encontrado, carregando sem favicon.")
    
    # Layout da página
    if os.path.exists('capa.png'):
        st.image('capa.png', width=100, caption='Laboratório de Educação e Inteligência Artificial - Geomaker. "A melhor forma de prever o futuro é inventá-lo." - Alan Kay', use_container_width=True)
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

  
    num_classes = st.sidebar.number_input("Número de Classes:", min_value=1, step=1)
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
            - REZAEEZADE, A.; BATINA, L. Regularizers to the rescue: fighting overfitting in deeplearning-based side-channel analysis. 2022. https://doi.org/10.21203/rs.3.rs-2386625/v1
            - SAKIZADEH, M.; MALIAN, A.; AHMADPOUR, E. Groundwater quality modeling with a small data set. *Ground Water*, 54(1), 115-120, 2015. https://doi.org/10.1111/gwat.12317
            """)
    

  
    l2_lambda = st.sidebar.number_input("L2 Regularization (Weight Decay):", min_value=0.0, max_value=0.1, value=0.01, step=0.01)
    
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

    # Novos controles para opções avançadas
    st.sidebar.subheader("Opções Avançadas")

    data_augmentation_method = st.sidebar.selectbox(
        "Método de Aumento de Dados:",
        options=['Padrão', 'Mixup', 'Cutmix']
    )

    optimizer_name = st.sidebar.selectbox(
        "Otimizador:",
        options=['Adam', 'AdamW', 'Ranger', 'Lion', 'Sophia']
    )

    lr_scheduler_name = st.sidebar.selectbox(
        "Agendador de Taxa de Aprendizado:",
        options=['Nenhum', 'Recozimento por Cosseno', 'Política de Um Ciclo']
    )

    cam_method = st.sidebar.selectbox(
        "Método CAM para Visualização:",
        options=['Grad-CAM', 'Score-CAM', 'LayerCAM']
    )

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

    if zip_file is not None and num_classes > 0 and train_split + valid_split <= 0.95:
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.read())
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Add a small delay to ensure file system is ready after extraction
        time.sleep(2)

        # Check if the extracted content is a single subdirectory and use that as data_dir
        extracted_contents = os.listdir(temp_dir)
        if len(extracted_contents) == 1 and os.path.isdir(os.path.join(temp_dir, extracted_contents[0])):
            data_dir = os.path.join(temp_dir, extracted_contents[0])
            st.info(f"Detected single root directory '\\{extracted_contents[0]}\' inside the ZIP. Using this as the data directory.")
        else:
            data_dir = temp_dir
            st.info("Using the root of the extracted ZIP content as the data directory.")

        # Verify if data_dir itself is a valid directory
        if not os.path.isdir(data_dir):
            st.error(f"Erro crítico: O caminho do diretório de dados calculado \'{data_dir}\' não é um diretório válido ou acessível.")
            shutil.rmtree(temp_dir)
            return

        # Verify if data_dir contains subdirectories (classes)
        if not any(os.path.isdir(os.path.join(data_dir, item)) for item in os.listdir(data_dir)):
             st.error("Erro: O diretório de dados extraído não contém subdiretórios (classes). Certifique-se de que o arquivo ZIP contenha pastas, cada uma representando uma classe, com as imagens dentro.")
             shutil.rmtree(temp_dir)
             return

        st.info(f"Diretório de dados final para ImageFolder: {data_dir}")
        st.info(f"Conteúdo do diretório de dados: {os.listdir(data_dir)}")

        st.write("Iniciando o treinamento supervisionado...")
        model_data = train_model(data_dir, num_classes, model_name, fine_tune, epochs, learning_rate, batch_size, train_split, valid_split, use_weighted_loss, l2_lambda, patience, optimizer_name, lr_scheduler_name, data_augmentation_method)

        if model_data is None:
            st.error("Erro no treinamento do modelo.")
            shutil.rmtree(temp_dir)
            return

        model, classes, train_losses, valid_losses, train_accuracies, valid_accuracies = model_data
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

        # Carregar o dataset original sem transformações (será usado para splits e rótulos)
        # Retry loading the dataset with ImageFolder due to potential transient file system issues
        retry_attempts = 5 # Increased retry attempts
        full_dataset = None
        for attempt in range(retry_attempts):
            try:
                # Explicitly list directories to check structure before ImageFolder
                st.info(f"Attempt {attempt + 1}/{retry_attempts}: Checking directory structure in {data_dir}")
                subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
                if not subdirs:
                     st.warning(f"Attempt {attempt + 1}/{retry_attempts}: No subdirectories found in {data_dir}. This might cause ImageFolder to fail.")
                     if attempt < retry_attempts - 1:
                         time.sleep(3) # Wait before retrying
                         continue # Skip to next attempt
                     else:
                          st.error(f"After {retry_attempts} attempts, no subdirectories found in {data_dir}. Please check the structure of your ZIP file. It should contain subfolders for each class.")
                          shutil.rmtree(temp_dir)
                          return
                st.info(f"Attempt {attempt + 1}/{retry_attempts}: Found subdirectories: {subdirs}")

                full_dataset = datasets.ImageFolder(root=data_dir)
                st.success(f"Conjunto de dados carregado com sucesso em {data_dir} (Tentativa {attempt + 1}).")
                break # Exit retry loop if successful
            except FileNotFoundError as e: # pylint: disable=unused-variable # (e é usado no log abaixo)
                if attempt < retry_attempts - 1:
                    st.warning(f"Tentativa {attempt + 1}/{retry_attempts} falhou ao carregar o conjunto de dados. Diretório '{data_dir}' não encontrado. Retentando em 3 segundos...")
                    time.sleep(3) # Increased delay
                else:
                    st.error(f"Todas as {retry_attempts} tentativas falharam ao carregar o conjunto de dados. Diretório '{data_dir}' não encontrado. Verifique a estrutura do arquivo ZIP e as permissões no ambiente Streamlit Cloud. Detalhes do último erro: {e}")
                    shutil.rmtree(temp_dir)
                    return
            except Exception as e: # pylint: disable=unused-variable # (e é usado no log abaixo)
                st.error(f"Ocorreu um erro inesperado ao carregar o conjunto de dados (Tentativa {attempt + 1}): {e}")
                shutil.rmtree(temp_dir)
                return

        # If full_dataset is still None after retries, something went wrong
        if full_dataset is None:
             st.error("Falha ao carregar o conjunto de dados após múltiplas tentativas.")
             shutil.rmtree(temp_dir)
             return

        # Carregar o dataset completo para extração de características
        # Este full_dataset é para extração de features, o train_model usa o seu próprio loaded_full_dataset
        full_dataset = datasets.ImageFolder(root=data_dir, transform=test_transforms)
        features, labels = extract_features(full_dataset, feature_extractor, batch_size)

        # Aplicar algoritmos de clustering
        st.write("Aplicando algoritmos de clustering...")
        features_reshaped = features.reshape(len(features), -1)
        hierarchical_labels, kmeans_labels = perform_clustering(features_reshaped, num_classes)

        # Avaliar e exibir os resultados
        st.write("Avaliando os resultados do clustering...")
        hierarchical_ari, hierarchical_nmi = evaluate_clustering(labels, hierarchical_labels, "Clustering Hierárquico")
        kmeans_ari, kmeans_nmi = evaluate_clustering(labels, kmeans_labels, "K-Means Clustering")

        # Visualizar clusters
        visualize_clusters(features_reshaped, labels, hierarchical_labels, kmeans_labels, classes)

        # Visualize embeddings of Augmented Data
        st.subheader("Visualização de Embeddings Após Aumento de Dados (Conjunto de Treinamento)")
        # Create a dataset with training transforms applied to the full dataset
        augmented_full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)

        # Extract features from the augmented dataset
        augmented_features, augmented_labels = extract_features(augmented_full_dataset, feature_extractor, batch_size)

        # Reshape features for PCA
        augmented_features_reshaped = augmented_features.reshape(len(augmented_features), -1)

        # Reduce dimensionality with PCA for visualization
        pca_augmented = PCA(n_components=2)
        reduced_augmented_features = pca_augmented.fit_transform(augmented_features_reshaped)

        # Map true labels to class names
        augmented_true_labels_named = [classes[label] for label in augmented_labels]

        # Use distinct and visible colors
        augmented_color_palette = sns.color_palette("tab10", len(set(augmented_labels)))

        # Plot the augmented embeddings
        fig_augmented, ax_augmented = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=reduced_augmented_features[:, 0], y=reduced_augmented_features[:, 1], hue=augmented_true_labels_named, palette=augmented_color_palette, ax=ax_augmented, legend='full')
        ax_augmented.set_title('Embeddings do Conjunto de Treinamento (Após Aumento de Dados)')
        ax_augmented.set_xlabel('Componente Principal 1')
        ax_augmented.set_ylabel('Componente Principal 2')
        st.pyplot(fig_augmented)

        # Download Results Section
        st.subheader("Download de Resultados para Análise Detalhada")

        # Convert training metrics to CSV
        training_metrics_data = []
        # Ensure the metrics lists are not empty before iterating
        if train_losses and valid_losses and train_accuracies and valid_accuracies:
            for i in range(len(train_losses)):
                training_metrics_data.append({
                    "Epoch": i + 1,
                    "Train Loss": train_losses[i],
                    "Valid Loss": valid_losses[i],
                    "Train Accuracy": train_accuracies[i],
                    "Valid Accuracy": valid_accuracies[i]
                })
        training_metrics_csv = to_csv_string(training_metrics_data)
        st.download_button(
            label="Baixar Métricas de Treinamento (CSV)",
            data=training_metrics_csv,
            file_name='training_metrics.csv',
            mime='text/csv',
            key='download_training_metrics'
        )

        # Convert classification report to CSV
        # The to_csv_string function already handles the dictionary format
        classification_report_csv = to_csv_string(classification_report_dict)
        st.download_button(
            label="Baixar Relatório de Classificação (CSV)",
            data=classification_report_csv,
            file_name='classification_report.csv',
            mime='text/csv',
            key='download_classification_report'
        )

        # Convert confusion matrix to CSV
        # Need to add class names as header and potentially as the first column
        cm_data_with_labels = []
        # Ensure confusion_matrix_array is not empty before processing
        if confusion_matrix_array is not None and len(confusion_matrix_array) > 0:
            cm_data_with_labels.append([""] + classes) # Header row with empty string for the first cell
            for i, row in enumerate(confusion_matrix_array):
                # Ensure the row has the expected number of columns
                if len(row) == len(classes):
                     cm_data_with_labels.append([classes[i]] + list(row))
                else:
                     st.warning(f"Confusion matrix row {i} has unexpected number of columns: {len(row)}. Expected {len(classes)}.")

        confusion_matrix_csv = to_csv_string(cm_data_with_labels)
        st.download_button(
            label="Baixar Matriz de Confusão (CSV)",
            data=confusion_matrix_csv,
            file_name='confusion_matrix.csv',
            mime='text/csv',
            key='download_confusion_matrix'
        )

        # Convert misclassified info to CSV
        # misclassified_info is a tuple: (misclassified_original_indices, misclassified_labels, misclassified_preds)
        misclassified_data = []
        # Ensure misclassified_info is not None and contains the expected number of elements
        # Check if misclassified_info is not None and is a tuple of at least 3 elements
        if misclassified_info and isinstance(misclassified_info, tuple) and len(misclassified_info) >= 3:
            misclassified_original_indices_list, misclassified_labels_list, misclassified_preds_list = misclassified_info
            # Ensure the lists are not empty before iterating
            if misclassified_original_indices_list and misclassified_labels_list and misclassified_preds_list:
                # Find the minimum length among the lists to avoid index errors
                min_len = min(len(misclassified_original_indices_list), len(misclassified_labels_list), len(misclassified_preds_list))
                for i in range(min_len):
                     # Safely get data and class names
                     original_idx = misclassified_original_indices_list[i]
                     true_label_idx = misclassified_labels_list[i].item()
                     predicted_label_idx = misclassified_preds_list[i].item()

                     true_class_name = classes[true_label_idx] if true_label_idx < len(classes) else f"Unknown Index ({true_label_idx})"
                     predicted_class_name = classes[predicted_label_idx] if predicted_label_idx < len(classes) else f"Unknown Index ({predicted_label_idx})"

                     misclassified_data.append({
                         "Original Dataset Index": original_idx,
                         "True Label": true_class_name,
                         "Predicted Label": predicted_class_name
                     })

        misclassified_csv = to_csv_string(misclassified_data)

        st.download_button(
            label="Baixar Análise de Erros (CSV)",
            data=misclassified_csv,
            file_name='misclassified_analysis.csv',
            mime='text/csv',
            key='download_misclassified_analysis'
        )

        # Convert clustering metrics to CSV
        # Ensure clustering metrics variables are defined and not None
        if all(v is not None for v in [hierarchical_ari, hierarchical_nmi, kmeans_ari, kmeans_nmi]):
            clustering_metrics_data = [
                {"Clustering Method": "Hierarchical", "Adjusted Rand Index": hierarchical_ari, "Normalized Mutual Information Score": hierarchical_nmi},
                {"Clustering Method": "K-Means", "Adjusted Rand Index": kmeans_ari, "Normalized Mutual Information Score": kmeans_nmi},
            ]
            clustering_metrics_csv = to_csv_string(clustering_metrics_data)
            st.download_button(
                label="Baixar Métricas de Clustering (CSV)",
                data=clustering_metrics_csv,
                file_name='clustering_metrics.csv',
                mime='text/csv',
                key='download_clustering_metrics'
            )

        # Provide a link/button to download the saved JSON config file
        config_filename = f"config_{model_name}_run1.json"
        config_filepath = os.path.join("./configs", config_filename)
        if os.path.exists(config_filepath):
            try:
                with open(config_filepath, "r") as f:
                     config_json_content = f.read()
                st.download_button(
                     label="Baixar Configuração de Treinamento (JSON)",
                     data=config_json_content,
                     file_name=config_filename,
                     mime='application/json',
                     key='download_config_json'
                 )
            except Exception as e:
                 st.warning(f"Erro ao ler o arquivo de configuração {config_filename} para download: {e}")
        else:
             st.warning(f"Arquivo de configuração {config_filename} não encontrado para download.")

        # Add this section to display training configuration for the completed run
        st.subheader("Configurações Técnicas do Treinamento Realizado")
        config_data = [
            {"Parâmetro": "Modelo", "Valor": str(model_name)},
            {"Parâmetro": "Fine-Tuning Completo", "Valor": "Sim" if fine_tune else "Não"},
            {"Parâmetro": "Épocas", "Valor": str(epochs)},
            {"Parâmetro": "Taxa de Aprendizagem", "Valor": str(learning_rate)},
            {"Parâmetro": "Tamanho de Lote", "Valor": str(batch_size)},
            {"Parâmetro": "Train Split", "Valor": str(train_split)},
            {"Parâmetro": "Valid Split", "Valor": str(valid_split)},
            {"Parâmetro": "L2 Regularization", "Valor": str(l2_lambda)},
            {"Parâmetro": "Paciência Early Stopping", "Valor": str(patience)},
            {"Parâmetro": "Use Weighted Loss", "Valor": "Sim" if use_weighted_loss else "Não"},
            {"Parâmetro": "Otimizador", "Valor": str(optimizer_name)},
            {"Parâmetro": "Agendador LR", "Valor": str(lr_scheduler_name)},
            {"Parâmetro": "Aumento de Dados", "Valor": str(data_augmentation_method)},
        ]
        st.table(config_data)
        save_config_to_json(config_data, model_name)

        # Limpar o diretório temporário
        shutil.rmtree(temp_dir)
    # Fim do bloco if zip_file is not None

if __name__ == "__main__":
    main()
