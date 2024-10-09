import os
import zipfile
import shutil
import tempfile
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms, models, datasets
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve)
from sklearn.decomposition import PCA
import streamlit as st

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

# Transformações comuns para as imagens
common_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Dataset personalizado com aumento de dados
class AugmentedDataset(torch.utils.data.Dataset):
    """
    Dataset que aplica transformações de aumento de dados às imagens originais.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.transforms_list = [
            transforms.Compose([]),  # Imagem original
            transforms.Compose([transforms.RandomHorizontalFlip(p=1)]),  # Inversão horizontal
            transforms.Compose([transforms.RandomRotation((45, 45))]),  # Rotação de 45 graus
            transforms.Compose([transforms.RandomRotation((90, 90))]),   # Rotação de 90 graus
            transforms.Compose([transforms.ColorJitter(brightness=0.5, contrast=0.5)]),  # Variação de brilho e contraste
            transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.8, 1.0))]),  # Zoom
            transforms.Compose([transforms.RandomAffine(degrees=0, shear=10)]),  # Shear
        ]

    def __len__(self):
        return len(self.dataset) * len(self.transforms_list)

    def __getitem__(self, idx):
        original_idx = idx // len(self.transforms_list)
        transform_idx = idx % len(self.transforms_list)
        image, label = self.dataset[original_idx]
        transform = self.transforms_list[transform_idx]
        image = transform(image)
        image = common_transforms(image)
        return image, label

def visualize_data(dataset, classes):
    """
    Exibe algumas imagens do dataset com suas classes.
    """
    st.write("Visualização de algumas imagens do conjunto de dados:")
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        idx = np.random.randint(len(dataset))
        image, label = dataset[idx]
        image = image.permute(1, 2, 0).numpy()
        axes[i].imshow(image)
        axes[i].set_title(classes[label])
        axes[i].axis('off')
    st.pyplot(fig)

def plot_class_distribution(dataset, classes):
    """
    Exibe a distribuição das classes no conjunto de dados.
    """
    labels = [label for _, label in dataset]
    fig, ax = plt.subplots()
    sns.countplot(labels, ax=ax)
    ax.set_xticklabels(classes)
    ax.set_title("Distribuição das Classes")
    st.pyplot(fig)

class CustomResNet(nn.Module):
    """
    Modelo ResNet personalizado com Dropout.
    """
    def __init__(self, num_classes, dropout_p=0.5):
        super(CustomResNet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        # Congelar as camadas convolucionais
        for param in self.model.parameters():
            param.requires_grad = False
        # Adicionar Dropout e camada totalmente conectada
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x

def train_model(data_dir, num_classes, epochs, learning_rate, batch_size, train_split, valid_split, use_weighted_loss, l2_lambda, patience):
    """
    Função principal para treinamento do modelo.
    """
    # Carregar o dataset original sem transformações
    full_dataset = datasets.ImageFolder(root=data_dir)

    # Exibir algumas imagens do dataset
    visualize_data(full_dataset, full_dataset.classes)
    plot_class_distribution(full_dataset, full_dataset.classes)

    # Criar o dataset aumentado
    augmented_dataset = AugmentedDataset(full_dataset)

    # Obter o número de classes detectadas
    detected_classes = len(full_dataset.classes)
    if detected_classes != num_classes:
        st.error(f"O número de classes detectadas ({detected_classes}) não coincide com o número fornecido ({num_classes}).")
        return None

    # Dividir o dataset em treino, validação e teste
    dataset_size = len(augmented_dataset)
    test_size = int((1 - train_split - valid_split) * dataset_size)
    valid_size = int(valid_split * dataset_size)
    train_size = dataset_size - valid_size - test_size

    train_dataset, valid_dataset, test_dataset = random_split(
        augmented_dataset, [train_size, valid_size, test_size],
        generator=torch.Generator().manual_seed(42))

    # Dataloaders
    if use_weighted_loss:
        # Calcular pesos para a perda ponderada
        targets = [augmented_dataset[i][1] for i in train_dataset.indices]
        class_counts = np.bincount(targets)
        class_weights = 1.0 / class_counts
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Carregar o modelo
    model = CustomResNet(num_classes, dropout_p=0.5)
    model = model.to(device)

    # Definir o otimizador com L2 regularization (weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)

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
        running_loss = 0.0
        running_corrects = 0
        model.train()

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size
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

        valid_epoch_loss = valid_running_loss / valid_size
        valid_epoch_acc = valid_running_corrects.double() / valid_size
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
    evaluate_model(model, test_loader, full_dataset.classes, num_classes)

    # Análise de Erros
    st.write("**Análise de Erros**")
    error_analysis(model, test_loader, full_dataset.classes)

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

def evaluate_model(model, dataloader, classes, num_classes):
    """
    Avalia o modelo no conjunto fornecido e exibe métricas detalhadas.
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
    report = classification_report(all_labels, all_preds, target_names=classes)
    st.text("Relatório de Classificação:")
    st.text(report)

    # Matriz de Confusão
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predito')
    ax.set_ylabel('Verdadeiro')
    ax.set_title('Matriz de Confusão')
    st.pyplot(fig)

    # Curva ROC
    if num_classes == 2:
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

            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    misclassified_images.append(inputs[i].cpu())
                    misclassified_labels.append(labels[i].cpu())
                    misclassified_preds.append(preds[i].cpu())

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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
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
    Visualiza os clusters usando redução de dimensionalidade.
    """
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Clustering Hierárquico
    sns.scatterplot(x=reduced_features[:,0], y=reduced_features[:,1], hue=hierarchical_labels, palette="deep", ax=axes[0], legend='full')
    axes[0].set_title('Clustering Hierárquico')

    # K-Means
    sns.scatterplot(x=reduced_features[:,0], y=reduced_features[:,1], hue=kmeans_labels, palette="deep", ax=axes[1], legend='full')
    axes[1].set_title('K-Means Clustering')

    st.pyplot(fig)

def evaluate_image(model, image, classes):
    """
    Avalia uma única imagem e retorna a classe predita e a confiança.
    """
    model.eval()
    image = common_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        class_idx = predicted.item()
        class_name = classes[class_idx]
        return class_name, confidence.item()

def main():
    set_seed(42)  # Definir a seed para reprodutibilidade

    st.title("Classificação e Clustering de Imagens com Aprendizado Profundo")
    st.write("Este aplicativo permite treinar um modelo de classificação de imagens e aplicar algoritmos de clustering para análise comparativa.")

    # Barra Lateral de Configurações
    st.sidebar.title("Configurações do Treinamento")
    num_classes = st.sidebar.number_input("Número de Classes:", min_value=1, step=1)
    epochs = st.sidebar.slider("Número de Épocas:", min_value=1, max_value=50, value=5, step=1)
    learning_rate = st.sidebar.select_slider("Taxa de Aprendizagem:", options=[0.1, 0.01, 0.001, 0.0001], value=0.001)
    batch_size = st.sidebar.selectbox("Tamanho de Lote:", options=[4, 8, 16, 32, 64], index=2)
    train_split = st.sidebar.slider("Percentual de Treinamento:", min_value=0.5, max_value=0.9, value=0.7, step=0.05)
    valid_split = st.sidebar.slider("Percentual de Validação:", min_value=0.05, max_value=0.4, value=0.15, step=0.05)
    l2_lambda = st.sidebar.number_input("L2 Regularization (Weight Decay):", min_value=0.0, max_value=0.1, value=0.01, step=0.01)
    patience = st.sidebar.number_input("Paciência para Early Stopping:", min_value=1, max_value=10, value=3, step=1)
    use_weighted_loss = st.sidebar.checkbox("Usar Perda Ponderada para Classes Desbalanceadas", value=False)

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
        model_data = train_model(data_dir, num_classes, epochs, learning_rate, batch_size, train_split, valid_split, use_weighted_loss, l2_lambda, patience)

        if model_data is None:
            st.error("Erro no treinamento do modelo.")
            shutil.rmtree(temp_dir)
            return

        model, classes = model_data
        st.success("Treinamento concluído!")

        # Extrair características usando o modelo pré-treinado (sem a camada final)
        st.write("Extraindo características para clustering...")
        # Remover a última camada do modelo para obter embeddings
        feature_extractor = nn.Sequential(*list(model.model.children())[:-1])
        feature_extractor = feature_extractor.to(device)
        feature_extractor.eval()

        # Carregar o dataset completo para extração de características
        full_dataset = datasets.ImageFolder(root=data_dir, transform=common_transforms)
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

        # Limpar o diretório temporário
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
