import os
import zipfile
import shutil
import tempfile
import numpy as np
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models, datasets
import streamlit as st
import matplotlib.pyplot as plt

# Definir o dispositivo (CPU ou GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformações comuns para as imagens
common_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Transformações para a imagem de avaliação
eval_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Dataset personalizado com aumento de dados
class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transforms_list = [
            transforms.Compose([]),  # Imagem original
            transforms.Compose([transforms.RandomHorizontalFlip(p=1)]),  # Inversão horizontal
            transforms.Compose([transforms.RandomRotation((45, 45))]),  # Rotação de 45 graus
            transforms.Compose([transforms.RandomRotation((90, 90))])   # Rotação de 90 graus
        ]

    def __len__(self):
        return len(self.dataset) * len(self.transforms_list)

    def __getitem__(self, idx):
        # Índice da imagem original
        original_idx = idx // len(self.transforms_list)
        # Índice da transformação
        transform_idx = idx % len(self.transforms_list)

        image, label = self.dataset[original_idx]
        transform = self.transforms_list[transform_idx]
        image = transform(image)
        image = common_transforms(image)
        return image, label

def train_model(data_dir, num_classes, epochs, learning_rate, batch_size, train_valid_split):
    # Carregar o dataset original sem transformações
    full_dataset = datasets.ImageFolder(root=data_dir)

    # Criar o dataset aumentado
    augmented_dataset = AugmentedDataset(full_dataset)

    # Obter o número de classes detectadas
    detected_classes = len(full_dataset.classes)
    if detected_classes != num_classes:
        st.error(f"O número de classes detectadas ({detected_classes}) não coincide com o número fornecido ({num_classes}).")
        return None

    # Dividir o dataset em treino e validação
    dataset_size = len(augmented_dataset)
    split = int(np.floor(train_valid_split * dataset_size))
    train_size = dataset_size - split
    valid_size = split

    train_dataset, valid_dataset = random_split(augmented_dataset, [train_size, valid_size])

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Carregar um modelo pré-treinado
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features

    # Ajustar a camada final para o número de classes
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # Definir a função de perda e o otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Listas para armazenar as perdas e acurácias
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

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

        st.write(f'Época {epoch+1}/{epochs}')
        st.write(f'Perda de Treino: {epoch_loss:.4f} | Acurácia de Treino: {epoch_acc:.4f}')
        st.write(f'Perda de Validação: {valid_epoch_loss:.4f} | Acurácia de Validação: {valid_epoch_acc:.4f}')

    # Gráficos
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Gráfico de Perda
    ax[0].plot(range(1, epochs+1), train_losses, label='Treino')
    ax[0].plot(range(1, epochs+1), valid_losses, label='Validação')
    ax[0].set_title('Perda por Época')
    ax[0].set_xlabel('Épocas')
    ax[0].set_ylabel('Perda')
    ax[0].legend()

    # Gráfico de Acurácia
    ax[1].plot(range(1, epochs+1), train_accuracies, label='Treino')
    ax[1].plot(range(1, epochs+1), valid_accuracies, label='Validação')
    ax[1].set_title('Acurácia por Época')
    ax[1].set_xlabel('Épocas')
    ax[1].set_ylabel('Acurácia')
    ax[1].legend()

    st.pyplot(fig)
    plt.close(fig)

    return model, full_dataset.classes

def evaluate_image(model, image, classes):
    model.eval()
    # Aplicar as transformações de avaliação na imagem
    image = eval_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        class_idx = predicted.item()
        class_name = classes[class_idx]
        confidence = torch.nn.functional.softmax(output, dim=1)[0][class_idx].item()
    return class_name, confidence

def main():
    st.title("Treinamento e Avaliação de Imagens")

    st.write("Faça upload de um arquivo ZIP contendo imagens organizadas em pastas por classe para treinar o modelo.")

    # Barra Lateral de Configurações
    st.sidebar.title("Configurações do Treinamento")
    num_classes = st.sidebar.number_input("Número de Classes:", min_value=1, step=1)
    epochs = st.sidebar.number_input("Número de Épocas:", min_value=1, value=5, step=1)
    learning_rate = st.sidebar.select_slider("Taxa de Aprendizagem:", options=[0.1, 0.01, 0.001, 0.0001], value=0.001)
    batch_size = st.sidebar.selectbox("Tamanho de Lote:", options=[4, 8, 16, 32, 64], index=1)
    train_valid_split = st.sidebar.slider("Percentual de Validação:", min_value=0.0, max_value=0.9, value=0.2, step=0.05)

    # Upload do arquivo ZIP
    zip_file = st.file_uploader("Upload do arquivo ZIP", type=["zip"])

    if zip_file is not None and num_classes > 0:
        # Criar um diretório temporário para extrair as imagens
        temp_dir = tempfile.mkdtemp()

        # Salvar o arquivo ZIP no diretório temporário
        zip_path = os.path.join(temp_dir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.read())

        # Extrair o arquivo ZIP
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        data_dir = temp_dir  # O diretório com as imagens é o temp_dir

        st.write("Iniciando o treinamento...")

        # Treinar o modelo
        model_data = train_model(data_dir, num_classes, epochs, learning_rate, batch_size, train_valid_split)

        if model_data is None:
            st.error("Erro no treinamento do modelo.")
            shutil.rmtree(temp_dir)
            return

        model, classes = model_data

        st.success("Treinamento concluído!")

        # Perguntar se o usuário deseja avaliar uma imagem
        evaluate = st.radio("Deseja avaliar uma imagem?", ("Sim", "Não"))

        if evaluate == "Sim":
            # Upload da imagem para avaliação
            eval_image_file = st.file_uploader("Faça upload da imagem para avaliação", type=["png", "jpg", "jpeg", "bmp", "gif"])

            if eval_image_file is not None:
                # Redefinir o ponteiro do arquivo para o início
                eval_image_file.seek(0)
                # Tentar abrir a imagem diretamente do arquivo enviado
                try:
                    eval_image = Image.open(eval_image_file).convert("RGB")
                except Exception as e:
                    st.error(f"Erro ao abrir a imagem: {e}")
                    return

                # Exibir a imagem
                st.image(eval_image, caption='Imagem para avaliação', width='stretch')

                # Avaliar a imagem
                class_name, confidence = evaluate_image(model, eval_image, classes)
                st.write(f"**Classe Predita:** {class_name}")
                st.write(f"**Confiança:** {confidence:.4f}")

        # Limpar o diretório temporário
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
