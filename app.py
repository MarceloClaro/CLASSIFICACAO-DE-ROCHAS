import os
import zipfile
import shutil
import tempfile
import numpy as np
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
import streamlit as st

# Definir o dispositivo (CPU ou GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformações para as imagens de treinamento e avaliação
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def train_model(data_dir, num_classes):
    # Criar o dataset e o dataloader
    dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Obter o número de classes detectadas
    detected_classes = len(dataset.classes)
    if detected_classes != num_classes:
        st.error(f"O número de classes detectadas ({detected_classes}) não coincide com o número fornecido ({num_classes}).")
        return None

    # Carregar um modelo pré-treinado
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features

    # Ajustar a camada final para o número de classes
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # Definir a função de perda e o otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Treinar por algumas épocas
    epochs = 3  # Você pode ajustar o número de épocas
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataset)
        st.write(f'Época {epoch+1}/{epochs}, Perda: {epoch_loss:.4f}')

    return model, dataset.classes

def evaluate_image(model, image, classes):
    model.eval()
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

    # Entrada para o número de classes
    num_classes = st.number_input("Insira o número de classes:", min_value=1, step=1)

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
        model_data = train_model(data_dir, num_classes)

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
                # Carregar a imagem
                eval_image = Image.open(eval_image_file).convert("RGB")

                # Exibir a imagem
                st.image(eval_image, caption='Imagem para avaliação', use_column_width=True)

                # Avaliar a imagem
                class_name, confidence = evaluate_image(model, eval_image, classes)
                st.write(f"**Classe Predita:** {class_name}")
                st.write(f"**Confiança:** {confidence:.4f}")

        # Limpar o diretório temporário
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
