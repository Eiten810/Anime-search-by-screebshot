from pathlib import Path
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from google import genai
from tqdm.auto import tqdm
import os
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F



def load_gemini_client():
    """
    Загружает Gemini API-клиент.

    Returns:
        genai.Client: Клиент Gemini.
    """

    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    api_key = os.getenv("GEMINI_API_KEY")

    if api_key is None:
        raise ValueError("GEMINI_API_KEY not found in .env")

    return genai.Client(api_key=api_key)


def get_gemini_embedding(image_path, client, mode="original"):
    """
    Получает эмбеддинг изображения с помощью модели Gemini Embedding 2.

    Изображение открывается, конвертируется в формат RGB и,
    при необходимости, масштабируется до 512×512 пикселей.
    После этого оно передается в Gemini API для получения эмбеддинга.

    Args:
        image_path (Path): Путь к изображению.

    Returns:
        list[float]: Вектор эмбеддинга изображения.
    """
    # Загружаем изображение и приводим его к формату RGB
    img = Image.open(image_path).convert("RGB")

    # При необходимости изменяем размер изображения
    if mode == "512":
        img = img.resize((512, 512))

    # Получаем эмбеддинг изображения через Gemini API
    response = client.models.embed_content(
        model="models/gemini-embedding-2",
        contents=img
    )

    return np.asarray(
        response.embeddings[0].values,
        dtype=np.float32,
    )


def save_database(save_dir, embeddings, labels, paths):
    """
    Сохраняет базу эмбеддингов.
    """

    np.save(
        save_dir / "embeddings.npy",
        np.asarray(embeddings, dtype=np.float32)
    )

    np.save(
        save_dir / "labels.npy",
        np.asarray(labels)
    )

    np.save(
        save_dir / "paths.npy",
        np.asarray(paths)
    )


def load_database(save_dir):
    """
    Загружает существующую базу эмбеддингов.
    """

    emb_file = save_dir / "embeddings.npy"

    if not emb_file.exists():
        return [], [], []

    embeddings = np.load(save_dir / "embeddings.npy", allow_pickle=True).tolist()
    labels = np.load(save_dir / "labels.npy", allow_pickle=True).tolist()
    paths = np.load(save_dir / "paths.npy", allow_pickle=True).tolist()

    return embeddings, labels, paths


def generate_gemini_embeddings(
    train_images,
    save_dir: Path,
    client,
    mode: str = "original",
    save_step: int = 100
):
    """
    Генерирует эмбеддинги изображений с помощью Gemini Embedding 2.

    Поддерживает продолжение после остановки (resume)
    и периодическое сохранение базы эмбеддингов.
    """

    embeddings, labels, paths = load_database(save_dir)

    if len(paths):
        print(f"Продолжаем: {len(paths)}")
    else:
        print("Новый запуск")

    processed = set(paths)
    counter = len(paths)

    for sample in tqdm(
        train_images,
        desc="Embedding train images"
    ):

        image_path = sample["path"]
        label = sample["label"]

        if image_path in processed:
            continue

        try:

            embedding = get_gemini_embedding(
                image_path=image_path,
                client=client,
                mode=mode
            )

            embeddings.append(embedding)
            labels.append(label)
            paths.append(image_path)

            counter += 1

            if counter % save_step == 0:

                save_database(
                    save_dir,
                    embeddings,
                    labels,
                    paths
                )

                print(f"→ {counter}")

        except Exception as e:
            print(f"{image_path}: {e}")

    save_database(
        save_dir,
        embeddings,
        labels,
        paths
    )

    print(f"\nГотово: {len(embeddings)}")

    return embeddings, labels, paths


def load_resnet_model(models_path: Path, device):
    """
    Загружает обученную модель ResNet50 для извлечения эмбеддингов.

    Args:
        models_path (Path): Путь к директории с моделями.
        device (torch.device): Устройство для вычислений.

    Returns:
        torch.nn.Module: Модель ResNet50 в режиме инференса.
    """

    # Создаем архитектуру модели
    resnet50 = models.resnet50(weights=None)

    # Восстанавливаем классификационную голову для загрузки весов
    resnet50.fc = nn.Linear(resnet50.fc.in_features, 23)

    # Загружаем обученные веса
    checkpoint = torch.load(
        models_path / "best_resnet50.pth",
        map_location=device
    )

    resnet50.load_state_dict(checkpoint)

    # Удаляем классификатор, оставляя только экстрактор признаков
    resnet50.fc = nn.Identity()

    # Переводим модель на устройство
    resnet50 = resnet50.to(device)
    resnet50.eval()

    print("Classification ResNet loaded.")

    return resnet50


def get_resnet_embedding(
    image_path: Path,
    model,
    transform,
    device
):
    """
    Извлекает эмбеддинг изображения с помощью обученной модели ResNet50.

    Изображение загружается, проходит необходимые преобразования,
    затем передается в модель. Полученный эмбеддинг L2-нормализуется
    и возвращается в виде массива NumPy.

    Args:
        image_path (Path): Путь к изображению.
        model: Загруженная модель ResNet50.
        transform: Преобразования изображения.
        device: Устройство для вычислений.

    Returns:
        np.ndarray: Нормализованный эмбеддинг изображения.
    """

    # Загружаем изображение
    image = Image.open(image_path).convert("RGB")

    # Подготавливаем изображение для модели
    image = transform(image).unsqueeze(0).to(device)

    # Получаем эмбеддинг
    with torch.no_grad():
        embedding = model(image)

    # L2-нормализация
    embedding = F.normalize(embedding, p=2, dim=1)

    return embedding.squeeze(0).cpu().numpy()


def generate_resnet_embeddings(
    train_images,
    save_dir: Path,
    model,
    transform,
    device,
    save_step: int = 100
):
    """
    Генерирует эмбеддинги изображений с помощью обученной модели ResNet50.

    Поддерживает продолжение после остановки (resume)
    и периодическое сохранение базы эмбеддингов.
    """

    embeddings, labels, paths = load_database(save_dir)

    if len(paths):
        print(f"Продолжаем: {len(paths)}")
    else:
        print("Новый запуск")

    processed = set(paths)
    counter = len(paths)

    for sample in tqdm(train_images, desc="Embedding train images"):

        image_path = Path(sample["path"])
        label = sample["label"]

        # Пропускаем уже обработанные изображения
        if str(image_path) in processed:
            continue

        # Пропускаем неподдерживаемые форматы
        if image_path.suffix.lower() not in {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
        }:
            continue

        embedding = get_resnet_embedding(
            image_path=image_path,
            model=model,
            transform=transform,
            device=device,
        )

        embeddings.append(embedding)
        labels.append(label)
        paths.append(str(image_path))

        counter += 1

        if counter % save_step == 0:
            save_database(
                save_dir,
                embeddings,
                labels,
                paths,
            )

            print(f"→ {counter}")

    save_database(
        save_dir,
        embeddings,
        labels,
        paths,
    )

    print(f"\nГотово: {len(embeddings)}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels)
    paths = np.asarray(paths)

    return embeddings, labels, paths

class EmbeddingNet(nn.Module):
    """
    Нейронная сеть для получения эмбеддингов изображений.

    В качестве backbone используется ResNet50 без классификационной головы.
    Вместо нее добавляется собственная проекционная голова, которая преобразует
    признаки в эмбеддинг заданной размерности. На выходе эмбеддинги
    L2-нормализуются.

    Args:
        embedding_size (int): Размерность выходного эмбеддинга.
    """

    def __init__(self, embedding_size):
        super().__init__()

        # Загружаем архитектуру ResNet50
        self.backbone = models.resnet50(weights=None)

        # Получаем размерность признаков перед классификатором
        features = self.backbone.fc.in_features

        # Удаляем стандартную классификационную голову
        self.backbone.fc = nn.Identity()

        # Проекционная голова для получения эмбеддингов
        self.head = nn.Sequential(
            nn.Linear(features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, embedding_size)
        )

    def forward(self, x):
        """
        Выполняет прямой проход модели.

        Args:
            x (torch.Tensor): Батч изображений.

        Returns:
            torch.Tensor: L2-нормализованные эмбеддинги.
        """
        x = self.backbone(x)
        x = self.head(x)

        # Нормализуем эмбеддинги для последующего сравнения
        return F.normalize(x, p=2, dim=1)


def load_triplet_resnet_model(models_path: Path, device):
    """
    Загружает обученную Triplet ResNet50 для извлечения эмбеддингов.
    """

    checkpoint = torch.load(
        models_path / "best_triplet_resnet50.pth",
        map_location=device
    )

    model = EmbeddingNet(
        checkpoint["embedding_size"]
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Triplet ResNet loaded.")

    return model

def get_triplet_embedding(
    image_path: Path,
    model,
    transform,
    device
):
    """
    Извлекает эмбеддинг изображения с помощью обученной Triplet ResNet50.

    Args:
        image_path (Path): Путь к изображению.
        model (torch.nn.Module): Загруженная Triplet ResNet50.
        transform: Преобразования изображения.
        device (torch.device): Устройство для вычислений.

    Returns:
        np.ndarray: Эмбеддинг изображения.
    """

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(image)

    return embedding.squeeze(0).cpu().numpy()


def generate_triplet_embeddings(
    train_images,
    save_dir: Path,
    model,
    transform,
    device,
    save_step: int = 100
):
    """
    Генерирует эмбеддинги изображений с помощью обученной Triplet ResNet50.

    Поддерживает продолжение после остановки (resume)
    и периодическое сохранение базы эмбеддингов.
    """

    embeddings, labels, paths = load_database(save_dir)

    if len(paths):
        print(f"Продолжаем: {len(paths)}")
    else:
        print("Новый запуск")

    processed = set(paths)
    counter = len(paths)

    for sample in tqdm(train_images, desc="Embedding train images"):

        image_path = Path(sample["path"])
        label = sample["label"]

        # Пропускаем уже обработанные изображения
        if str(image_path) in processed:
            continue

        # Пропускаем неподдерживаемые форматы
        if image_path.suffix.lower() not in {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
        }:
            continue

        embedding = get_triplet_embedding(
            image_path=image_path,
            model=model,
            transform=transform,
            device=device,
        )

        embeddings.append(embedding)
        labels.append(label)
        paths.append(str(image_path))

        counter += 1

        if counter % save_step == 0:
            save_database(
                save_dir,
                embeddings,
                labels,
                paths,
            )

            print(f"→ {counter}")

    save_database(
        save_dir,
        embeddings,
        labels,
        paths,
    )

    print(f"\nГотово: {len(embeddings)}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels)
    paths = np.asarray(paths)

    return embeddings, labels, paths