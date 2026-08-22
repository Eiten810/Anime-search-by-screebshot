import random
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict
import random




def collect_dataset(root_path):
    """
    Собирает информацию об изображениях датасета.

    Args:
        root_path (Path): Путь к директории с датасетом.

    Returns:
        list[dict]: Список словарей с путями к изображениям и их метками.
    """
    dataset = []

    for anime_folder in root_path.iterdir():
        # Пропускаем файлы, оставляя только директории с классами
        if not anime_folder.is_dir():
            continue

        # Собираем информацию обо всех изображениях класса
        for img_path in anime_folder.glob("*.jpg"):
            dataset.append({
                "path": str(img_path),
                "label": anime_folder.name
            })

    return dataset


# Загружаем информацию о датасете
class AnimeDataset(Dataset):
    """
    Датасет для обучения модели классификации ResNet50.

    Args:
        dataset (list[dict]): Список изображений и их меток.
        class_to_idx (dict): Словарь соответствия названия класса его индексу.
        transform (callable, optional): Преобразования изображений.
    """

    def __init__(
        self,
        dataset,
        class_to_idx,
        transform=None,
    ):
        self.dataset = dataset
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        sample = self.dataset[idx]

        image = Image.open(sample["path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[sample["label"]]

        return image, label


class TripletDataset(Dataset):
    """
    Датасет для обучения с использованием Triplet Loss.

    Для каждого изображения формируется триплет:
    - anchor — исходное изображение;
    - positive — другое изображение того же класса;
    - negative — изображение другого класса.

    Args:
        images (list[tuple]): Список пар (путь к изображению, метка класса).
        transform (callable): Преобразования, применяемые к изображениям.
    """

    def __init__(self, images, transform):
        self.images = images
        self.transform = transform

        # Группируем пути к изображениям по классам
        self.by_label = defaultdict(list)

        for path, label in images:
            self.by_label[label].append(path)

        self.labels = list(self.by_label.keys())

    def __len__(self):
        """
        Возвращает количество изображений в датасете.

        Returns:
            int: Размер датасета.
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        Возвращает один триплет изображений.

        Args:
            index (int): Индекс anchor-изображения.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                Anchor, Positive и Negative изображения.
        """
        anchor_path, anchor_label = self.images[index]

        # Выбираем положительный пример из того же класса
        positive_path = anchor_path
        while positive_path == anchor_path:
            positive_path = random.choice(self.by_label[anchor_label])

        # Выбираем отрицательный пример из другого класса
        negative_label = random.choice(self.labels)
        while negative_label == anchor_label:
            negative_label = random.choice(self.labels)

        negative_path = random.choice(self.by_label[negative_label])

        # Загружаем изображения и применяем преобразования
        anchor = self.transform(Image.open(anchor_path).convert("RGB"))
        positive = self.transform(Image.open(positive_path).convert("RGB"))
        negative = self.transform(Image.open(negative_path).convert("RGB"))

        return anchor, positive, negative