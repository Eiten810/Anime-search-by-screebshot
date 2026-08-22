import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class EmbeddingNet(nn.Module):
    """
    Нейронная сеть для получения эмбеддингов изображений.

    В качестве backbone используется предобученная ResNet50.
    Вместо стандартной классификационной головы используется
    проекционная голова, формирующая эмбеддинги заданной размерности.
    На выходе эмбеддинги L2-нормализуются.
    """

    def __init__(self, embedding_size):
        super().__init__()

        # Загружаем предобученную архитектуру ResNet50
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )

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