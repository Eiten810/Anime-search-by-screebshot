from tqdm.auto import tqdm
import torch


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Выполняет одну эпоху обучения модели.

    Args:
        model (nn.Module): Обучаемая модель.
        loader (DataLoader): Загрузчик обучающих данных.
        criterion (nn.Module): Функция потерь.
        optimizer (torch.optim.Optimizer): Оптимизатор модели.

    Returns:
        tuple[float, float]:
            Среднее значение функции потерь и точность за эпоху.
    """
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader):
        # Переносим данные на устройство вычислений
        images = images.to(device)
        labels = labels.to(device)

        # Обнуляем накопленные градиенты
        optimizer.zero_grad()

        # Получаем предсказания модели
        outputs = model(images)

        # Вычисляем значение функции потерь
        loss = criterion(outputs, labels)

        # Вычисляем градиенты и обновляем веса модели
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Подсчитываем количество верных предсказаний
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return (
        total_loss / len(loader),
        correct / total
    )


def validate_epoch(model, loader, criterion, device):
    """
    Выполняет одну эпоху валидации модели.

    Args:
        model (nn.Module): Проверяемая модель.
        loader (DataLoader): Загрузчик валидационных данных.
        criterion (nn.Module): Функция потерь.

    Returns:
        tuple[float, float]:
            Среднее значение функции потерь и точность за эпоху.
    """
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in tqdm(loader):
            # Переносим данные на устройство вычислений
            images = images.to(device)
            labels = labels.to(device)

            # Получаем предсказания модели
            outputs = model(images)

            # Вычисляем значение функции потерь
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            # Подсчитываем количество верных предсказаний
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return (
        total_loss / len(loader),
        correct / total
    )


def train_triplet_epoch(model, loader, criterion, optimizer, device):
    """
    Выполняет одну эпоху обучения модели.

    Args:
        model (nn.Module): Обучаемая модель.
        loader (DataLoader): Загрузчик обучающих данных.
        criterion (nn.Module): Функция потерь.
        optimizer (torch.optim.Optimizer): Оптимизатор модели.

    Returns:
        float: Среднее значение функции потерь за эпоху.
    """
    model.train()

    total_loss = 0.0

    for anchor, positive, negative in tqdm(loader):
        # Переносим данные на устройство вычислений
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        # Обнуляем накопленные градиенты
        optimizer.zero_grad()

        # Получаем эмбеддинги изображений
        emb_anchor = model(anchor)
        emb_positive = model(positive)
        emb_negative = model(negative)

        # Вычисляем Triplet Loss
        loss = criterion(
            emb_anchor,
            emb_positive,
            emb_negative
        )

        # Выполняем обратное распространение ошибки
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate_triplet(model, loader, criterion, device):
    """
    Выполняет одну эпоху валидации модели.

    Args:
        model (nn.Module): Проверяемая модель.
        loader (DataLoader): Загрузчик валидационных данных.
        criterion (nn.Module): Функция потерь.

    Returns:
        float: Среднее значение функции потерь за эпоху.
    """
    model.eval()

    total_loss = 0.0

    for anchor, positive, negative in loader:
        # Переносим данные на устройство вычислений
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        # Получаем эмбеддинги изображений
        emb_anchor = model(anchor)
        emb_positive = model(positive)
        emb_negative = model(negative)

        # Вычисляем значение функции потерь
        loss = criterion(
            emb_anchor,
            emb_positive,
            emb_negative
        )

        total_loss += loss.item()

    return total_loss / len(loader)