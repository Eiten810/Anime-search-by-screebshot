from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from time import perf_counter
from src.embeddings import (
    get_gemini_embedding,
    get_resnet_embedding,
    get_triplet_embedding,
)


def search(
    query_embedding,
    embeddings,
    labels,
    paths,
    top_k=5,
):
    """
    Выполняет поиск наиболее похожих изображений по косинусному сходству.

    Args:
        query_embedding (np.ndarray): Эмбеддинг изображения-запроса.
        embeddings (np.ndarray): База эмбеддингов.
        labels (np.ndarray): Метки классов.
        paths (np.ndarray): Пути к изображениям.
        top_k (int): Количество возвращаемых результатов.

    Returns:
        list[tuple[str, float, str]]:
        (label, similarity_score, image_path)
    """

    scores = cosine_similarity(
        query_embedding.reshape(1, -1),
        embeddings,
    )[0]

    indices = np.argsort(scores)[::-1][:top_k]

    return [
        (labels[i], scores[i], paths[i])
        for i in indices
    ]


def search_all_models(
    image_path,
    client,
    resnet50,
    triplet_resnet,
    transform,
    device,
    gemini_original_db,
    gemini_512_db,
    resnet50_db,
    resnet50_triplet_db,
):
    """
    Выполняет поиск похожих изображений всеми моделями.

    Args:
        image_path (Path | str): Путь к изображению.
        client: Gemini Client.
        resnet50: Загруженная модель ResNet50.
        triplet_resnet: Загруженная Triplet ResNet50.
        transform: Преобразования изображения.
        device: Устройство вычислений.
        gemini_original_db: База эмбеддингов Gemini Original.
        gemini_512_db: База эмбеддингов Gemini 512.
        resnet50_db: База эмбеддингов ResNet50.
        resnet50_triplet_db: База эмбеддингов Triplet ResNet50.

    Returns:
        tuple:
            gemini_original_results,
            gemini_original_time,
            gemini_512_results,
            gemini_512_time,
            resnet50_results,
            resnet50_time,
            triplet_results,
            triplet_time
    """


    # Gemini Original
    start = perf_counter()

    gemini_original_embedding = get_gemini_embedding(
        image_path=image_path,
        client=client,
        mode="original",
    )

    gemini_original_results = search(
        gemini_original_embedding,
        gemini_original_db[0],
        gemini_original_db[1],
        gemini_original_db[2],
    )

    gemini_original_time = perf_counter() - start

    # Gemini 512
    start = perf_counter()

    gemini_512_embedding = get_gemini_embedding(
        image_path=image_path,
        client=client,
        mode="512",
    )

    gemini_512_results = search(
        gemini_512_embedding,
        gemini_512_db[0],
        gemini_512_db[1],
        gemini_512_db[2],
    )

    gemini_512_time = perf_counter() - start

    # ResNet50
    start = perf_counter()

    resnet50_embedding = get_resnet_embedding(
        image_path=image_path,
        model=resnet50,
        transform=transform,
        device=device,
    )

    resnet50_results = search(
        resnet50_embedding,
        resnet50_db[0],
        resnet50_db[1],
        resnet50_db[2],
    )


    resnet50_time = perf_counter() - start

    # Triplet ResNet50
    start = perf_counter()

    triplet_embedding = get_triplet_embedding(
        image_path=image_path,
        model=triplet_resnet,
        transform=transform,
        device=device,
    )

    triplet_results = search(
        triplet_embedding,
        resnet50_triplet_db[0],
        resnet50_triplet_db[1],
        resnet50_triplet_db[2],
    )

    triplet_time = perf_counter() - start

    return (
        gemini_original_results,
        gemini_original_time,
        gemini_512_results,
        gemini_512_time,
        resnet50_results,
        resnet50_time,
        triplet_results,
        triplet_time,
    )