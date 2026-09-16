from pathlib import Path

FOLDER_PATH = Path("/home/nekaido/PycharmProjects/clothes_cv/data/images")


def detect_image_extension(file_path: Path) -> str | None:
    try:
        with open(file_path, "rb") as f:
            header = f.read(12)
    except Exception:
        return None

    if not header:
        return None

    if header.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    elif header.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    elif header.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    elif header.startswith(b"BM"):
        return ".bmp"
    elif header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return ".webp"

    return None


def rename_images(folder_path: Path):
    if not folder_path.exists():
        print(f"Ошибка: Папка {folder_path} не найдена.")
        return

    valid_files = []

    for file in folder_path.iterdir():
        if not file.is_file():
            continue

        ext = detect_image_extension(file)
        if ext:
            valid_files.append((file, ext))

    if not valid_files:
        print("Изображения для переименования не найдены.")
        return

    valid_files.sort(key=lambda x: x[0].name)

    temp_renames = []
    for index, (file, ext) in enumerate(valid_files, start=1):
        temp_name = f"__temp_image_{index:03d}{ext}"
        temp_path = folder_path / temp_name
        file.rename(temp_path)
        temp_renames.append((temp_path, index, ext))

    for temp_path, index, ext in temp_renames:
        final_name = f"image_{index:03d}{ext}"
        final_path = folder_path / final_name
        temp_path.rename(final_path)
        print(f"Обработан: {temp_path.name} -> {final_name}")

    print("Переименование завершено.")


if __name__ == "__main__":
    rename_images(FOLDER_PATH)