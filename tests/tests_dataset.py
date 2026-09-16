from src.dataset import load_image

def test_load_image():
    image = load_image("data/example.jpg")

    assert image is not None