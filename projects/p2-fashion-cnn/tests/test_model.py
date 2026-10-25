import numpy as np
import pytest
import torch
from PIL import Image

from fashion_cnn.model import MEAN, STD, FashionCNN, preprocess_image
from fashion_cnn.predict import ARTIFACTS


def test_model_output_shape():
    model = FashionCNN().eval()
    out = model(torch.randn(4, 1, 28, 28))
    assert out.shape == (4, 10)


def test_preprocess_any_size_rgb_to_28x28():
    img = Image.fromarray(np.zeros((300, 200, 3), dtype=np.uint8))
    x = preprocess_image(img)
    assert x.shape == (1, 1, 28, 28)
    assert x.dtype == torch.float32


def test_preprocess_inverts_light_background():
    """Foto produk biasa: latar putih. Setelah preprocessing latar harus 'gelap' seperti FashionMNIST."""
    white = Image.fromarray(np.full((50, 50), 255, dtype=np.uint8))
    x = preprocess_image(white)
    assert torch.allclose(x, torch.full_like(x, (0 - MEAN) / STD), atol=1e-5)


def test_preprocess_keeps_dark_background():
    dark = np.zeros((28, 28), dtype=np.uint8)
    dark[10:18, 10:18] = 255
    x = preprocess_image(Image.fromarray(dark))
    assert x[0, 0, 0, 0] < 0 and x[0, 0, 14, 14] > 0


@pytest.mark.skipif(not (ARTIFACTS / "model.pt").exists(), reason="latih model dulu")
def test_trained_model_predicts_sample_correctly():
    from fashion_cnn.predict import export_samples, predict_image

    correct = 0
    for p in export_samples(20):
        expected = p.stem.split("_", 1)[1].replace("-", "/")
        correct += predict_image(Image.open(p))[0]["label"] == expected
    assert correct >= 16          # >= 80% dari 20 contoh pertama
