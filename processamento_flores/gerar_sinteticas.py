
import cv2
import numpy as np
from pathlib import Path

# Pastas do projeto
pasta = Path(__file__).parent
pasta_sinteticas = pasta / "imagens_sinteticas"

pasta_sinteticas.mkdir(exist_ok=True)

# Criar três imagens sintéticas
configuracoes = [
    {
        "nome": "sintetica_1",
        "formas": [
            (150, 150, 35, 120),
            (300, 150, 45, 140)
        ]
    },
    {
        "nome": "sintetica_2",
        "formas": [
            (150, 150, 65, 120),
            (350, 150, 55, 150),
            (250, 350, 45, 160)
        ]
    },
    {
        "nome": "sintetica_3",
        "formas": [
            (100, 100, 30, 115),
            (250, 100, 45, 130),
            (400, 100, 35, 145),
            (150, 300, 55, 155),
            (350, 300, 65, 165)
        ]
    }
]

for config in configuracoes:

    # Criar fundo verde
    imagem_hsv = np.zeros((500, 500, 3), dtype=np.uint8)
    imagem_hsv[:, :] = [60, 180, 180]

    # Criar máscara real das formas
    mascara_real = np.zeros((500, 500), dtype=np.uint8)

    for x, y, raio, tonalidade in config["formas"]:

        # Cor em HSV
        cor = np.array([tonalidade, 220, 220], dtype=np.uint8)

        # Desenhar círculo colorido
        cv2.circle(
            imagem_hsv,
            (x, y),
            raio,
            tuple(int(v) for v in cor),
            -1
        )

        # Registrar pixels reais
        cv2.circle(
            mascara_real,
            (x, y),
            raio,
            255,
            -1
        )

    # Converter para BGR para salvar
    imagem_bgr = cv2.cvtColor(
        imagem_hsv,
        cv2.COLOR_HSV2BGR
    )

    # Salvar imagem e máscara de referência
    nome = config["nome"]

    cv2.imwrite(
        str(pasta_sinteticas / f"{nome}.png"),
        imagem_bgr
    )

    cv2.imwrite(
        str(pasta_sinteticas / f"{nome}_referencia.png"),
        mascara_real
    )

    # Quantidade exata de pixels coloridos
    pixels_reais = cv2.countNonZero(mascara_real)

    print(f"\n{nome}")
    print("Quantidade real de pixels:", pixels_reais)

print("\nTrês imagens sintéticas criadas!")
