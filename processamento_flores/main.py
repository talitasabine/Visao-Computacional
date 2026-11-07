
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Localizar a pasta do projeto
pasta_projeto = Path(__file__).parent
pasta_resultados = pasta_projeto / "resultados"

# Criar pasta de resultados
pasta_resultados.mkdir(exist_ok=True)

# Imagens que serão processadas
arquivos = ["flor1.jpeg", "flor2.jpeg"]

for nome in arquivos:

    # Carregar imagem
    caminho = pasta_projeto / "imagens" / nome
    imagem = cv2.imread(str(caminho))

    if imagem is None:
        print(f"Erro ao carregar: {nome}")
        continue

    # Converter imagem para HSV
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

    # Faixa inicial de roxo, violeta e azul
    limite_inferior = np.array([110, 40, 40])
    limite_superior = np.array([175, 255, 255])

    # Criar máscara
    mascara = cv2.inRange(
        hsv,
        limite_inferior,
        limite_superior
    )

    # Aplicar máscara na imagem original
    resultado = cv2.bitwise_and(
        imagem,
        imagem,
        mask=mascara
    )

    # Contar pixels identificados
    pixels_flor = cv2.countNonZero(mascara)

    total_pixels = imagem.shape[0] * imagem.shape[1]

    percentual = (pixels_flor / total_pixels) * 100

    print("\nImagem:", nome)
    print("Pixels identificados:", pixels_flor)
    print("Total de pixels:", total_pixels)
    print(f"Percentual de ocupação: {percentual:.2f}%")

    # Salvar imagens processadas
    cv2.imwrite(
        str(pasta_resultados / f"mascara_{nome}.png"),
        mascara
    )

    cv2.imwrite(
        str(pasta_resultados / f"segmentada_{nome}.png"),
        resultado
    )

    # Converter para RGB para exibição
    imagem_rgb = cv2.cvtColor(
        imagem, cv2.COLOR_BGR2RGB
    )

    resultado_rgb = cv2.cvtColor(
        resultado, cv2.COLOR_BGR2RGB
    )

    # Exibir as três etapas
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(imagem_rgb)
    plt.title("Imagem original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mascara, cmap="gray")
    plt.title("Máscara de segmentação")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(resultado_rgb)
    plt.title("Flores identificadas")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
