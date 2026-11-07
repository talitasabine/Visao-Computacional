
import cv2
import numpy as np
from pathlib import Path

pasta_projeto = Path(__file__).parent
pasta_sinteticas = pasta_projeto / "imagens_sinteticas"

# Mesmos limites HSV utilizados no main.py
limite_inferior = np.array([110, 40, 40])
limite_superior = np.array([175, 255, 255])

for numero in range(1, 4):
    nome = f"sintetica_{numero}"

    caminho_imagem = pasta_sinteticas / f"{nome}.png"
    caminho_referencia = pasta_sinteticas / f"{nome}_referencia.png"

    imagem = cv2.imread(str(caminho_imagem))
    referencia = cv2.imread(str(caminho_referencia), cv2.IMREAD_GRAYSCALE)

    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

    mascara_detectada = cv2.inRange(
        hsv, limite_inferior, limite_superior
    )

    pixels_reais = cv2.countNonZero(referencia)
    pixels_detectados = cv2.countNonZero(mascara_detectada)

    erro = abs(pixels_reais - pixels_detectados)
    erro_percentual = (erro / pixels_reais) * 100

    print(f"\n{nome}")
    print(f"Pixels reais: {pixels_reais}")
    print(f"Pixels detectados: {pixels_detectados}")
    print(f"Erro absoluto: {erro}")
    print(f"Erro percentual: {erro_percentual:.2f}%")

    cv2.imwrite(
        str(pasta_sinteticas / f"{nome}_detectada.png"),
        mascara_detectada
    )

print("\nValidação concluída!")
