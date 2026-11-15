# Análise de desempenho por vídeo ⚽

Fork do [`roboflow/sports`](https://github.com/roboflow/sports) usado no projeto de
análise de desempenho. A partir de um vídeo de jogo, ele detecta e rastreia os
jogadores, separa os dois times, projeta todo mundo no campo visto de cima (radar
tático) e gera dados para o app **Soccer Scouting**.

## Como usar (Google Colab)

1. Abra [`notebooks/soccer_scouting.ipynb`](notebooks/soccer_scouting.ipynb) no Colab
   (*Arquivo → Abrir notebook → GitHub*, e cole `rodrigodezordi/sports`).
2. **Ambiente de execução → Alterar o tipo de ambiente de execução → T4 GPU**.
3. Em 🔑 *Secrets*, crie `ROBOFLOW_API_KEY` com a sua *Private API Key* da Roboflow.
4. Rode a célula 1 e **espere terminar** (3 a 5 min). Se todas as bibliotecas aparecerem
   como "ok", vá em *Reiniciar sessão* e rode da célula 2 em diante.

Toda vez que o Colab abre uma máquina nova (depois de um tempo parado, ou ao trocar
para GPU), a instalação e o vídeo enviado se perdem: rode a partir da célula 1 de novo.

## O que o notebook gera

| arquivo | o que é |
|---|---|
| `jogo_marcado.mp4` | o vídeo com cada jogador marcado e o radar no canto |
| `football-ai-<jogo>.json` | posições a cada 0,2 s, posse, distância e finalizações, para importar na aba **Tático** do Soccer Scouting |
| `finalizacoes.csv`, `mapa_finalizacoes.png` | cada chute: quem, de onde, distância, ângulo, velocidade da bola e direção |
| `chute_XX_tNNs.mp4` | 3 s de vídeo de cada chute, para conferir |

## O que este fork acrescenta à biblioteca

- `sports/common/calibration.py`: `PitchCalibrator`. Calibra o campo a partir dos
  keypoints, recusa quadros em que os pontos estão alinhados (por exemplo, só a linha
  do meio aparece, o que gera uma projeção errada) e reaproveita a última calibração
  boa por um tempo curto.
- `sports/common/shots.py`: `detect_shots`. Conta finalizações ao gol pela trajetória
  da bola no campo: posse antes do chute, velocidade mínima, distância até o gol e
  direção para a boca do gol. Os limites ficam em `ShotDetectorConfig`.
- `tests/`: testes com trajetórias simuladas (`python -m pytest tests`).

## Limitações

- **Gol ou não:** com uma câmera só, a bola no ar é projetada no lugar errado, então
  "na direção do gol" é uma estimativa. Confira pelos recortes de vídeo.
- **Bola:** o modelo de jogadores perde a bola em vários quadros, e cada chute em que
  isso acontece pode ficar de fora. O próximo passo é o detector dedicado de bola
  (`notebooks/train_ball_detector.ipynb`).
- **IDs ≠ nomes:** o rastreador dá números, e eles podem trocar quando jogadores se
  cruzam.
- **Medidas do campo:** os modelos usam um campo de 120 × 70 m. Ajuste
  `SoccerPitchConfiguration` às medidas reais para que distâncias e velocidades
  fiquem certas.
