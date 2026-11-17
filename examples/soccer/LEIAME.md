# Análise de desempenho por vídeo ⚽

Fork do [`roboflow/sports`](https://github.com/roboflow/sports) usado no projeto de
análise de desempenho. A partir de um vídeo de jogo, ele detecta e rastreia os
jogadores, separa os dois times, projeta todo mundo no campo visto de cima (radar
tático) e gera dados para o app **Soccer Scouting**.

## Como usar (Google Colab)

1. Abra [`notebooks/soccer_scouting.ipynb`](notebooks/soccer_scouting.ipynb) no Colab
   (*Arquivo → Abrir notebook → GitHub*, e cole `rodrigodezordi/sports`).
2. **Ambiente de execução → Alterar o tipo de ambiente de execução → T4 GPU**.
3. Não precisa de chave: os modelos YOLOv8 do `roboflow/sports` são baixados uma vez para
   `MyDrive/football-ai/modelos` e rodam com PyTorch na GPU (meia precisão).
4. Rode a célula 1 e **espere terminar** (3 a 5 min). Se todas as bibliotecas aparecerem
   como "ok", vá em *Reiniciar sessão* e rode da célula 2 em diante.

**Deixe o vídeo no Google Drive** (célula 2, `USAR_DRIVE = True`). A análise é feita em
blocos de 1 minuto, e cada bloco pronto fica salvo em `MyDrive/football-ai/resultados/<jogo>/blocos`.
Se a conexão cair, rode as células 1 a 5 de novo: os blocos prontos são pulados e a
análise continua de onde parou, com os mesmos uniformes (`times.pkl`).

### Qualidade, velocidade e direção do campo (célula 3)

O padrão é a qualidade máxima. Para ficar cerca de 4x mais rápido, use os valores da coluna "rápido".

| ajuste | padrão | rápido | efeito |
|---|---|---|---|
| `PULAR_QUADROS` | 1 | 2 | analisa todos os quadros, ou 1 a cada 2 |
| `CALIBRAR_A_CADA` | 1 | 5 | de quantos em quantos quadros recalcula as linhas do campo |
| `CLASSIFICAR_ID_VEZES` | 1000 | 5 | quantas vezes olha o uniforme de cada jogador antes de fixar o time |
| `QUALIDADE_VIDEO` | 20 | 23 | nitidez do vídeo marcado (H.264; menor = mais nítido) |
| `GERAR_VIDEO` | True | False | `False` gera só os dados, sem o vídeo marcado |
| `INVERTER_LADOS` | False | | troca esquerda e direita no radar e nos dados |
| `INVERTER_CIMA_BAIXO` | False | | troca as laterais de cima e de baixo |

Ao mudar qualquer ajuste, troque também o `NOME_JOGO`: os blocos já prontos de um nome são reaproveitados.

## O que o notebook gera

| arquivo | o que é |
|---|---|
| `jogo_marcado.mp4` | o vídeo com cada jogador marcado e o radar no canto |
| `football-ai-<jogo>.json` | posições a cada 0,2 s, posse, distância e finalizações, para importar na aba **Tático** do Soccer Scouting |
| `finalizacoes.csv`, `mapa_finalizacoes.png` | cada chute: quem, de onde, distância, ângulo, velocidade da bola e direção |
| `chute_XX_tNNs.mp4` | 3 s de vídeo de cada chute, para conferir |

### Campo e times (célula 3)

| ajuste | padrão | efeito |
|---|---|---|
| `CALIBRACAO` | `"camera_fixa"` | usa a calibração da câmera do campo de treino (`calibracao/campo_treino.json`) e acompanha quando ela gira ou dá zoom. `"modelo"` volta ao modelo de linhas do tutorial, que só funciona com imagem de TV |
| `COMPRIMENTO_M`, `LARGURA_M` | `105`, `68` | medidas do campo (padrão oficial). Se mudar, o notebook refaz a calibração com as novas medidas. `None` usa a estimativa feita pela imagem |
| `TIMES_POR` | `"cor"` | separa os times pela cor da camisa. `"siglip"` é o método do tutorial, que mistura os times quando os jogadores aparecem pequenos |
| `TEM_ARBITRO` | True | `False` faz quem o modelo marcar como árbitro virar jogador |

**Outra câmera:** marque alguns pontos das linhas (lateral de perto, linha do meio, círculo central e, se der, a grande área) num quadro do vídeo, como em `calibracao/pontos_campo_treino.json`, e rode `python calibracao/calibrar.py pontos.json quadro.png minha_camera.json`. Use `--comprimento` e `--largura` se souber as medidas daquele campo. Depois aponte `ARQ_CALIBRACAO` para o arquivo novo e ajuste `COMPRIMENTO_M` e `LARGURA_M`.

## O que este fork acrescenta à biblioteca

- `sports/common/calibration.py`: `PitchCalibrator`. Calibra o campo a partir dos
  keypoints, recusa quadros em que os pontos estão alinhados (por exemplo, só a linha
  do meio aparece, o que gera uma projeção errada) e reaproveita a última calibração
  boa por um tempo curto.
- `sports/common/shots.py`: `detect_shots`. Conta finalizações ao gol pela trajetória
  da bola no campo: posse antes do chute, velocidade mínima, distância até o gol e
  direção para a boca do gol. Os limites ficam em `ShotDetectorConfig`.
- `sports/common/camera.py`: `CameraTracker`. Para câmera fixa que gira e dá zoom: calibra uma
  vez numa imagem de referência e, a cada quadro, acha a posição da câmera comparando a imagem
  (árvores, prédios, placas) com a referência.
- `sports/common/team_color.py`: `ColorTeamClassifier`. Separa os times pela cor da camisa,
  tirando a grama em volta do jogador, e diz quando a camisa não é de nenhum dos dois (árbitro).
- `tests/`: testes com trajetórias, imagens e camisas simuladas (`python -m pytest tests`).

## Limitações

- **Gol ou não:** com uma câmera só, a bola no ar é projetada no lugar errado, então
  "na direção do gol" é uma estimativa. Confira pelos recortes de vídeo.
- **Bola:** o modelo de jogadores perde a bola em vários quadros, e cada chute em que
  isso acontece pode ficar de fora. O próximo passo é o detector dedicado de bola
  (`notebooks/train_ball_detector.ipynb`).
- **IDs ≠ nomes:** o rastreador dá números, e eles podem trocar quando jogadores se
  cruzam.
- **Medidas do campo:** a câmera do treino está calibrada com o padrão oficial, 105 × 68 m.
  Se o campo real for diferente, distâncias e velocidades saem proporcionalmente erradas:
  coloque as medidas certas em `COMPRIMENTO_M` e `LARGURA_M`.
