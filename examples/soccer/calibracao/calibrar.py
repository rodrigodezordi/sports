"""
Calibra uma câmera fixa a partir de pontos das linhas do campo marcados à mão
numa imagem de referência, e grava o arquivo que o notebook usa.

Uso:
    python calibrar.py pontos_campo_treino.json referencia.png campo_treino.json

O arquivo de pontos tem, em pixels da imagem de referência, alguns pontos
sobre cada linha visível (não precisam ser cantos):

    linha_lateral_perto   a lateral do lado da câmera
    linha_do_meio         a linha do meio-campo
    circulo_central       pontos sobre o círculo central
    area_frente_direita   a linha da frente da grande área da direita (opcional)
    meia_lua_direita      pontos sobre a meia-lua da direita (opcional)
    ignorar_px            retângulos [x0, y0, x1, y1] com textos fixos na tela

O ajuste usa um modelo de câmera (foco, rotação, posição) e estima também a
largura do campo (pelo círculo central) e o comprimento (pela grande área).
Se você souber as medidas reais, passe --comprimento e --largura.
"""
import argparse
import json

import cv2
import numpy as np
from scipy.optimize import least_squares

from sports.common.camera import save_calibration

CIRCLE_R = 9.15


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pontos")
    ap.add_argument("imagem")
    ap.add_argument("saida")
    ap.add_argument("--comprimento", type=float)
    ap.add_argument("--largura", type=float)
    a = ap.parse_args()

    pts = json.load(open(a.pontos, encoding="utf-8"))
    img = cv2.imread(a.imagem)
    h_img, w_img = img.shape[:2]
    get = lambda k: np.array(pts.get(k, []), float).reshape(-1, 2)
    near, half, circ = get("linha_lateral_perto"), get("linha_do_meio"), get("circulo_central")
    front, darc = get("area_frente_direita"), get("meia_lua_direita")

    # mundo: origem na lateral de perto, sobre a linha do meio; X para a ESQUERDA
    # da imagem, Y para dentro do campo (lateral de longe em Y = largura)
    def camera(p):
        f, rx, ry, rz, tx, ty, tz = p[:7]
        k = np.array([[f, 0, w_img / 2], [0, f, h_img / 2], [0, 0, 1]])
        r, _ = cv2.Rodrigues(np.array([rx, ry, rz]))
        return k, r, np.array([tx, ty, tz])

    def proj(p, xy):
        k, r, t = camera(p)
        cam = (r @ np.c_[xy, np.zeros(len(xy))].T).T + t
        uv = (k @ cam.T).T
        return uv[:, :2] / uv[:, 2:3]

    def gap(meas, curve):
        if not len(meas):
            return np.zeros(0)
        return np.linalg.norm(meas[:, None] - curve[None], axis=2).min(1)

    def dims(q):
        width = a.largura if a.largura else q[7]
        length = a.comprimento if a.comprimento else q[8]
        return length, width

    def residuals(q):
        length, width = dims(q)
        s = np.linspace(-60, 60, 400)
        ang = np.linspace(0, 2 * np.pi, 720)
        out = [gap(near, proj(q, np.c_[s, np.zeros_like(s)])),
               gap(half, proj(q, np.c_[np.zeros(200), np.linspace(0, width, 200)])),
               gap(circ, proj(q, np.c_[CIRCLE_R * np.cos(ang), width / 2 + CIRCLE_R * np.sin(ang)]))]
        xf = -(length / 2 - 16.5)
        out.append(gap(front, proj(q, np.c_[np.full(200, xf), np.linspace(width / 2 - 20.16, width / 2 + 20.16, 200)])))
        arc = np.c_[-(length / 2 - 11) + CIRCLE_R * np.cos(ang), width / 2 + CIRCLE_R * np.sin(ang)]
        out.append(gap(darc, proj(q, arc[arc[:, 0] > xf])))
        return np.concatenate(out)

    def look_at(cam_x, f0, w0):
        c = np.array([cam_x, -20.0, -8.0])
        z = np.array([0.0, 35.0, 0.0]) - c
        z /= np.linalg.norm(z)
        x = np.cross(z, [0, 0, -1.0])
        x /= np.linalg.norm(x)
        r = np.vstack([x, np.cross(z, x), z])
        rv, _ = cv2.Rodrigues(r)
        return np.r_[f0, rv.ravel(), -r @ c, w0]

    def solve(fun, starts):
        best = None
        for q0 in starts:
            try:
                sol = least_squares(fun, q0, method="lm", max_nfev=3000)
            except ValueError:
                continue
            if best is None or sol.cost < best.cost:
                best = sol
        return best

    # 1º: só a lateral de perto, a linha do meio e o círculo (posição da câmera e largura)
    n_basic = len(near) + len(half) + len(circ)
    basic = lambda q8: residuals(np.r_[q8, 105.0])[:n_basic]
    stage1 = solve(basic, [look_at(cx, f0, 68.0) for cx in (0, 10, 20, 30) for f0 in (600, 900, 1300)])
    # 2º: tudo junto, incluindo a grande área (comprimento)
    best = solve(residuals, [np.r_[stage1.x, l0] for l0 in (95.0, 100.0, 105.0, 110.0)])
    q = best.x
    length, width = dims(q)
    err = residuals(q)
    print(f"erro médio {np.sqrt(np.mean(err ** 2)):.1f} px (máx {err.max():.1f}) · campo {length:.1f} x {width:.1f} m")

    k, r, t = camera(q)
    world_to_img = k @ np.c_[r[:, 0], r[:, 1], t]
    img_to_world = np.linalg.inv(world_to_img)
    # para as coordenadas do SoccerPitchConfiguration (cm): x de 0 (gol da esquerda
    # da imagem) até o comprimento, y de 0 (lateral de longe, em cima no radar) até a largura
    to_cfg = np.array([[-100, 0, length * 50], [0, -100, width * 100], [0, 0, 1.0]])
    h = to_cfg @ img_to_world
    save_calibration(a.saida, img, h / h[2, 2], length, width,
                     ignore=pts.get("ignorar_px", []), description=pts.get("descricao", ""))
    print("gravado em", a.saida)


if __name__ == "__main__":
    main()
