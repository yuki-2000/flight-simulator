#pip install keyboard
#でライブラリをインストールしておくこと


import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import keyboard
import time
import mpl_toolkits.mplot3d.art3d as art3d

# --- 機体パラメータ ---
g = 9.81
# 初期条件
U0 = 293.8
W0 = 0
theta0 = 0.0

# 縦の有次元安定微係数: ロッキードP2V-7
Xu = -0.0215
Zu = -0.227
Mu = 0

Xa = 14.7
Za = -236
Ma = -3.78
Madot = -0.28

Xq = 0
Zq = -5.76
Mq = -0.992

X_deltat = 0.01#0
Z_deltae = -12.9
Z_deltat = 0
M_deltae = -2.48
M_deltat = 0


# 横・方向の有次元安定微係数: ロッキードP2V-7
Yb=-45.4
Lb=-1.71
Nb=0.986

Yp=0.716
Lp=-0.962
Np=-0.0632

Yr=2.66
Lr=0.271
Nr=-0.215

#Y_deltaa=0
L_deltaa=1.72
N_deltaa=-0.0436

Y_deltar=9.17
L_deltar=0.244
N_deltar=-0.666







# --- 縦の運動方程式 ---
# 行列A, Bの定義
A_long = np.array([[Xu, Xa, -W0, -g * np.cos(theta0)],
                   [Zu / U0, Za / U0, 1 + Zq / U0, (g / U0) * np.sin(theta0)],
                   [Mu + Madot * (Zu / U0), Ma + Madot * (Za / U0), Mq + Madot * (1 + Zq / U0),
                    (Madot * g / U0) * np.sin(theta0)],
                   [0, 0, 1, 0]])

B_long = np.array([[0, X_deltat],
                   [Z_deltae / U0, Z_deltat / U0],
                   [M_deltae + Madot * (Z_deltae / U0), M_deltat + Madot * (Z_deltat / U0)],
                   [0, 0]])


# --- 横・方向の運動方程式 ---
# 行列A, Bの定義
A_lat = np.array([[Yb / U0, (W0 + Yp) / U0, (Yr / U0 - 1), g * np.cos(theta0) / U0, 0],
                  [Lb, Lp, Lr, 0, 0],
                  [Nb, Np, Nr, 0, 0],
                  [0, 1, np.tan(theta0), 0, 0],
                  [0, 0, 1 / np.cos(theta0), 0, 0]])

B_lat = np.array([[0, Y_deltar / U0],
                  [L_deltaa, L_deltar],
                  [N_deltaa, N_deltar],
                  [0, 0],
                  [0, 0]])




# --- 回転行列の定義 ---
def rotation_matrix(psi, theta, phi):
  # z軸周り回転
  R_z = np.array([[np.cos(psi), -np.sin(psi), 0],
                   [np.sin(psi), np.cos(psi), 0],
                   [0, 0, 1]])
  # y軸周り回転
  R_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])
  # x軸周り回転
  R_x = np.array([[1, 0, 0],
                   [0, np.cos(phi), -np.sin(phi)],
                   [0, np.sin(phi), np.cos(phi)]])
  # 全体の回転行列
  R = R_z @ R_y @ R_x
  return R





# --- 統合モデル ---

def aircraft_dynamics(t, x, A_long, B_long, A_lat, B_lat, u_long, u_lat):
  # 縦の運動方程式
  dxdt_long = A_long @ np.atleast_2d(x[:4]).T + B_long @ np.atleast_2d(u_long).T
  dxdt_long = dxdt_long.flatten()

  # 横・方向の運動方程式
  dxdt_lat = A_lat @ np.atleast_2d(x[4:9]).T + B_lat @ np.atleast_2d(u_lat).T
  dxdt_lat = dxdt_lat.flatten()

  # 機体座標系での速度ベクトル
  u_b = U0 + x[0]
  v_b = u_b * np.sin(x[4])
  w_b = u_b * np.tan(x[1])
  vel_b = np.array([u_b, v_b, w_b])

  # 全体座標系での速度ベクトル
  psi = x[8]
  theta = x[3]
  phi = x[7]
  vel_e = rotation_matrix(psi, theta, phi) @ np.atleast_2d(vel_b).T
  vel_e = vel_e.flatten()


  # 縦と横・方向の状態量の変化と位置の変化を結合
  dxdt = np.concatenate((dxdt_long, dxdt_lat, vel_e))
  return dxdt













# --- キーボード入力を更新する関数 ---

# キーボード入力の初期化
elevator=0
throttle=0
aileron=0
rudder=0

def update_input():
    global elevator, throttle, aileron, rudder

    if keyboard.is_pressed('4'):
        aileron += 0.01
    elif keyboard.is_pressed('6'):
        aileron -= 0.01
    else:
        aileron *= 0.5  # 徐々に減衰


    if keyboard.is_pressed('9'):
        rudder += 0.01
    elif keyboard.is_pressed('7'):
        rudder -= 0.01
    else:
        rudder *= 0.5  # 徐々に減衰


    if keyboard.is_pressed('8'):
        elevator += 0.01
    elif keyboard.is_pressed('5'):
        elevator -= 0.01
    else:
        elevator *= 0.5  # 徐々に減衰


    if keyboard.is_pressed('+'):
        throttle += 1
    elif keyboard.is_pressed('-'):
        throttle -= 1
    else:
        throttle *= 0.5  # 徐々に減衰









# --- 紙飛行機を描写する ---
def plot_paper_airplane_update(triangles, x, y, z, phi, theta, psi, scale, ax):
    """
    3次元座標と角度が与えられたとき、その状態の紙飛行機のような図形をプロットする

    Args:
      x: x座標
      y: y座標
      z: z座標
      psi: ロール角 (ラジアン)
      theta : ピッチ角 (ラジアン)
      phi: ヨー角 (ラジアン)
      機体の大きさをいじりたければscaleを変える

    """

    #三角形を描く
    poly_left_wing = scale * np.array([[2, 0.0, 0],
                                      [-1, 1, 0],
                                      [-1, 0.1, 0]])
    poly_right_wing = poly_left_wing.copy()
    poly_right_wing[:,1] = -1 * poly_right_wing[:,1]

    poly_left_body = scale * np.array([[2, 0.0, 0.0],
                                      [-1, 0.0, +0.1],
                                      [-1, 0.1, 0.0]])
    poly_right_body = poly_left_body.copy()
    poly_right_body[:,1] = -1 * poly_right_body[:,1]



    R = rotation_matrix(psi, theta, phi) # yaw, pitch, roll


    for triangle, new_points in zip(triangles, [poly_left_wing, poly_left_body, poly_right_wing, poly_right_body]):
        # 紙飛行機の点を回転
        translated_rotated_points = (R @ new_points.T).T + np.array([x, y, z])
        #描写
        triangle.set_verts(translated_rotated_points)



















# --- 可視化(アニメーション with 紙飛行機) ---

# リアルタイムプロットの設定

plt.ion()
fig = plt.figure(dpi=100)
ax = Axes3D(fig)
fig.add_axes(ax)


#飛行の軌跡
line, = ax.plot([], [], [], 'b-')

#紙飛行機の描写
try_left_wing = art3d.Poly3DCollection([np.array([[2, 0.0, 0],[-1, 1, 0],[-1, 0.1, 0]])],facecolors='orangered', linewidths=1, edgecolors='k', alpha=0.6)
try_right_wing = art3d.Poly3DCollection([np.array([[2, 0.0, 0],[-1, 1, 0],[-1, 0.1, 0]])],facecolors='orangered', linewidths=1, edgecolors='k', alpha=0.6)
try_left_body = art3d.Poly3DCollection([np.array([[2, 0.0, 0],[-1, 1, 0],[-1, 0.1, 0]])],facecolors='lime', linewidths=1, edgecolors='k', alpha=0.6)
try_right_body = art3d.Poly3DCollection([np.array([[2, 0.0, 0],[-1, 1, 0],[-1, 0.1, 0]])],facecolors='lime', linewidths=1, edgecolors='k', alpha=0.6)

triangles = [try_left_wing, try_right_wing, try_left_body, try_right_body]
for triangle in triangles:
    ax.add_collection3d(triangle)


ax.set_xlim(0, 5000)
ax.set_ylim(-2500, 2500)
ax.set_zlim(1000, -1000)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")








# --- シミュレーション ---

# 初期状態 (縦と横・方向、位置を結合)
x0_long = np.array([0, 0, 0, 0])  # 例: 全て0で初期化
x0_lat = np.array([0, 0, 0, 0, 0])
x0_pos = np.array([0, 0, 0])
x0 = np.concatenate((x0_long, x0_lat, x0_pos))


# 初期制御入力 (縦と横・方向)
u0_long = np.array([0, 0])  # 例: elevator=0, throttle=0
u0_lat = np.array([0, 0])  # 例: aileron=0, rudder=0


# シミュレーションの設定

t_span = (0, 100)  # 開始時間と終了時間
dt = 0.1 #時間幅

t = 0 #現在時間
x = x0 #現在状態量

all_soly =  np.zeros((12, 1)) #変数をすべて保存する変数


# シミュレーションの実行
while t < t_span[1]:
    start_time = time.time()

    #キーボード入力を取得
    update_input() 
    u_long = [elevator, throttle]
    u_lat = [aileron, rudder]
    
    # t~t+dtまで微分方程式を解く (RK45メソッドを使用)
    sol = solve_ivp(aircraft_dynamics, [t, t+dt], x,
        args=(A_long, B_long, A_lat, B_lat, u_long, u_lat),
        method='RK45')

    #t~t+dtまでの結果を追加
    all_soly = np.append(all_soly, sol.y, axis=1)

    #次のt+dtにおける初期条件
    x = sol.y[:, -1] #現在の最終状態
    t += dt
    
    


    # プロットの更新
    line.set_data_3d(np.append(line.get_data_3d()[0], x[9]), np.append(line.get_data_3d()[1], x[10]), np.append(line.get_data_3d()[2], x[11]))

    #紙飛行機の更新
    plot_paper_airplane_update(triangles=triangles, x=x[9],  y=x[10], z=x[11], phi=x[7], theta=x[3], psi=x[8], scale=400, ax=ax)


    #軸の更新
    ax.set_xlim(min(min(all_soly[9]),0), max(max(all_soly[9]), 5000),)
    ax.set_ylim(min(min(all_soly[10]),-2500), max(max(all_soly[10]), 2500),)
    ax.set_zlim(max(max(all_soly[11]), 1000),min(min(all_soly[11]),-1000))
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    elapsed_time = time.time() - start_time
    print("実行時間：", elapsed_time)
    time.sleep(max(0, dt - elapsed_time)) #実行時間を考慮

plt.ioff()
plt.show()


































