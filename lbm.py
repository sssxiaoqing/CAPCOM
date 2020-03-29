#格子ボルツマン(lbm:lattice boltzmann method)法を用いた二次元流体シミュレーション
#研究で使われているモデル（コード約十万行）はc++で書かれているため、ここでは核心の内容を必要最低限でpythonに書き換えました
#格子ボルツマン法の概要：流体を並進・衝突する仮想の粒子の集合と仮定する
#                    並進過程：粒子は1タイムステップ後に隣の格子点に移動する、あるいは元の場所に位置する（二次元だと9(3*3)方向）
#                    衝突過程：粒子の分布は局所平衡状態へと緩和する

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


class lattice_boltzmann_method:
    
    # パラメータ
    ny    = 40                             # y方向grid数
    nx    = 200                            # x方向grid数
    k_vis = 0.005                          # 動粘性係数（無次元化後）    （乱流モデルは実装されていないため、小さくしすぎると、計算は破綻する）
    omega = 1.0 / (3*k_vis + 0.5)          # 時間緩和係数の逆数
    u0    = 0.1                            # 初期風速（x成分、無次元化後）（乱流モデルは実装されていないため、大きくしすぎると、計算は破綻する）
    v0    = 0.0                            # 初期風速（y成分、無次元化後）
    rho0  = 1.0                            # 初期密度
    solid = np.zeros((ny,nx), bool)        # solidを適切に設置
    solid[15:20, 20:30] = True
    solid[25:30, 35:45] = True
    solid[10:15, 50:60] = True
    solid[20:25, 85:95] = True


    # コンストラクタ
    def __init__(self):
        self.fig = plt.figure(figsize=(6,3))
        self.init()
        self.anim = animation.FuncAnimation(self.fig, self.time_evolution, interval=1)  # アニメーション


    # 9方向の各重み係数
    def factor(self, j, i):
        if abs(j) + abs(i) == 0:
            return 4.0 / 9.0
        elif abs(j) + abs(i) == 1:
            return 1.0 / 9.0
        else:
            return 1.0 / 36.0


    # 平衡分布関数の計算
    def feq(self, u, v, rho, j, i):
        ivel = u*i + v*j
        vel  = u*u + v*v
        return self.factor(j, i) * rho * (1.0 + 3*ivel + 4.5*ivel*ivel - 1.5*vel)


    # 初期化
    def init(self):
        global f_lbm
        f_lbm  = np.zeros((9,self.ny,self.nx))

        for j in range(-1, 2):
            for i in range(-1, 2):
                dir = 3*(j+1) + (i+1)
                f_lbm[dir,:,:] = self.feq(self.u0, self.v0, self.rho0, j, i)

    # 並進過程
    def streaming(self):
        global fn_lbm
        fn_lbm = np.zeros((9,self.ny,self.nx))

        for j in range(-1, 2):
            for i in range(-1, 2):
                dir    = 3*(j+1) + (i+1)
                dir_re = 3*(-j+1) + (-i+1)

                fn_lbm[dir,:,:] = np.roll(np.roll(f_lbm[dir,:,:], i, axis=1), j, axis=0)  # 並進
                solid_dir = np.roll(np.roll(self.solid, i, axis=1), j, axis=0)
                fn_lbm[dir,solid_dir] = f_lbm[dir_re,solid_dir]                           # solidに対する跳ね返り（bounce back）


    # 衝突過程
    def collision(self):
        global fn_lbm, rho, u, v
        rho = np.zeros((self.ny,self.nx))
        u   = np.zeros((self.ny,self.nx))
        v   = np.zeros((self.ny,self.nx))

        for j in range(-1, 2):
            for i in range(-1, 2):
                dir = 3*(j+1) + (i+1)
                rho += fn_lbm[dir,:,:]           # 密度の計算

        for j in range(-1, 2):
            for i in range(-1, 2):
                dir = 3*(j+1) + (i+1)
                u += fn_lbm[dir,:,:] * i / rho   # uの計算
                v += fn_lbm[dir,:,:] * j / rho   # vの計算

        for j in range(-1, 2):
            for i in range(-1, 2):
                dir = 3*(j+1) + (i+1)
                fn_lbm[dir,:,:] = fn_lbm[dir,:,:] - self.omega * (fn_lbm[dir,:,:] - self.feq(u, v, rho, j, i))   # 分布関数の計算


    # 境界条件
    def boundary_condition_x(self):
        global fn_lbm, rho, u, v
        for j in range(-1, 2):
            for i in range(-1, 2):
                dir = 3*(j+1) + (i+1)
                fn_lbm[dir,:,0]         = self.feq(self.u0, self.v0, self.rho0, j, i)                         # 流入端（x=0）
                fn_lbm[dir,:,self.nx-1] = self.feq(u[:,self.nx-2], v[:,self.nx-2], rho[:,self.nx-2], j, i)    # 流出端（x=ny-1)


    # 分布関数の入れ替え
    def swap(self):
        global f_lbm
        f_lbm = fn_lbm    


    # 時間の発展
    def time_evolution(self, i):
        for step in range(50):                # 50タイムステップ後一つフレーム
            self.streaming()
            self.collision()
            self.boundary_condition_x()       # コメントアウトするとx方向は周期境界になる
            self.swap()
        
        # 可視化 
        plt.clf()
        plt.title("u")
        plt.imshow(u, cmap='bwr', interpolation='none', origin='lower', vmax=0.2, vmin=-0.2)  # uの可視化（赤：x+方向、青：x-方向）
        plt.colorbar(orientation='horizontal',shrink=0.4)
        solidarrary = np.zeros((self.ny, self.nx, 4), np.uint8)
        solidarrary[self.solid,3] = 255
        plt.imshow(solidarrary, interpolation='none', origin='lower')                         # solidの可視化


    def animation(self):
        plt.show()
        return self


if __name__ == "__main__":
    lattice_boltzmann_method().animation()
