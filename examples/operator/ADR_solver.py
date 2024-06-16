"""
这段代码是一个Python脚本，实现了一个名为 `ADR_solver` 的一维(1D)偏微分方程(PDE)求解器。它专门用于解决以下形式的方程：

\[ u_t = (k(x) u_x)_x - v(x) u_x + g(u) + f(x, t) \]

这里，`u` 是依赖于空间变量 `x` 和时间变量 `t` 的函数，`k(x)` 是空间变量的系数函数，`v(x)` 是空间扩散项的系数，
`g(u)` 是非线性项，`f(x, t)` 是源项，而 `u_t` 和 `u_x` 分别表示 `u` 对时间 `t` 和空间 `x` 的导数。

代码的主要部分是 `solve_ADR` 函数，它采用以下参数：
- `xmin`, `xmax`：空间域的边界。
- `tmin`, `tmax`：时间域的边界。
- `k`, `v`, `g`, `dg`, `f`：分别是上面方程中的系数函数、非线性函数及其导数、源函数。
- `u0`：初始条件函数。
- `Nx`, `Nt`：空间和时间网格点的数量。

该函数使用有限差分法来离散化PDE，并使用显式时间步进方案来求解。它构建了用于线性方程组的矩阵 `M`，然后通过迭代的方式计算每个时间步的解 `u`。

`main` 函数设置了方程的具体参数，包括系数函数、源项、初始条件和解的真值（如果已知）。然后调用 `solve_ADR` 函数来求解PDE，并打印出数值解
与真值之间的最大绝对误差。最后，使用 `matplotlib` 绘制了空间域上 `u` 的分布。

代码的最后部分是一个标准的Python入口点，当脚本被执行时将调用 `main` 函数。

简而言之，`ADR_solver` 是一个用于求解特定类型的一维偏微分方程的数值求解器，它使用有限差分法和显式时间积分方法。
"""
import matplotlib.pyplot as plt
import numpy as np


def solve_ADR(xmin, xmax, tmin, tmax, k, v, g, dg, f, u0, Nx, Nt):
    """Solve 1D
    u_t = (k(x) u_x)_x - v(x) u_x + g(u) + f(x, t)
    with zero boundary condition.
    """
    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h**2

    D1 = np.eye(Nx, k=1) - np.eye(Nx, k=-1)
    D2 = -2 * np.eye(Nx) + np.eye(Nx, k=-1) + np.eye(Nx, k=1)
    D3 = np.eye(Nx - 2)
    k = k(x)
    M = -np.diag(D1 @ k) @ D1 - 4 * np.diag(k) @ D2
    m_bond = 8 * h2 / dt * D3 + M[1:-1, 1:-1]
    v = v(x)
    v_bond = 2 * h * np.diag(v[1:-1]) @ D1[1:-1, 1:-1] + 2 * h * np.diag(
        v[2:] - v[: Nx - 2]
    )
    mv_bond = m_bond + v_bond
    c = 8 * h2 / dt * D3 - M[1:-1, 1:-1] - v_bond
    f = f(x[:, None], t)

    u = np.zeros((Nx, Nt))
    u[:, 0] = u0(x)
    for i in range(Nt - 1):
        gi = g(u[1:-1, i])
        dgi = dg(u[1:-1, i])
        h2dgi = np.diag(4 * h2 * dgi)
        A = mv_bond - h2dgi
        b1 = 8 * h2 * (0.5 * f[1:-1, i] + 0.5 * f[1:-1, i + 1] + gi)
        b2 = (c - h2dgi) @ u[1:-1, i].T
        u[1:-1, i + 1] = np.linalg.solve(A, b1 + b2)
    return x, t, u


def main():
    xmin, xmax = -1, 1
    tmin, tmax = 0, 1
    k = lambda x: x**2 - x**2 + 1
    v = lambda x: np.ones_like(x)
    g = lambda u: u**3
    dg = lambda u: 3 * u**2
    f = (
        lambda x, t: np.exp(-t) * (1 + x**2 - 2 * x)
        - (np.exp(-t) * (1 - x**2)) ** 3
    )
    u0 = lambda x: (x + 1) * (1 - x)
    u_true = lambda x, t: np.exp(-t) * (1 - x**2)

    # xmin, xmax = 0, 1
    # tmin, tmax = 0, 1
    # k = lambda x: np.ones_like(x)
    # v = lambda x: np.zeros_like(x)
    # g = lambda u: u ** 2
    # dg = lambda u: 2 * u
    # f = lambda x, t: x * (1 - x) + 2 * t - t ** 2 * (x - x ** 2) ** 2
    # u0 = lambda x: np.zeros_like(x)
    # u_true = lambda x, t: t * x * (1 - x)

    Nx, Nt = 100, 100
    x, t, u = solve_ADR(xmin, xmax, tmin, tmax, k, v, g, dg, f, u0, Nx, Nt)

    print(np.max(abs(u - u_true(x[:, None], t))))
    plt.plot(x, u)
    plt.show()


if __name__ == "__main__":
    main()
