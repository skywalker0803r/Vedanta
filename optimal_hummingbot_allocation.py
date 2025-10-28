import math

def optimal_hummingbot_allocation(r, N, M, tol=1e-8, max_iter=1000):
    """
    最優 Hummingbot Miner 資金分配計算
    ------------------------------------
    r : list of float
        各池子的每週獎勵 [r1, r2, r3]
    N : list of float
        各池子的總流動性 [N1, N2, N3]
    M : float
        總資金
    tol : float
        解方程的容忍誤差
    max_iter : int
        最大迭代次數

    回傳:
        (m1, m2, m3) 的最優資金分配
    """

    n = len(r)

    def total_funds(lambda_):
        """根據 λ 計算目前分配的總資金量"""
        total = 0
        for i in range(n):
            val = math.sqrt(r[i] * N[i] / lambda_) - N[i]
            if val > 0:
                total += val
        return total

    # 先確定 lambda 搜尋範圍
    lambda_low = 1e-12
    lambda_high = max(r[i] * N[i] for i in range(n))

    for _ in range(max_iter):
        lambda_mid = (lambda_low + lambda_high) / 2
        current = total_funds(lambda_mid)

        if abs(current - M) < tol:
            break

        if current > M:
            # λ 太小 -> 投入太多
            lambda_low = lambda_mid
        else:
            # λ 太大 -> 投入太少
            lambda_high = lambda_mid

    # 用最終 λ 計算最優分配
    lambda_opt = lambda_mid
    m = []
    for i in range(n):
        mi = math.sqrt(r[i] * N[i] / lambda_opt) - N[i]
        m.append(max(mi, 0))

    return m,lambda_opt


if __name__ == "__main__":
    # === 使用範例 ===
    # 輸入池子的參數
    r1, r2, r3 = 500, 500, 262 #每週獎勵
    N1, N2, N3 = 6071, 8248, 6076 #總流動性
    M = 1000 #本金

    r = [r1, r2, r3]
    N = [N1, N2, N3]

    m,lambda_opt = optimal_hummingbot_allocation(r, N, M)
    total = sum(m)

    print("最優資金分配結果：")
    for i, mi in enumerate(m, 1):
        print(f"  池{i}: {mi:.2f}")
    print(f"合計: {total:.2f} (總資金 {M})")
    print(f"最優lambda:{lambda_opt}")