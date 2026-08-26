import numpy as np
import math


def dense(k, m, n, b, core_id, round_id):
    # 记号转换
    k = min(k,m*b)
    j = core_id
    r = round_id
    if j>k:
        return None
    
    p = (math.ceil(r/m)-1)*k+j
    
    # 计算w,y
    w = p%b or b
    y = math.ceil(p/b)
    # 计算x
    y1 = y%m or m
    r1 = r%m or m    
    x = y1 + r1 - 1
    if x>m:
        x -= m
    
    if 1<=w<=b and 1<=x<=m and 1<=y<=n:
        return w, (x, y)
    else:
        # 超出范围的分配，返回 None
        return None

def case0(k, m, n, core_id, round_id):
    epsilon = 0
    j = core_id
    r = round_id

    # 核心编号超过 ⌊n/2⌋ + 1 时无任务可处理
    if j > (n // 2) + 1:
        return None

    if j % 2 == 1:  # 奇核
        if r + j <= n + 1:
            x = r + j - 1
            y = j
        else:
            x = 2 * n + 2 - j - r
            y = n + 3 - j - (n % 2)
    else:  # 偶核
        if j <= r + 1 - (n % 2):
            x = n + j - r-1+n%2
            y = j
        else:
            x = n + 2 + r - j - (n % 2)
            y = n + 3 - j - (n % 2)

    # 检查 (x, y) 是否在下三角范围内
    if 1 <= y <= m and y <= x <= m:
        return (x, y)
    else:
        return None

def case0_rec(k, m, n, core_id, round_id):

    # Choose method based on core count
    if 2 * k < m + 1 and k<n:
        (x, y) = g2k(k,core_id, round_id, m, 0)
        
    else:
        result = case0(k, m, m, core_id, round_id)
        if result is not None:
            x, y = result
        else:
            return None
    # Check if the computed position is within the valid submatrix
    if 1 <= y <= n and y <= x <= m:
        return (x, y)
    else:
        return None


def g2k(k, j, a, L1, offset):
    # 奇核分支
    if j % 2 == 1:
        if a <= L1 - j + 1:
            y = j + offset
            x = y + a - 1
        else:
            y = 2 * k + 1 - j + offset
            x = y + 2 * L1 - 2 * k + 1 - a
    # 偶核分支
    else:
        if a >= L1 - 2 * k + 1 + j:
            y = j + offset
            x = y + 2 * L1 - 2 * k + 1 - a
        else:
            y = 2 * k + 1 - j + offset
            x = y + a - 1

    return (x, y)


def g3k(j, a, L1, L2, k, offset):
    if k % 2 == 1:
        h = 3 * (k + 1) // 2
        I1_min, I1_max = 1, h + L2 - 1
        I2_min = h + L2
        I2_max = h + L2 + (L1 - k) - 1
        I3_min = h + L2 + L1 - k
        I3_max = h + L1 + 2 * L2 - 2

        bj = (j + (k + 1) // 2) % k
        if bj == 0:
            bj = k
        cj = h - j - bj

        if I1_min <= a <= I1_max:
            if a <= k - j + 1:
                y = j + offset
                x = y + a - 1
            elif a >= 2 * k - j - bj + 3:
                y = 2 * k + cj + offset
                x = y + (I1_max - a)
            else:
                y = k + bj + offset
                x = y + (a - (k - j + 2))
        elif I2_min <= a <= I2_max:
            mod_val = (a - h - L2 + j) % (L1 - k)
            tmp = mod_val + (k - j + 1)
            if mod_val == 0:
                tmp = (L1 - k) + (k - j + 1)
            y = j + offset
            x = y + tmp - 1
        else:  # a in I3
            mod_val = (a - h - L2 - L1 + k + bj) % (L2 + k - 1)
            tmp = mod_val + (k - bj + 1)
            if mod_val == 0:
                tmp = (L2 + k - 1) + (k - bj + 1)
            y = k + bj + offset
            x = y + tmp - 1

    # Even k branch
    else:
        h = 3 * k // 2 + 2
        I1_min, I1_max = 1, h + L2 - 1
        I2_min = h + L2
        I2_max = h + L2 + (L1 - k) - 1
        I3_min = h + L2 + L1 - k
        I3_max = h + L1 + 2 * L2 - 2

        bj = (j + k // 2) % k
        if bj == 0:
            bj = k
        if j <= k // 2:
            cj = h - j - bj
        else:
            cj = (h - 1) - j - bj

        if I1_min <= a <= I1_max:
            if a <= k - j + 1:
                y = j + offset
                x = y + a - 1
            elif j > k // 2 and (k - j + 2) <= a <= (2 * k - j - bj + 2):
                y = k + bj + offset
                x = y + (a - (k - j + 2))
            elif j <= k // 2 and a == (k - j + 2):
                return None
            elif j <= k // 2 and (k - j + 3) <= a <= (2 * k - j - bj + 3):
                y = k + bj + offset
                x = y + (a - (k - j + 3))
            else:
                y = 2 * k + cj + offset
                x = y + (I1_max - a)
        elif I2_min <= a <= I2_max:
            mod_val = (a - h - L2 + j) % (L1 - k)
            tmp = mod_val + (k - j + 1)
            if mod_val == 0:
                tmp = (L1 - k) + (k - j + 1)
            y = j + offset
            x = y + tmp - 1
        else:  # a in I3
            mod_val = (a - h - L2 - L1 + k + bj) % (L2 + k - 1)
            tmp = mod_val + (k - bj + 1)
            if mod_val == 0:
                tmp = (L2 + k - 1) + (k - bj + 1)
            y = k + bj + offset
            x = y + tmp - 1

    return (x, y)
