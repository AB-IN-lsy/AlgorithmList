# DP

## [Beautiful Array](https://codeforces.com/contest/1155/problem/D)

**难度：** 中等

**标签：** 最大子段和模型

**题意与思路：**

题意：给出一个长度为n的数列和数字x，经过最多一次操作将数列中的一个子段都乘x，使该数列的子段和最大。

思路：分三段，设$dp[i][j]$为以$i$结尾的第$j+1$段的最大子段和



| 状态名     | 含义                                             | 举例说明                  |
| ---------- | ------------------------------------------------ | ------------------------- |
| `dp[i][0]` | 没有乘过 `x` 的情况下，以 `i` 结尾的最大子段和   | 像 Kadane 算法            |
| `dp[i][1]` | 当前正在乘 `x` 的子段，以 `i` 结尾的最大子段和   | 乘法区间的中间部分        |
| `dp[i][2]` | 已经结束了乘 `x` 的操作，以 `i` 结尾的最大子段和 | 从乘完 `x` 后继续往右延伸 |

```python
dp[i][0]=max(dp[i-1][0]+a[i],0LL);
dp[i][1]=max(dp[i][0],dp[i-1][1]+a[i]*x);
dp[i][2]=max(dp[i][1],dp[i-1][2]+a[i]);
ans=max(ans,dp[i][2]);
```



## [整数划分](https://www.acwing.com/problem/content/902/)

**难度：** 简单

**标签：** 计数类DP、完全背包

**题意与思路：**

题意：给定一个正整数 n，请你求出 n 共有多少种不同的划分（表示成若干个正整数之和）方法。

1. **完全背包**：背包容量为$n$, 第i个物品的体积为i(i = 1 \~ n)，每个物品有无限个，问恰好装满背包的方案数。

   1. 表示只从1~i物品中选，体积**恰好为j**的方案数，定义 $dp[i][j]$
   2. 总转移： $dp[i][j]=dp[i−1][j]+dp[i][j−i]$。优化后：$dp[j]+=dp[j−i]$

2. 计数DP：$f[i][j]$表示：将整数i拆成恰好j个正整数的方案数

   `f[0][0] = 1`：把 0 拆成 0 个数，只有 1 种方法（空集）；

   1. 最小值是 1 的方案

   > 如果一个方案中包含了至少一个 `1`，你可以**把这个 `1` 拿出来**，剩下的部分和为 `i-1`，且用了 `j-1` 个数。

   所以这些方案可以由：
   $$
   f[i-1][j-1]
   $$
   转移而来。 

   

   2. 所有数都 ≥ 2 的方案

   > 如果方案中所有数都大于等于 2，那么我们可以把所有数都减去 1，得到一个新问题：
   >
   > 和为 `i-j`，用 `j` 个数（因为每个数都减 1）。

   所以这些方案可以由：
   $$
   f[i-j][j]
   $$
   转移而来。

   综合转移式：
   $$
   f[i][j] = f[i-1][j-1] + f[i-j][j]
   $$
   最终答案

   你要求的是把 `n` 拆成任意个数的和的方案数，即：
   $$
   ans = f[n][1] + f[n][2] + ... + f[n][n]
   $$



```python
N = 1010
MOD = int(1e9 + 7)
dp = [0] * N

n = int(input())

dp[0] = 1  #代表一个数都不选时，体积是0，方案数是1
for i in range(1, n + 1):
    for j in range(i, n + 1):
        dp[j] = (dp[j] + dp[j - i]) % MOD

print(dp[n])
```



```python
N = 1100
MOD = int(1e9 + 7)
dp = [[0] * N for _ in range(N)] # 表示总和为i, 并且恰好为j个数的方案

n = int(input())
dp[0][0] = 1 #总和为0，恰好0个数的方案有一个

for i in range(1, n + 1):
    for j in range(1, n + 1):
        dp[i][j] = (dp[i - 1][j - 1] + dp[i - j][j]) % MOD

res = 0
for i in range(1, n + 1):
    res = (res + dp[n][i]) % MOD
print(res)
```



## [**买卖股票的最佳时机 V**](https://leetcode.cn/problems/maximum-profit-of-operating-k-transactions/)

**难度：** 中等
**标签：** `动态规划`、`记忆化搜索`、`股票系列`、`多空双向交易`

------

### **题意与思路：**

> - 一共 **k 次** 完整交易（开仓→平仓算一次）。
> - 允许 **做多**（先买后卖）和 **做空**（先卖后买）。
> - 同一天不能重复买入/卖出；必须先平掉当前仓位，才可开始下一笔。

把 “持仓状态” + “已用交易次数” 当作维度即可。

| 变量  | 取值      | 意义                             |
| ----- | --------- | -------------------------------- |
| `i`   | 0 … n     | 第 i 天                          |
| `t`   | 0 … k     | 已用完的交易次数                 |
| `pos` | 0 / 1 / 2 | 0 = 空仓；1 = 持多头；2 = 持空头 |

> **重点是要定义好持股的状态！！**

#### 转移规则（把动作翻译成方程）

1. **什么也不做** → `dfs(i+1, t, pos)`
2. **开仓**（仅当空仓且 `t<k`）
   - 开多：`dfs(i+1, t+1, 1) − price[i]`
   - 开空：`dfs(i+1, t+1, 2) + price[i]`
3. **平仓**（仅当持仓）
   - 平多：`dfs(i+1, t, 0) + price[i]`
   - 平空：`dfs(i+1, t, 0) − price[i]`

边界：走到最后一天 `i == n`，只有空仓才合法（利润 = 0），否则记 `-∞`。

> **时间复杂度** `O(n · k · 3)`；
> **空间复杂度** 由 `@lru_cache` 托管，同级别 `O(n · k · 3)`。

------

### **代码：**

```python
'''
Author: NEFU AB-IN
Date: 2025-06-07 22:27:36
FilePath: \LeetCode\CP158_2\b\b.py
LastEditTime: 2025-06-07 23:01:48
'''
# 3.8.6 import
import bisect
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta
from functools import cache, lru_cache, reduce
from heapq import heapify, heappop, heappush, nlargest, nsmallest
from itertools import combinations, compress, permutations, groupby, accumulate
from math import ceil, floor, fabs, gcd, log, exp, sqrt, hypot, inf
from string import ascii_lowercase, ascii_uppercase
from bisect import bisect_left, bisect_right
from sys import exit, setrecursionlimit, stdin
from typing import Any, Callable, Dict, List, Optional, Tuple
from random import randint

# Constants
N = int(2e5 + 10)
M = int(20)
INF = int(1e12)
OFFSET = int(100)
MOD = int(1e9 + 7)

# Set recursion limit
setrecursionlimit(int(1e9))


class Arr:
    array = staticmethod(lambda x=0, size=N: [x() if callable(x) else x for _ in range(size)])
    array2d = staticmethod(lambda x=0, rows=N, cols=M: [Arr.array(x, cols) for _ in range(rows)])
    graph = staticmethod(lambda size=N: [[] for _ in range(size)])


class Math:
    max = staticmethod(lambda a, b: a if a > b else b)
    min = staticmethod(lambda a, b: a if a < b else b)


class Std:
    pass

# ————————————————————— Division line ——————————————————————


class Solution:

    def maximumProfit(self, prices: List[int], k: int) -> int:
        n = len(prices)

        @lru_cache(None)
        def dfs(i: int, t: int, pos: int) -> int:
            if i == n:
                return 0 if pos == 0 else -INF
            # ① 今天什么都不操作
            res = dfs(i + 1, t, pos)

            price = prices[i]
            if pos == 0: # ② 可以开仓
                if t < k:
                    res = Math.max(res, dfs(i + 1, t + 1, 1) - price) # 开多
                    res = Math.max(res, dfs(i + 1, t + 1, 2) + price) # 开空
            elif pos == 1: # ③ 平掉多头
                res = Math.max(res, dfs(i + 1, t, 0) + price)
            else:		   # ④ 平掉空头
                res = Math.max(res, dfs(i + 1, t, 0) - price)

            return res

        ans = dfs(0, 0, 0)
        dfs.cache_clear()
        return ans

```

------

### **一句话总结重点：**

 只需把 “空仓 / 持多 / 持空” 视为 3 个状态，开仓立刻消耗 1 次交易、平仓不额外计数，再用 `dfs(i, t, pos)` ＋ `@lru_cache` 递归即可，把多空双方向和交易次数一次性搞定。



------

## [吃水果](https://www.acwing.com/problem/content/4499/)

**难度：** 中等

**标签：** 计数型DP、组合计数、记忆化搜索

------

### **题意与思路：**

**题目简述：**

有 $n$ 个小朋友排成一排，每人发一个水果，共有 $m$ 种水果，每种水果的数量都足够多。

要求总共有 **恰好 $k$ 个小朋友** 拿到的水果与他左边小朋友不同（第一个人左边没人，不计入），问有多少种发水果的方案，结果对 $998244353$ 取模。

------

**思路：**

我们用动态规划进行建模：

- **状态定义：**

  - 设 $dp[i][j]$ 表示前 $i$ 个小朋友中，**恰好有 $j$ 个** 与左边水果不同的分发方案数。

- **转移方式：**

  - 第 $i$ 个小朋友与左边相同（水果种类固定）：从 $dp[i-1][j]$ 转移
  - 第 $i$ 个小朋友与左边不同（水果种类有 $(m-1)$ 种可选）：从 $dp[i-1][j-1] \cdot (m - 1)$ 转移

  $$
  \boxed{dp[i][j] = dp[i-1][j] + dp[i-1][j-1] \cdot (m-1)}
  $$

- **边界初始化：**

  - $dp[1][0] = m$，表示第一个人有 $m$ 种选法，但不可能有“不同”的人。

- **最终答案：**

  - 输出 $dp[n][k]$ 即为所求。

------

**时间复杂度：** $O(n \cdot k)$，空间复杂度 $O(n \cdot k)$，也可优化为滚动数组。

------

### **代码：**



```python
'''
Author: NEFU AB-IN
Date: 2025-06-08
FilePath: /acwing/4496.py
Description: 记忆化搜索解法
'''

from functools import lru_cache
from sys import setrecursionlimit
setrecursionlimit(10000)

MOD = 998244353

n, m, k = map(int, input().split())

@lru_cache(None)
def dfs(i: int, j: int) -> int:
    """
    dfs(i, j): 表示前 i 个小朋友中，恰好有 j 个“不同”的分发方案数
    """
    # 不合法状态直接剪枝
    if j < 0 or j > i - 1:
        return 0
    # 边界：第一个人，必须 j == 0
    if i == 1:
        return m if j == 0 else 0

    # 情况一：第 i 个小朋友和左边相同
    same = dfs(i - 1, j)

    # 情况二：第 i 个小朋友和左边不同，有 m-1 种水果选
    diff = dfs(i - 1, j - 1) * (m - 1) % MOD

    return (same + diff) % MOD

print(dfs(n, k))
```



## [最大的和](https://www.acwing.com/problem/content/1053/)

**难度：** 中等

**标签：** **最大子段和**、**前后缀分解**

**题意与思路：**

最大子段和问题分为两种类型：**连续最大子段和**和**非连续最大子段和**。

1. **连续最大子段和**要求子段必须连续，其递推公式为：$dp[i] = \max(dp[i-1] + A[i], A[i])$，表示以第 $i$ 个元素结尾的最大连续子段和；初始条件为 $dp[0] = A[0]$。
2. **非连续最大子段和**，（最大子序列和），允许子段不连续，其递推公式为：$dp[i] = \max(dp[i-1], dp[i-1] + A[i], A[i])$，表示前 $i$ 个元素中非连续子段的最大和；初始条件为 $dp[0] = \max(A[0], 0)$。连续子段和更适合保留顺序的场景，而非连续子段和适用于子段无顺序限制的问题。

最终答案分别为 $\max(dp[i])$ 和 $dp[n-1]$。



我们无法一次性求出两个子段的组合，但可以**枚举“两个子段的分割点”**，然后用**前缀最大子段和 + 后缀最大子段和**的方法来做

1. 用前缀 Kadane 算法预处理 left[i]
2. 用后缀 Kadane 算法预处理 right[i]
3. 枚举断点 i，最大值为 max(left[i] + right[i+1])





```python
'''
Author: NEFU AB-IN
Date: 2023-03-23 21:42:54
FilePath: \Acwing\1051\1051.py
LastEditTime: 2023-03-23 21:52:59
'''
read = lambda: map(int, input().split())
from collections import Counter, deque
from heapq import heappop, heappush
from itertools import permutations

N = int(5e4 + 10)
INF = int(2e9)

dp, g, h, w = [0] * N, [0] * N, [0] * N, [0] * N

for _ in range(int(input())):
    n = int(input())
    w[1:] = list(read())

    s = -INF
    dp[0] = g[0] = -INF  # 这里设成负无穷 是因为每一段都不能是空的，无解就设成-INF
    for i in range(1, n + 1):
        # dp[i] = w[i] + max(0, dp[i - 1])
        s = max(0, s) + w[i]
        # g[i] = max(dp[i], g[i - 1])
        g[i] = max(s, g[i - 1])
    s = -INF
    dp[n + 1] = h[n + 1] = -INF
    for i in range(n, 0, -1):
        # dp[i] = w[i] + max(0, dp[i + 1])
        s = max(0, s) + w[i]
        # h[i] = max(dp[i], h[i + 1])
        h[i] = max(s, h[i + 1])

    res = -INF
    for i in range(1, n + 1):
        res = max(res, g[i] + h[i + 1])
    print(res)
```



## [最大上升子序列和](https://www.acwing.com/problem/content/3665/)

**难度：** 困难

**标签：** LIS模型、**树状数组**

**题意与思路：**

给定一个长度为$n$的整数序列$a = [a_1, a_2, \ldots, a_n]$，要求选出一个严格上升的子序列，使其元素和最大。

**暴力解法**：定义$dp[i]$为以$a[i]$结尾的严格上升子序列的最大和，枚举每个元素$a[i]$之前的所有元素$a[j]$，若$a[j] < a[i]$，则更新$dp[i] = \max(dp[i], dp[j] + a[i])$，最终结果为$\max(dp[i])$，时间复杂度为$O(n^2)$。

**优化解法**：使用树状数组，把原问题看成一个**前缀最大值查询问题**：

首先对$a$进行离散化，将值映射到排名，树状数组维护每个排名对应的最大子序列和；对于每个元素$a[i]$，查询树状数组中所有比$a[i]$小的排名的最大值，并用$a[i]$更新树状数组，时间复杂度优化至$O(n \log n)$。最终答案为树状数组中的最大值。



```python
# import
import sys
from collections import Counter, deque
from heapq import heappop, heappush
from bisect import bisect_left, bisect_right

# Final
N = int(1e5 + 10)
INF = int(2e9)

# Define
sys.setrecursionlimit(INF)
read = lambda: map(int, input().split())

tr = [0] * N


def lowbit(x):
    return x & -x


def add(x, v):
    while x < N:
        tr[x] = max(tr[x], v)
        x += lowbit(x)


def query(x):
    res = 0
    while x:
        res = max(res, tr[x])
        x -= lowbit(x)
    return res


n, = read()
a = list(read())

xs = a[:]
xs = sorted(list(set(xs)))

res = 0
for i in range(n):
    k = bisect_left(xs, a[i]) + 1  # 保证下标大于0
    s = query(k - 1) + a[i]
    res = max(res, s)
    add(k, s)

print(res)
```



## [最长公共上升子序列](https://www.acwing.com/problem/content/description/274/)

**难度：** 困难

**标签：** LIS模型、LCS模型、**LCIS**

**题意与思路：**

该题要求找到两个数组 $A$ 和 $B$ 的最长公共上升子序列（LCIS）的长度。公共上升子序列是指同时在 $A$ 和 $B$ 中出现，且元素严格递增的子序列。





状态表示：

`f[i][j]`代表所有a[1 ~ i]和b[1 ~ j]中以b[j]结尾的公共上升子序列的集合；
`f[i][j]`的值等于该集合的子序列中长度的最大值；
状态计算（对应集合划分）：

首先依据公共子序列中是否包含a[i]，将`f[i][j]`所代表的集合划分成两个不重不漏的子集：

1. 不包含a[i]的子集，最大值是`f[i - 1][j]`；
2. 包含a[i]的子集，将这个子集继续划分，依据是子序列的倒数第二个元素在b[]中是哪个数：
   1. 子序列只包含b[j]一个数，长度是1；
   2. 子序列的倒数第二个数是b[1]的集合，最大长度是`f[i - 1][1] `+ 1；
   3. …
   4. 子序列的倒数第二个数是b[j - 1]的集合，最大长度是`f[i - 1][j - 1] `+ 1；
   5. 如果直接按上述思路实现，需要三重循环：

```c++
for (int i = 1; i <= n; i ++ )
{
    for (int j = 1; j <= n; j ++ )
    {
        f[i][j] = f[i - 1][j];
        if (a[i] == b[j])
        {
            int maxv = 1;
            for (int k = 1; k < j; k ++ )
                if (a[i] > b[k])
                    maxv = max(maxv, f[i - 1][k] + 1);
            f[i][j] = max(f[i][j], maxv);
        }
    }
}
```









然后我们发现每次循环求得的maxv是满足a[i] > b[k]的`f[i - 1][k] `+ 1的前缀最大值。
因此可以直接将maxv提到第一层循环外面，减少重复计算，此时只剩下两重循环。

最终答案枚举子序列结尾取最大值即可



```c++
for (int i = 1; i <= n; i ++ )
{
    int maxv = 1;
    for (int j = 1; j <= n; j ++ )
    {
        f[i][j] = f[i - 1][j];
        if (a[i] == b[j]) f[i][j] = max(f[i][j], maxv);
        if (a[i] > b[j]) maxv = max(maxv, f[i - 1][j] + 1);
    }
}
```



## [最小数组和](https://leetcode.cn/problems/minimum-array-sum/description/)

**难度：** 困难

**标签：** DP，**记忆化搜索**、分类讨论

**题意与思路：**



题意：给定一个整数数组 `nums` 和三个整数 $k$、$op_1$、$op_2$，你可以对数组的每个数执行以下两种操作，每个数每个操作最多一次。

- **操作 1**：选择一个下标 `i`，将 `nums[i]` 除以 2，并 **向上取整** 到最接近的整数。你最多可以执行此操作 `op1` 次，并且每个下标最多只能执行**一次**。
- **操作 2**：选择一个下标 `i`，仅当 `nums[i]` 大于或等于 `k` 时，从 `nums[i]` 中减去 `k`。你最多可以执行此操作 `op2` 次，并且每个下标最多只能执行**一次**。

**注意：** 两种操作可以应用于同一下标，但每种操作最多只能应用一次。

返回在执行任意次数的操作后，`nums` 中所有元素的 **最小** 可能 **和** 。





解法：对于每个元素，可以选择不操作、执行操作 1（将当前数减半并向上取整）、执行操作 2（若当前数大于等于 $k$ 则减去 $k$），或者同时执行操作 1 和操作 2 组合。依次递归剩余的数组，同时递减对应的操作次数 (`op1` 和 `op2`)。



时间复杂度为 $O(n \cdot op_1 \cdot op_2)$，其中 $n$ 为数组 `nums` 的长度。由于每个状态只会计算一次，动态规划的时间复杂度等于状态个数乘以单个状态的计算时间。本题状态个数为 $O(n \cdot op_1 \cdot op_2)$，单个状态的计算时间为 $O(1)$，因此总时间复杂度为 $O(n \cdot op_1 \cdot op_2)$。





```python
class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        @lru_cache(None)  # 缓存装饰器，避免重复计算 dfs 的结果（记忆化）
        def dfs(i: int, op1: int, op2: int) -> int:
            if i < 0:
                return 0
            x = nums[i]
            res = dfs(i - 1, op1, op2) + x
            if op1:
                res = min(res, dfs(i - 1, op1 - 1, op2) + (x + 1) // 2)
            if op2 and x >= k:
                res = min(res, dfs(i - 1, op1, op2 - 1) + x - k)
                if op1:
                    y = (x + 1) // 2 - k if (x + 1) // 2 >= k else (x - k + 1) // 2 #注意：如果能先除再减，那么先除再减更优，否则只能先减再除。
                    res = min(res, dfs(i - 1, op1 - 1, op2 - 1) + y)
            return res
        return dfs(len(nums) - 1, op1, op2)
```



## [目标和](https://leetcode.cn/problems/target-sum/)

**难度：** 中等

**标签：** 经典DP、01背包衍生题

**题意与思路：**

**题意**: 给定一个非负整数数组 $nums$ 和一个整数 $target$，需要为数组中的每个元素添加 '+' 或 '-'，构造一个表达式使得其运算结果等于 $target$。返回所有满足条件的不同表达式的数量。

**思路**: 这个问题可以抽象为**0-1背包问题**（简化问题：找到所有子集，使得这些子集的和等于 (sum(nums) - target) / 2 或者 (sum(nums) + target) / 2，也就是**背包容量**，可以数学推出）。

![image-20250527121431624](D:\Code\OtherProject\AlgorithmList\images\image-20250527121431624.png)

通过 DFS 结合记忆化搜索求解。我们需要将数组 $nums$ 中的每个元素赋予一个 '+' 或 '-' 符号，使得表达式的总和等于目标值 $target$。将问题转换为背包问题后，目标是寻找所有组合，使得差值满足条件。

使用 DFS 枚举每个数是否选择正号或负号，并通过记忆化优化重复计算。在 DFS 中，每个状态由索引 $i$ 和当前差值 $c$ 决定：当 $i < 0$ 且 $c = 0$ 时，表示成功找到一种方案。状态转移公式为从上一个状态 $dfs(i-1, c)$（当前数不选）和 $dfs(i-1, c-nums[i])$（当前数选）中累加结果。如果当前体积不足，则直接跳过选择。通过递归转移，最终返回所有合法组合的数量。时间复杂度为 $O(n \cdot t)$，其中 $t$ 为目标和的范围。



```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        s = sum(nums) - abs(target)
        if s < 0 or s % 2:
            return 0
        m = s // 2  # 背包容量

        @cache  # 缓存装饰器，避免重复计算 dfs 的结果（记忆化）
        def dfs(i: int, c: int) -> int:
            if i < 0:
                return 1 if c == 0 else 0
            if c < nums[i]:
                return dfs(i - 1, c)  # 只能不选
            return dfs(i - 1, c) + dfs(i - 1, c - nums[i])  # 不选 + 选
        return dfs(len(nums) - 1, m)
```





当然可以，下面是你要的题目 “01 背包（划分权重）” 整理成与截图一致的美观格式：

------

## [划分权重（01背包建模）]([16届蓝桥杯14天国特冲刺营 - 01背包（划分权重） - 蓝桥云课](https://www.lanqiao.cn/courses/51805/learning/?id=4071544&compatibility=false))

| 题目语言                  | 属于哪类问题 | 状态表示             | 常用初始化         |
| ------------------------- | ------------ | -------------------- | ------------------ |
| “是否能”                  | 存在性       | `dp[j] = True/False` | `dp[0] = True`     |
| “有几种”                  | 方案数       | `dp[j] = 方案数`     | `dp[0] = 1`        |
| “最多能选几个” / “最大值” | 最大值       | `dp[j] = 最大值`     | `dp[0] = 0 / -INF` |
| “最少”                    | 最小值       | `dp[j] = 最小值`     | `dp[0] = 0 / INF`  |



> 在 01 背包中，**所有维度中涉及“个数、容量、使用次数”的维度都要倒序遍历**，否则会重复使用同一个物品。



**难度：中等**
 **标签：01 背包、子集划分、组合优化**

------

### **题意与思路：**

**题意：**
 给定 40 个整数，将它们任意划分成两个非空子集，使得两组的权值乘积最大。
 每组的权值是该组所有元素的和。划分的总权值为两组权值之积。

**思路：**

01 背包存在性问题



> 因为没有具体的选了什么，就给什么价值，所以考虑从限制条件中延展出的存在性



设总和为 `S`，问题等价于：
从数组中选出若干个数，使其和为 `x`，求 `x × (S - x)` 最大值。
只需用 01 背包判断哪些和 `x` 是可达的，再遍历 `x ∈ [1, S-1]`，取最大乘积。

------

### **状态表说明：**

| 状态名   | 含义                                      | 举例说明                        |
| -------- | ----------------------------------------- | ------------------------------- |
| `dp[i]`  | 是否可以从前若干个数中选出和为 `i` 的子集 | 即：是否存在一个子集和为 `i`    |
| `S`      | 所有数的总和                              | 最大子集和不会超过它            |
| `x(S-x)` | 每种子集和能形成的乘积                    | 乘积最大值发生在 `x ≈ S/2` 附近 |

------

### **代码**

```python
def max_partition_product(nums):
    S = sum(nums)
    dp = [False] * (S + 1)
    dp[0] = True  # 空集和为0是合法的

    for num in nums:
        for i in range(S, num - 1, -1):  # 01背包倒序
            dp[i] |= dp[i - num]

    max_prod = 0
    for x in range(1, S):  # 两组都至少一个数
        if dp[x]:
            max_prod = max(max_prod, x * (S - x))
    return max_prod

```



---

## [拆分互异正整数和（三维01背包）]([16届蓝桥杯14天国特冲刺营 - 01背包（2022） - 蓝桥云课](https://www.lanqiao.cn/courses/51805/learning/?id=4071546&compatibility=false))

**难度：中等偏上**  
**标签：整数划分、01 背包、多维 DP、组合计数**

---

### **题意与思路：**

**题意：**  
将 `2022` 拆成 `10` 个**互不相同的正整数之和**，问总共有多少种不同的拆法？  

- 顺序不同视为同一种（即组合问题）
- 数字要求互不相同，正整数

**思路：**  
本质是一个组合计数类的 **三维 01 背包** 问题：

- 枚举使用哪些数 `i ∈ [1, 2022]`
- 状态：选到和为 `j`，用了 `k` 个不同数
- 每个数最多选一次（互不相同 → 01 背包）

---

### **状态表说明：**

| 状态名        | 含义                                             | 举例说明                           |
| ------------- | ------------------------------------------------ | ---------------------------------- |
| `dp[i][j][k]` | 用前 `i` 个数，总和为 `j`，用了 `k` 个数的方案数 | 比如 `dp[2022][2022][10]` 即为答案 |
| 01背包        | 每个数字至多使用一次                             | 所以 j 和 k 维都必须倒序转移       |
| 滚动数组优化  | 由于只依赖 `i-1` 层，可以压缩为二维（见优化版）  | 空间复杂度从 O(N³) 降到 O(N²)      |

---

### **PYTHON 实现：标准三维 DP（原始写法）**

```python
n, k = 2022, 10
f = [[[0] * (k + 1) for _ in range(n + 1)] for _ in range(n + 1)]
f[0][0][0] = 1  # 初始状态

for i in range(1, n + 1):           # 枚举选的数
    for j in range(n + 1):          # 当前总和
        for cnt in range(k + 1):    # 当前用了几个数
            f[i][j][cnt] = f[i - 1][j][cnt]  # 不选 i
            if j >= i and cnt >= 1:
                f[i][j][cnt] += f[i - 1][j - i][cnt - 1]  # 选 i

print(f[n][n][k])
```

---

### **优化版：二维 DP + 倒序枚举**

```python
n, k = 2022, 10
dp = [[0] * (k + 1) for _ in range(n + 1)]
dp[0][0] = 1

for num in range(1, n + 1):
    for j in range(n, num - 1, -1):       # 01背包：从大到小枚举总和
        for cnt in range(k, 0, -1):       # 01背包：从大到小枚举个数
            dp[j][cnt] += dp[j - num][cnt - 1]

print(dp[n][k])
```

---

### 维度说明与倒序原理

| 维度       | 解释                     | 是否需要倒序 | 原因                                     |
| ---------- | ------------------------ | ------------ | ---------------------------------------- |
| 数字 `num` | 可选的数字，从 1 到 2022 | 不倒序       | 遍历的是物品本身，不会重复               |
| 总和 `j`   | 当前目标和               | ✅            | 避免重复使用当前数字（01 背包标准技巧）  |
| 个数 `cnt` | 当前选了多少个数         | ✅            | 确保每个数字只被用于一个组合中（互异性） |

## 更小的数

**难度：** 中等

**标签：** 区间 DP、记忆化搜索、枚举子区间

------

### 题意与思路

> 给定一个长度为 `n` 的数字字符串 `num`（允许前导 `0`），你可以 **至多一次** 选定一段连续子串并将其整体反转，得到新串 `num_new`。
>  统计一共有多少种不同的选择方式，使得 `num_new < num`；只要两次选择的区间在原串中的位置不完全相同，就视为两种不同方案。

#### 1. 区间状态设计

- 设 `cmp(l, r)` 表示 **原区间 `num[l…r]`** 与 **其反转** 的字典序比较结果
  - `-1` → 反转后更小 `0` → 相等 `1` → 反转后更大

#### 2. 状态转移（首尾对决）

- 若 `l ≥ r` (长度 ≤ 1)：两串完全一致 ⇒ `0`
- 若 `num[l] == num[r]`：比较内部子区间 `cmp(l+1, r-1)`
- 否则直接由首尾字符决定：
   `cmp(l,r) = -1` 若 `num[r] < num[l]` ，否则 `1`

#### 3. 枚举统计

- 双重循环遍历所有区间 `(l,r)`，若 `cmp(l,r) == -1` 则计入答案
- 利用记忆化 (`@lru_cache`) 让每个区间比较 **仅计算一次**，避免重复

------

### 复杂度分析

| 项目       | 复杂度                                       |
| ---------- | -------------------------------------------- |
| 状态数     | `O(n²)` 个区间                               |
| 单状态计算 | `O(1)`                                       |
| **总时间** | `O(n²)`                                      |
| **总空间** | `O(n²)`（递归栈深度 `O(n)`，缓存表 `O(n²)`） |

------

### Python 代码（3.8.6 可直接 AC）

```python
import sys
from functools import lru_cache

s = sys.stdin.readline().strip()
n = len(s)

@lru_cache(maxsize=None)
def cmp_interval(l: int, r: int) -> int:
    """
    比较原串 num[l..r] 与其反转后的字典序：
        -1 : 反转串更小
         0 : 相等
         1 : 反转串更大
    """
    if l >= r:                 # 长度 0/1
        return 0
    if s[l] == s[r]:           # 首尾相等 → 缩小区间
        return cmp_interval(l + 1, r - 1)
    return -1 if s[r] < s[l] else 1   # 首尾不等 → 直接决胜负

ans = 0
for l in range(n):
    for r in range(l, n):
        if cmp_interval(l, r) == -1:  # 反转后更小
            ans += 1
print(ans)
```

> **一句话总结重点：** 用 “首尾字符对决 + 记忆化缓存” 把区间比较降到 `O(1)`，再暴力枚举所有区间即可在 `O(n²)` 内完成计数。





## [健身](https://www.lanqiao.cn/courses/51805/learning/?id=4072894&compatibility=false)

**难度：** 中等偏上

**标签：** 完全背包、区间 DP、位运算（`2^k` 长度）

------

### 题意与思路（总览）

> 有 `n` 天，其中 `q` 天被其他安排占用；
>  有 `m` 个健身计划，第 `i` 个必须连续做 `len_i = 2^{k_i}` 天，完成得收益 `s_i`，且同一计划可无限次重复；
>  每天至多执行一个计划。问 **最大收益和**。

将「连续做 `len` 天」视作 **重量 = `len`、价值 = `gain` 的无限物品**，问题秒变 **带禁用日的一维完全背包**。下面给出两种实现思路。

------

### 方案一：线性 DP（整条时间线顺推）

一条时间线顺推的“一维完全背包 + 前缀判断”

| 关键点     | 说明                                                         |
| ---------- | ------------------------------------------------------------ |
| 状态       | `dp[i]`：做完 **前 `i` 天** 的最⼤收益                       |
| 转移       | 若 `day i` 被占用 → `dp[i]=dp[i-1]`<br />否则枚举所有计划 `(len_i, gain_i)` 并判断 `[i-len_i+1,i]` 这段是否无冲突。 `dp[i] = max(dp[i-1], max_{plans}( dp[i-len]+gain ))`，且需保证区间 `[i-len+1,i]` 无禁用日 |
| 判区间合法 | 预处理 `preBad[i] = 被占用日的前缀和`，O(1) 判断             |

**复杂度**
 时间 `O(n·m)`，空间 `O(n)`（可滚动到 `O(n)`/`O(1)` 皆可接受）。

#### 代码（使用你的模板）

```python
from sys import stdin, setrecursionlimit
setrecursionlimit(int(1e6))

# ---- 通用模板 ----
N = int(2e5 + 10); M = 20; INF = int(1e12); OFFSET = 100; MOD = int(1e9 + 7)
class Arr:
    array = staticmethod(lambda x=0, size=N: [x() if callable(x) else x for _ in range(size)])
class Math:
    max = staticmethod(lambda a, b: a if a > b else b)
class IO:
    input = staticmethod(lambda: stdin.readline().strip())
    read  = staticmethod(lambda: map(int, IO.input().split()))
# ------------------

n, m, q = IO.read()
bad = Arr.array(False, n + 1)
for day in IO.read():
    bad[day] = True

plans = []
for _ in range(m):
    k, s = IO.read()
    plans.append((1 << k, s))

preBad = Arr.array(0, n + 1)
for i in range(1, n + 1):
    preBad[i] = preBad[i-1] + bad[i]


def no_bad(l: int, r: int) -> bool:
    return preBad[r] - preBad[l-1] == 0


dp = Arr.array(0, n + 1)
for i in range(1, n + 1):
    if bad[i]:
        dp[i] = dp[i-1]
        continue
    best = dp[i-1]
    for length, gain in plans:
        l = i - length
        if l >= 0 and no_bad(l+1, i):
            best = Math.max(best, dp[l] + gain)
    dp[i] = best

print(dp[n])
```

------

### 方案二：分段 Knapsack（先切段，再各段完全背包）

#### 思路

1. 利用被占用日把 `[1,n]` 切成 **若干连续空闲段** `seg₁, seg₂, …`
2. 对每段长度 `L` 单独做一次完全背包：
   - `f[x]` = 恰用 `x` 天可得最大收益
   - 经典一维完全背包（物品仍是 `(len_i, gain_i)` 无限）
3. 各段结果相加即为答案。

**优势**

- 内存峰值 = `max(seg_len)`，不再随整条时间线线性增长。
- 每段可独立计算，易于并行或流式处理。

**复杂度**
 时间 `Σ(seg_len)·m  ≤ n·m`；空间 `O(max_seg_len)`。

#### 代码

```python
from sys import stdin, setrecursionlimit
setrecursionlimit(int(1e6))

# ---- 模板复用 ----
N = int(2e5 + 10); M = 20
class Arr:
    array = staticmethod(lambda x=0, size=N: [x() if callable(x) else x for _ in range(size)])
class Math:
    max = staticmethod(lambda a, b: a if a > b else b)
class IO:
    input = staticmethod(lambda: stdin.readline().strip())
    read  = staticmethod(lambda: map(int, IO.input().split()))
# ------------------

# 读入
n, m, q = IO.read()
bad = Arr.array(False, n + 2)       # n+1 作为哨兵
for d in IO.read():
    bad[d] = True


plans = [(1 << k, s) for k, s in (IO.read() for _ in range(m))]

# 切段
segments, cur = [], 0
for i in range(1, n + 2):           # 哨兵终止
    if i <= n and not bad[i]:
        cur += 1
    else:
        if cur: segments.append(cur)
        cur = 0

# 完全背包求一段
def solve(L: int) -> int:
    f = Arr.array(0, L + 1)
    for length, gain in plans:
        if length > L: continue
        for t in range(length, L + 1):
            f[t] = Math.max(f[t], f[t - length] + gain)
    return f[L]

# 汇总
ans = 0
for seg in segments:
    ans += solve(seg)
print(ans)
```



------

## [李白打酒加强版](https://www.lanqiao.cn/courses/51805/learning/?id=4072895&compatibility=false)

**难度：** 中等偏难（记忆化搜索 + 状态剪枝）

**标签：** 记忆化搜索、状态设计、递归剪枝、状态 DP 建模

------

### **题意与思路：**

李白一开始有 2 斗酒，要走完 `n` 家酒店和 `m` 朵花（顺序任意）。
 规则如下：

- 遇到酒店：酒量变为原来的两倍；
- 遇到花：酒量减 1，前提是当前酒量 ≥ 1；
- 要求：最终一次操作必须是**遇花**，并且**酒恰好喝光（0 斗）**。

------

#### **状态设计（记忆化搜索）**

定义状态函数 `dfs(i, j, o, pre)` 表示：

- 当前剩下 `i` 个酒店、`j` 朵花；
- 当前酒量为 `o`；
- `pre` 表示上一次操作（1 = 酒店，2 = 花），用于判断最后一步是否是遇花。

------

#### **终止条件**

```python
if i == 0 and j == 0 and o == 0 and pre == 2:
    return 1
```

- 所有事件走完；
- 酒量为 0；
- 最后一次是遇花。

------

#### **剪枝优化（非常关键）**

- `i < 0 or j < 0`: 不合法状态；
- `j > 0 and o == 0`: 没酒还想遇花；
- `o > j`: 酒比剩下的花还多，不可能刚好喝完（最多每花减 1）。

这些剪枝能极大压缩状态空间，是通过的重要保障。

------

### **代码：**

```python
from functools import lru_cache

MOD = 10 ** 9 + 7
n, m = IO.read()  # 快速输入封装，适合蓝桥杯

@lru_cache(None)
def dfs(i, j, o, pre):
    if i == 0 and j == 0 and o == 0 and pre == 2:
        return 1
    if i < 0 or j < 0:
        return 0
    if j > 0 and o == 0:
        return 0
    if o > j:
        return 0  # 关键剪枝：酒太多，花不够减完，无法收尾

    ans = 0
    if i > 0:
        ans += dfs(i - 1, j, o * 2, 1)
    if j > 0 and o > 0:
        ans += dfs(i, j - 1, o - 1, 2)
    return ans % MOD

print(dfs(n, m, 2, -1) % MOD)
```

## [机器人可以获得的最大金币数](https://leetcode.cn/problems/maximum-amount-of-money-robot-can-earn/)

**难度：** 中等

**标签：** DP，图论

**题意与思路：**

**题意：**题目给定一个大小为 $m \times n$ 的网格，每个格子包含一个整数 $coins[i][j]$，可以是正数、负数或零。机器人从左上角 $(0, 0)$ 出发，到右下角 $(m-1, n-1)$。每次只能向右或向下移动，机器人需要最大化所收集的金币数量，如果 $coins[i][j] \geq 0$，则机器人获得对应数量的金币；如果 $coins[i][j] < 0$，机器人会失去这些金币的绝对值；机器人有**两次**特殊能力，可以免除负金币的影响。需要返回机器人从起点到终点路径上可以获得的最大金币数。

**思路：**遇到这种图的题，从**左上角到右下角，想到DP，而不是开状态数组标记路径**。使用动态规划（DFS + 记忆化搜索）解决问题，定义状态为 $dfs(x, y, k)$，表示机器人在位置 $(x, y)$，还剩 $k$ 次特殊能力时的最大金币数。状态转移过程包括：

1. 如果当前位置为负数且 $k > 0$，可以选择使用特殊能力绕过负金币。
2. 向右或向下递归搜索，并累加当前位置的金币值。
3. 终点状态单独处理，如果有剩余特殊能力，需选择是否使用以最大化金币。



```python
class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        m, n = len(coins), len(coins[0])

        @lru_cache(None)
        def dfs(x, y, k):
            if not (0 <= x < m and 0 <= y < n):
                return -inf
            if x == m - 1 and y == n - 1:
                return max(coins[x][y], 0) if k else coins[x][y]
            
            ans = max(dfs(x + 1, y, k), dfs(x, y + 1, k)) + coins[x][y]
            if coins[x][y] < 0 and k > 0:
                ans = max(ans, dfs(x + 1, y, k - 1), dfs(x, y + 1, k - 1))
            
            return ans

        return dfs(0, 0, 2)
```


