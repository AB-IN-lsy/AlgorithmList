# DP

## [执行操作可获得的最大总奖励 II](https://leetcode.cn/problems/maximum-total-reward-using-operations-ii/description/)

**难度：** 困难

**标签：** 01 背包变种（存在性问题）、**位运算优化布尔型DP（子集和问题）**

**题意与思路：**

**题意：**给定一个整数数组 `rewardValues`，表示奖励的值，初始总奖励 $x$ 为 0，所有下标均未标记。

每次可以选择一个未标记的下标 $i$，如果 ，则可以将其加到 $x$ 并标记该下标。目标是通过最优操作顺序，使最终的总奖励最大，并返回该值。



**思路：**先将 `rewardValues` 按从小到大的顺序排序，确保尽可能选择更多的奖励值。

使用**动态规划**解决，**定义 $f[i][j]$ 表示从前 $i$ 个奖励中是否能通过某种选择组合使总奖励为 $j$**（类似01背包），转移方程为：

1. 不选第 $i$ 个奖励，状态不变：$f[i][j] = f[i-1][j]$。
2. 2. 选第 $i$ 个奖励，前提是当前 $j \geq v$ 且 $j - v < v$ （当前 $x$ 小于 $rewardValues[i]$）：$f[i][j] = f[i-1][j-v]$。
   3. 初始状态 $f[0][0] = \text{true}$。
   4. $f[i][j]=f[i−1][j] \ or \ f[i−1][j−v]$
   5. 通过遍历可能的总奖励值，取满足条件的最大 $j$ 即为结果。通过 bitset 优化可以提升效率。
   6. 优化为滚动数组：$dp[j] = dp[j] \ or\  dp[j - v]$
   7. 位运算优化：可以不用掌握（因为只有v, 2v这个区间的在变，所以我们需要让这个区间的数 或 上一个状态的 0, v。首先创建掩码，将 f 的低v位通过与运算取出来，左移v位，最后或即可）

这是传统01背包的位运算优化

用 `int` 表示 dp

- `f = 1` → 只表示 `dp[0] = True`
- 如果我们要把 `v` 加进去（就像前面的转移 `dp[j] |= dp[j - v]`）
  就写成：

```python
f = 1        # 二进制是 ...0001，表示 dp[0] = True

for v in [1, 3, 4]:
    f |= f << v
    
```





```python
class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        rewardValues.sort()
        rewardValues = [0] + rewardValues
        n = len(rewardValues) - 1
        m = rewardValues[-1] * 2  # 最大的可能是最大值*2

        dp = Arr.array(False, m + 1)
        dp[0] = True

        for i in range(1, n + 1):
            v = rewardValues[i]
            for j in range(2 * v - 1, v - 1, -1):
                dp[j] = dp[j] or dp[j - v]

        for j in range(m - 1, -1, -1):
            if dp[j]:
                return j
        return 0	
    
    
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        rewardValues.sort()
        f = 1
        for v in rewardValues:
            mask = (1 << v) - 1 # 筛出 pre < v 的状态
            # 2. f & mask , 从 f 中筛出低位部分
            # 3. (f & mask) << v , 把这些合法的状态 +v（左移 v 位）
            f |= (f & mask) << v 
        return f.bit_length() - 1
```



## [统计打字方案数](https://leetcode.cn/problems/count-number-of-texts/)

**难度：** 中等

**标签：** DP、**爬楼梯类型**

**题意与思路：**

**题意：**将一串数字转化为字符串，每个字母对应一种数字方案，比如 ′c′ 对应 ′222′，以及 ′d′ 对应 ′3′ 等等，求总共有几种可能的字符串。

**思路：**分段 + 每段爬楼梯，爬楼梯的进阶版。即 f[i] 表示长为 i 的只有一种字符的字符串所对应的文字信息种类数。比如222222，我可以将末尾的一个2变成字母，两个2变成一个字母，三个2变成一个字母，这样他们的状态来自22222，2222,222，也就是 f[i]=f[i−1]+f[i−2]+f[i−3]。同理四个字母的



`f[]`：用于处理最多能连 3 个的数字（如 2,3,4,5,6,8）

`g[]`：用于处理最多能连 4 个的数字（如 7,9）



```python
f = [1, 1, 2, 4]
g = [1, 1, 2, 4]
for _ in range(10 ** 5 - 3):  # 预处理所有长度的结果
    f.append((f[-1] + f[-2] + f[-3]) % MOD)
    g.append((g[-1] + g[-2] + g[-3] + g[-4]) % MOD)

class Solution:
    def countTexts(self, pressedKeys: str) -> int:
        ans = 1
        for ch, s in groupby(pressedKeys):
            m = len(list(s))
            ans = ans * (g[m] if ch in "79" else f[m]) % MOD
        return ans
```



### `groupby(iterable)` 简明速查



将 **连续相同元素** 分组，返回 `(key, group)` 对。

------

### 使用方式：

```python
from itertools import groupby

s = "112233"

for ch, g in groupby(s):
    print(ch, list(g))
```

输出：

```css
1 ['1', '1']
2 ['2', '2']
3 ['3', '3']
```



## [从栈中取出 K 个硬币的最大面值和](https://leetcode.cn/problems/maximum-value-of-k-coins-from-piles/)

**难度：** 中等

**标签：** DP、**分组背包**

**题意与思路：**

**题意**：给定一个二维数组 `piles`，其中每个子数组表示一堆硬币。每堆硬币的元素表示硬币的面值。你可以从任意一堆硬币中选择硬币，选择时需要从每一堆硬币的顶部开始选择，并且每次只能选择一个硬币。你最多可以选择 $k$ 个硬币。目标是返回选择的 $k$ 个硬币的最大总面值。



**思路**：**分组背包！**

**对每个堆求前缀和，这样选择每堆中的一个物品，对应原先的若干个物品**



问题转化：从 $n$ 个堆中选择物品，每堆选择至多一个物品（可以不选），要求物品的总数为 $k$，同时需要求物品的价值总和的最大值。

使用 `dfs(i, j)` 表示在考虑前 $i$ 个堆的情况下，选择 $j$ 个物品时的最大面值。

**时间复杂度**：$O(n \cdot k)$，其中 $n$ 是硬币堆的数量，$k$ 是最大可以选择的硬币数。每次递归会遍历当前堆中所有可能选取的硬币数量（最多 $k$ 个），并计算每种选择的累积面值。





```python
class Solution:
    def maxValueOfCoins(self, piles: List[List[int]], k: int) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i < 0:
                return 0
            # 不选这一组中的任何物品
            res = dfs(i - 1, j)
            # 枚举选哪个
            for w, v in enumerate(accumulate(piles[i][:j]), 1):
                res = max(res, dfs(i - 1, j - w) + v)
            return res
        return dfs(len(piles) - 1, k)
```



## [分割回文串 II](https://leetcode.cn/problems/palindrome-partitioning-ii/)

**难度：** 困难

**标签：** DP、字符串、回文

**题意与思路：**

**题意**：给定一个字符串 $s$，要求将其分割成一些子串，使得每个子串都是回文串，并返回最少的分割次数。

思路：使用 **动态规划 + 记忆化搜索** 来解决该问题。

定义 `dfs(r)` 表示将 `s[0:r]` 分割成回文子串所需的最小切割次数。首先，如果 `s[0:r]` 已经是回文串，则无需切割，直接返回 `0`。

否则，遍历所有可能的分割点 `l`，如果 `s[l:r]` 是回文，则在 `l-1` 和 `l` 之间切一刀，递归计算 `dfs(l-1) + 1`，取所有可能情况的最小值作为答案。此外，使用 `is_palindrome(l, r)` 判断 `s[l:r]` 是否是回文，避免重复计算，通过 `@cache` 进行记忆化优化。

复杂度：由于 `@cache` 进行记忆化，`dfs(r)` 只计算一次，每次计算最多 $O(n)$，最终复杂度为 **$O(n^2)$**。





```python
class Solution:
    def minCut(self, s: str) -> int:
        # 返回 s[l:r+1] 是否为回文串
        @cache  # 缓存装饰器，避免重复计算 is_palindrome（一行代码实现记忆化）
        def is_palindrome(l: int, r: int) -> bool:
            if l >= r:
                return True
            return s[l] == s[r] and is_palindrome(l + 1, r - 1)

        @cache  # 缓存装饰器，避免重复计算 dfs（一行代码实现记忆化）
        def dfs(r: int) -> int:
            if is_palindrome(0, r):  # 已是回文串，无需分割
                return 0
            res = inf
            for l in range(1, r + 1):  # 枚举分割位置
                if is_palindrome(l, r):
                    res = min(res, dfs(l - 1) + 1)  # 在 l-1 和 l 之间切一刀
            return res

        return dfs(len(s) - 1)
```



## [删除一次得到子数组最大和](https://leetcode.cn/problems/maximum-subarray-sum-with-one-deletion/description/)

**难度：** 中等

**标签：** 最大子数组和

**题意与思路：**

**题意**：给定一个整数数组 $arr$，我们需要找出一个非空子数组，使得在**一次可选的删除操作**后，该子数组的元素和最大。换句话说，允许我们删除数组中的一个元素，剩余的子数组必须至少包含一个元素，返回该子数组中的最大元素和。

思路：

本题采用动态规划的方法，通过两个状态变量 $dp0$ 和 $dp1$ 来表示两个不同的状态：

$dp0[i]$：表示不删除元素的情况下，以 $arr[i]$ 结尾的子数组的最大和。

$dp1[i]$：表示已经删除过一个元素的情况下，以 $arr[i]$ 结尾的子数组的最大和。





1. 第一个转移方程: $dp0[i] = \max(dp0[i-1] + arr[i], arr[i])$

表示在不删除的情况下，以 `arr[i]` 为结尾的非空子数组的最大和 `dp[i][0]` 与 `dp[i - 1][0]` 有关，当 `dp[i - 1][0] > 0` 时，直接将 `arr[i]` 与 `i - 1` 时的最大非空子数组连接时，取得最大和，否则只选 `arr[i]` 时，取得最大和。



2. 第二个转移方程 $dp1[i] = \max(dp1[i-1] + arr[i], dp0[i-1])$

表示在删除一次的情况下，以 `arr[i]` 为结尾的非空子数组有两种情况：

1. 不删除 `arr[i]`，那么选择 `arr[i]` 与 `dp[i - 1][1]` 对应的子数组（已执行一次删除）。
2. 删除 `arr[i]`，那么选择 `dp[i - 1][0]` 对应的非空子数组（未执行一次删除，但是等同于删除了 `arr[i]`）。

`dp[i][1]` 取以上两种情况的最大和的最大值。



```python
class Solution:
    def maximumSum(self, arr: List[int]) -> int:
        @cache  # 记忆化搜索
        def dfs(i: int, j: int) -> int:
            if i < 0: 
                return -inf  # 子数组至少要有一个数，不合法
            if j == 0: 
                return max(dfs(i - 1, 0), 0) + arr[i]
            return max(dfs(i - 1, 1) + arr[i], dfs(i - 1, 0))
        return max(max(dfs(i, 0), dfs(i, 1)) for i in range(len(arr)))
```







记忆化搜索


```python
from functools import cache
from typing import List

class Solution:
    def maximumSum(self, arr: List[int]) -> int:
        n = len(arr)
        NEG_INF = float('-inf')

        @cache
        def dfs(i: int, deleted: int) -> int:
            """
            i      : 当前位置（必须选 arr[i] 作为结尾）
            deleted: 0 -> 还没删过，1 -> 已经删过一次
            返回值  : 以 arr[i] 结尾的合法最大和
            """
            if i < 0:
                return NEG_INF       # 选不到任何元素→非法
            if deleted == 0:         # 还没删过
                return max(dfs(i-1, 0), 0) + arr[i]
            else:                    # 已经删过一次
                keep_i   = dfs(i-1, 1) + arr[i]   # 不再删（之前删过了），接上 arr[i]
                drop_i   = dfs(i-1, 0)            # 把 arr[i] 当成唯一一次删除
                return max(keep_i, drop_i)

        ans = NEG_INF
        for i in range(n):
            ans = max(ans, dfs(i, 0), dfs(i, 1))
        return ans
```

其中 $j = 0$ 表示不能删除数字，$j = 1$表示必须删除一个数。

因此，定义$dfs(i, j)$表示子数组的右端点下标是$i$，不能/必须删除数字的情况下，子数组元素和的最大值。

根据上面讨论出的子问题，可以得到：

- 如果 $j = 0$（不能删除）：
  - 如果不考虑 $arr[i]$左边的数，那么$dfs(i, 0) = arr[i]$。
  - 如果考虑 $arr[i]$左边的数，那么$dfs(i, 0) = dfs(i - 1, 0) + arr[i]$。
- 如果 $j = 1$（必须删除）：
  - 如果不删除 $arr[i]$，那么 $dfs(i, 1) = dfs(i - 1, 1) + arr[i]$。
  - 如果删除 $arr[i]$，那么 $dfs(i, 1) = dfs(i - 1, 0)$。

取最大值，就得到了 $dfs(i, j)$。写成式子就是

$$\begin{align*}
dfs(i, 0) &= \max(dfs(i - 1, 0), 0) + arr[i] \\
dfs(i, 1) &= \max(dfs(i - 1, 1) + arr[i], dfs(i - 1, 0))
\end{align*}$$



递归边界：$dfs(-1, j) = -\infty$。这里 $-1$表示子数组中「没有数字」，但题目要求子数组不能为空，所以这种情况不合法，用$-\infty$表示，这样取$\max$的时候就自然会取到合法的情况。

递归入口：$dfs(i, j)$。枚举子数组右端点 $i$，以及是否需要删除数字 $j = 0, 1$，取所有结果的最大值，作为答案。





## [**到达第 K 级台阶的方案数**]()

**难度：** 中等偏上

**标签：** 记忆化搜索、剪枝、指数跳跃、DFS

------

### **题意与思路：**

> **题面简述**
>
> - 从台阶 `1` 出发，目标到达台阶 `k`。
> - 有一个变量 `jump`，初值 `0`。
> - **一次操作**有两种选：
>   1. **上跳**到 `i + 2^jump`，然后 `jump += 1`
>   2. **下走**到 `i − 1`，但下走 **不能连续用**，且在 0 层不能用
> - 抵达 `k` 后还可以继续走，**每次回到 k 都算不同方案**
> - 求所有合法操作序列条数，结果不取模（题目保证结果 ≤ 1e9）

------

#### 状态设计

| 变量      | 含义                                                         |
| --------- | ------------------------------------------------------------ |
| `n`       | 当前所处台阶                                                 |
| `is_back` | 是否刚刚用过“向下走”0 ⇒ 上一动作不是下走，1 ⇒ 刚用了下走，因此下一步禁止继续下走 |
| `jump`    | 当前上跳对应的指数 `2^jump`                                  |

因此一个状态用三元组 `(n, is_back, jump)` 描述。

> **对于这种倍增往上的题，一定要考虑清楚边界和剪枝**
>
> **剪枝关键**：
> 当 `n - (not is_back) > k` 时，一定到不了 k：
>
> - 如果上一步不是下走，我们**还有**下走机会，最远能往下再走 1 格；
> - 若此时 `n-1 > k`，哪怕立刻下走也超 k；只能继续上跳，位置只会更大。
>   所以直接 `return 0`。

#### 递归转移

```text
dfs(n, is_back, jump):
    if n - (not is_back) > k: return 0        # 剪枝
    res = 1  if n == k  else 0                # 经过 k 就计一次

    # 🚶‍♂️ 下走（若没刚用过 & 不在 0）
    if not is_back and n > 0:
        res += dfs(n-1, True,  jump)

    # 🦅 上跳
    res += dfs(n + 2**jump, False, jump+1)

    return res
```

- **计数位置**：只要当前 `n == k` 就先把方案数 `+1`，因为题目允许“再离开再回来”视为不同方案。
- **下走** 把 `is_back` 置 `True`，禁止连续下走；
  **上跳** 把 `is_back` 置 `False` 并 `jump+1`，恢复下走资格。

#### 复杂度

- 状态上界：
  - `n` 最多涨到 `k + 1` 附近就会被剪掉；
  - `jump` 最多涨到 `log₂(k)+1`；
  - `is_back` 只有 0/1
    因此总状态 ≈ `O(k log k)`（`k ≤ 1000` 时 1e4 量级）。
- `@lru_cache` 保证每状态只算一次，递归深度 ≤ `log₂(k)`，运行稳。

------

### **代码：**

```python
from functools import lru_cache
from sys import setrecursionlimit
setrecursionlimit(10000)

class Solution:
    def waysToReachStair(self, k: int) -> int:
        @lru_cache(None)
        def dfs(n: int, is_back: bool, jump: int) -> int:
            # 剪枝：即使马上下走 1 步也还是 > k，则永远回不到 k
            if n - int(not is_back) > k:
                return 0

            # 本格计数：只要到过 k 就累计一次
            res = int(n == k)

            # ① 下走（不能连续，下走后禁止再次下走）
            if not is_back and n > 0:
                res += dfs(n - 1, True, jump)

            # ② 上跳
            res += dfs(n + (1 << jump), False, jump + 1)
            return res

        return dfs(1, False, 0)
```

------

### **补充 · 关键小技巧**

| Trick                      | 说明                                          |
| -------------------------- | --------------------------------------------- |
| `n - int(not is_back) > k` | 用 1 行表达“最乐观也回不去 k” 的剪枝          |
| `int(n == k)`              | Python 布尔值可直接当 0/1，加法计方案一行搞定 |
| `1 << jump`                | 比 `2 ** jump` 快一截，跳跃长度位运算         |
| `lru_cache(None)`          | 缓存所有状态，无上限但状态体量可控            |



## [最高乘法得分](https://leetcode.cn/problems/maximum-score-of-good-subarray/)

**难度：** 中等

**标签：** 记忆化搜索、动态规划、区间选取

------

### **题意**

- 给定长度 **恰为 4** 的数组 `a = [a0, a1, a2, a3]`。
- 给定长度 **≥ 4** 的数组 `b`。
- 需在 `b` 中选出 **4 个下标** `i0 < i1 < i2 < i3`，得分定义为

$a_0\!\times\!b[i_0]\;+\;a_1\!\times\!b[i_1]\;+\;a_2\!\times\!b[i_2]\;+\;a_3\!\times\!b[i_3].$

- 求得分最大值（一定存在解，因为 `|b| ≥ 4`）。

------

### **思路（记忆化搜索版）**

#### 1 | 状态设计

令

- `idx`：当前正考察 `b` 的位置（0 ≤ `idx` ≤ n），
- `k` ：已经选了多少个下标（0 ≤ `k` ≤ 4）。

定义

$dfs(idx,\;k)=\text{在区间 }b[idx:] \text{ 内再选 }(4-k)\text{ 个下标所能取得的最大额外得分}.$

> 递归出口
>
> - 若 `k == 4` → 已经选满 4 个，额外得分 0；
> - 若 `idx == n` 但 `k < 4` → 无法再选，记作 `-∞`（无效状态）。

用 `dfs(idx, k)` 表示“到下标 idx、已选 k 个”时的最优子结构，再做「选 / 不选」二分支即可。

#### 2 | 状态转移

在位置 `idx` 有两种决策：

| 决策              | 得分贡献        | 下一状态          |
| ----------------- | --------------- | ----------------- |
| **跳过** `b[idx]` | 0               | `dfs(idx+1, k)`   |
| **选择** `b[idx]` | `a[k] * b[idx]` | `dfs(idx+1, k+1)` |

取两者最大即可。

#### 3 | 复杂度

- **状态数** ≈ `n × 4`
- 每个状态只做 O(1) 转移 → **时间** `O(n)`
- 记忆化表 `O(n × 4)` → **空间** `O(n)`

------

### **代码**

```python
from typing import List
from functools import lru_cache

class Solution:
    def maximumScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)

        @lru_cache(None)
        def dfs(idx: int, k: int) -> int:
            """idx: 当前下标，k: 已选元素个数"""
            if k == 4:                     # 4 个都选完
                return 0
            if idx == n:                   # b 用完但未选满
                return -10**18             # 负无穷，表示无效
            # ① 跳过 b[idx]
            res = dfs(idx + 1, k)
            # ② 选择 b[idx]
            take = a[k] * b[idx] + dfs(idx + 1, k + 1)
            res = max(res, take)
            return res

        return dfs(0, 0)
```

## [摘樱桃](https://leetcode.cn/problems/cherry-pickup/description/?envType=problem-list-v2&envId=TNm66WyS)

**难度：** 困难
**标签：** 状态压缩 DP、记忆化搜索、栅格路径

------

### **题意**

- `grid[i][j] = -1 / 0 / 1` 分别代表 **荆棘 / 空地 / 1 颗樱桃**。
- 先从 `(0,0)` 走到 `(n-1,n-1)`（只能 **→ / ↓**），再走回 `(0,0)`（只能 **← / ↑**）。
- 经过樱桃格就摘下樱桃并把该格变空。
- 求能摘到的 **最多樱桃数**；若往返都不可达，输出 `0`。

------

### **思路**

#### 1. 行程“折返”⇢“两人同步前进”　(经典转换)

往返一次 = **两个人** 同时从 `(0,0)` 出发，各走到 `(n-1,n-1)`，
 每时刻二人都走 **一步**（右或下），共 `T = 2n-2` 步。

- 若其中任何一步踩到荆棘，则该路径失效；
- 二人走到 **同一格** 只算一颗樱桃。

#### 2. 状态设计（压掉一维）

设

- 总步数 `t` (`0 ≤ t ≤ T`)
- 第 1 人列坐标 `j` （行坐标自动是 `i1 = t - j`）
- 第 2 人列坐标 `k` （行坐标 `i2 = t - k`）

> **状态**
>
> $f(t,\,j,\,k)=\text{走了 }t\text{ 步后，二人分别在 }(t-j,\,j)\text{ 与 }(t-k,\,k)\text{ 时能摘到的最多樱桃数}$

- 无需额外存行坐标，三维即可（`t`、`j`、`k`）。
- 合法性条件：`0 ≤ j,k ≤ n-1` 且 `t-j,t-k ∈ [0,n-1]`，且两格都不是荆棘。

### 3. 转移方程

每人上一步要么来自 **左**（列 –1）要么来自 **上**（行 –1），故共有 `2×2=4` 组合：

```text
f(t,j,k) = max(
    f(t-1, j    , k    ),  # 两人都从上面来
    f(t-1, j    , k-1 ),  # 人1上，人2左
    f(t-1, j-1 , k    ),  # 人1左，人2上
    f(t-1, j-1 , k-1 )   # 两人都从左来
) + cur
```

其中

```
cur = grid[t-j][j]                 # 人1所在格
    + (grid[t-k][k] if j != k else 0)  # 同格时别重复加
```

### 4. 边界 & 答案

- `t=0`、`j=k=0` ：`f(0,0,0)=grid[0][0]`
- 不可达或踩荆棘 → 记为 `-∞`（用大负数）。
- 结果 `f(T,n-1,n-1)`，若 <0 则输出 0。

### 5. 复杂度

- 状态量 `T·n·n = (2n)·n² ≈ 2n³`，`n ≤ 50` 时约 `250 000`，可接受。
- 记忆化 / 自底向上都行，空间 `O(n²)`（滚动数组）或 `O(n³)`（缓存）。

------

### **代码**

```python
from typing import List
from functools import lru_cache

class Solution:
    def cherryPickup(self, grid: List[List[int]]) -> int:
        n = len(grid)
        T = 2 * n - 2
        NEG_INF = -10**9

        @lru_cache(None)
        def dfs(t: int, j: int, k: int) -> int:
            """
            已走 t 步（0-index），
            第 1 人在 (t-j, j)，第 2 人在 (t-k, k)
            返回最大可摘樱桃数
            """
            # 坐标合法性 & 避荆棘
            i1, i2 = t - j, t - k
            if min(i1, i2, j, k) < 0 or max(i1, i2, j, k) >= n:
                return NEG_INF
            if grid[i1][j] == -1 or grid[i2][k] == -1:
                return NEG_INF

            # 起点
            if t == 0:                       # 必然 j = k = 0
                return grid[0][0]

            # 上一步四种组合
            best_prev = max(
                dfs(t - 1, j,     k    ),
                dfs(t - 1, j,     k - 1),
                dfs(t - 1, j - 1, k    ),
                dfs(t - 1, j - 1, k - 1),
            )

            if best_prev == NEG_INF:
                return NEG_INF

            cur = grid[i1][j]
            if j != k:                       # 不在同格，才加第二个人的樱桃
                cur += grid[i2][k]
            return best_prev + cur

        ans = dfs(T, n - 1, n - 1)
        return max(ans, 0)                   # 不可达时返回 0
```

------

### 一句话总结重点

把“往返一次”转化为“两个人同步出发到终点”，用 `t+j+k = constant` 压掉一维后做三维 DP；4 个转移、合法性校验，再加“同格不重摘”即可得到 `O(n³)` 解。







## [子2023](https://www.lanqiao.cn/courses/51805/learning/?id=4093712&compatibility=false)

**难度：** 中等 
**标签：** DP、记忆化搜索、子序列计数  

---

### 题意与思路

给定一个字符串序列  
```

S = "12345678910111213...20222023"

```
我们想要统计有多少种【不要求连续】地从 S 中选出子序列，使得拼起来正好等于 `"2023"`。

- 这是一个典型的「长文本 + 短模式串」的子序列计数问题。  
- 我们用记忆化搜索（DFS + `lru_cache`）来做：  
  1. 维护状态 `(i, j)`，表示当前在 S 的第 `i` 个字符处，已经匹配到模式串 P 的第 `j` 个字符。  
  2. 每一步可做两件事：  
     - **跳过** S[i]： 走到 `(i+1, j)`  
     - **匹配** S[i] == P[j]：走到 `(i+1, j+1)`  
  3. 用 `@lru_cache` 缓存每个 `(i,j)` 的结果，确保全局只访问大约 `O(|S|×|P|)` 个不同状态。  
  4. 边界：  
     - 若 `j == len(P)`，说明 P 已全部匹配，返回 `1` 种方案。  
     - 若 `i == len(S)` 还没匹配完 P，则返回 `0`。  

这种写法符合「先试跳过，再试匹配」的思路，代码结构清晰，易读易写。

---

### 代码

```python
# -*- coding: utf-8 -*-
import sys
from functools import lru_cache

# Python 3.8.6 支持，提升递归深度
sys.setrecursionlimit(int(1e7))

def count_2023_subseq_memo(S: str) -> int:
    P = "2023"
    n, m = len(S), len(P)

    @lru_cache(None)
    def dfs(i: int, j: int) -> int:
        # 如果已经全匹配，算作一种方案
        if j == m:
            return 1
        # 主串走到末尾还没匹配完，方案无效
        if i == n:
            return 0

        # 1) 跳过 S[i]
        res = dfs(i+1, j)
        # 2) 如果 S[i] == P[j]，拿它当配对字符
        if S[i] == P[j]:
            res += dfs(i+1, j+1)
        return res

    return dfs(0, 0)

if __name__ == "__main__":
    # 构造 S = "1","2","3",...,"2023" 拼接后的长串
    S = "".join(str(i) for i in range(1, 2024))
    ans = count_2023_subseq_memo(S)
    print(ans)
# 运行后输出（即子序列“2023”的总方案数）：
5484660609
```

------



```python
def solve():

    s = str("".join(map(str, range(1, 2024))))

    cnt_ = [Counter() for _ in range(len(s))]

    for i in range(len(s)):
        cnt_[i] = cnt_[i - 1]
        if s[i] == '2':
            cnt_[i]['2'] += 1
            cnt_[i]['202'] += cnt_[i]['20']
        if s[i] == '0':
            cnt_[i]['20'] += cnt_[i]['2']
        if s[i] == '3':
            cnt_[i]['2023'] += cnt_[i]['202']

    print(cnt_[len(s) - 1]['2023'])

    return
```





**点评：**

- 记忆化搜索用递归把问题拆成「跳过 / 匹配」两种决策，思考简单。
- 借助 `lru_cache`，状态访问上限约为 `|S|×|P|≈7000×4`，实测秒级内可完成。
- 如果担心递归性能，也可改用「倒序滚动数组」的迭代 DP 版本，复杂度同样为 `O(|S|·|P|)`。

