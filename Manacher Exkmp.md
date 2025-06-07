# Manacher算法



求解字符串s的最长回文子串长度只是Manacher算法最微不足道的应用，其中理解回文半径数组有大用处

```python
class Manacher:
    """Manacher"""

    def __init__(self, raw: str):
        self.raw = raw
        self.ss = self._build_extended(raw)
        self.n = len(self.ss)
        self.p = Arr.array(0, self.n)  # 回文半径数组

    def _build_extended(self, s: str) -> List[str]:
        m = len(s)
        ext_len = m * 2 + 1           # 扩展串长度
        ext = Arr.array('#', ext_len)  # 初始化全部为 '#'
        for j in range(m):
            ext[2 * j + 1] = s[j]
        return ext

    def longest_pal_length(self) -> int:
        """
        返回原串中的最长回文子串长度（原串下标意义）
        """
        if not self.raw:
            return 0

        c = 0   # 当前最右回文的中心位置
        r = 0   # 当前回文覆盖的右边界（开区间）
        ans = 0

        for i in range(self.n):
            if i < r:
                mirror = 2 * c - i
                length = Math.min(self.p[mirror], r - i)
            else:
                length = 1

            # 继续向外扩展
            while (
                i + length < self.n and
                i - length >= 0 and
                self.ss[i + length] == self.ss[i - length]
            ):
                length += 1

            if i + length > r:
                r = i + length
                c = i

            self.p[i] = length  # 最终赋值
            ans = Math.max(ans, length)

        return ans - 1  # 回文长度 = 半径 - 1
```





## 原理

------

### 1. 暴力方法如何寻找最长回文子串

- 枚举每个中心，向左右扩展判断是否为回文；
- 每次中心最多扩展 $O(n)$ 次，总复杂度为 $O(n^2)$；
- 奇偶长度需要分类处理，逻辑较为繁琐。

------

### 2. Manacher 扩展串的构造思想

- 在原字符串中插入特殊字符（如 `#`）形成扩展串（如 `"aba"` → `"#a#b#a#"`）；
- 好处：
  - 不用分别处理奇偶长度的回文；
  - 插入字符不影响回文结构；
- 扩展串长度为 $2n + 1$。

------

### 3. 回文半径与真实回文长度的关系

- 回文半径数组：`p[i]` 表示以扩展串位置 `i` 为中心的回文半径（包括中心）；
- 对应原串的真实回文长度为：
   `实际长度 = p[i] - 1`

------

### 4. 扩展串回文结束位置与原串下标的关系

- 回文在扩展串的右端点为 `i + p[i] - 1`
- 映射回原串的真实下标为：
   `原串位置 = (i + p[i] - 1) // 2`

------

### 5. 三个关键变量的含义

- `p[]`: 回文半径数组，记录以每个位置为中心的最长回文长度；
- `r`: 当前回文能扩展到的最右边界；
- `c`: 这个最右边界 `r` 所对应的**回文中心位置**；
- 每次遍历 `i` 时利用 `p`, `r`, `c` 做出判断来加速。

------

### 6. Manacher 算法的核心加速逻辑

当扫描到扩展串的位置 `i` 时，根据它与当前回文区间 `[c - p[c] + 1, r]` 的关系分四种情况：

#### a. i 不在 r 内（i > r）

- 无法复用信息；
- 直接以 `i` 为中心向外扩展。

#### b. i 被 r 包住，且对称点 `j = 2*c - i` 的回文半径 **完全在当前回文区间内**

- 直接复制对称点：
   `p[i] = p[j]`

#### c. i 被 r 包住，但对称点 j 的回文超出了当前区间

- 只能用到边界：
   `p[i] = r - i`

#### d. i 被 r 包住，且对称点 j 的回文**刚好撞到 r 的边界**

- 初始设置 `p[i] = r - i`；
- 然后从 `r` 开始暴力扩展。

### 7. 时间复杂度分析

- 每个字符只会被扩展一次（即使暴力扩展也不会重复）；
- 总时间复杂度为：**O(n)**

------

### 8. 代码讲解 & while 循环的统一处理技巧

- 通过一个 `while` 循环统一处理所有扩展逻辑；
- 利用 `r` 和 `c` 动态维护当前的最右回文和中心；
- 精妙之处在于：不需要人工分类讨论四种情况，while 内部自动处理！



------

# 扩展 KMP（Z-algorithm / Z 函数）

扩展 KMP，又称 **Z 算法 / Z 函数（Z-array）**，是用于求解字符串的**前缀匹配信息**的重要算法。
理解了 Manacher 算法，再来学习扩展 KMP 将会非常轻松。

------

## 原理

------

### z 数组、匹配右边界 `r`、匹配中心 `c`

- `z[i]` 表示从位置 `i` 开始，与 `s` 的前缀 `s[0:]` 匹配的最长公共前缀长度；
- `r` 表示当前已知匹配区间的最右边界；
- `c` 是这个最大右边界对应的匹配中心。

------

### 四种加速情况的图解逻辑

当我们来到某个位置 `i`，尝试计算 `z[i]` 时，如何利用已有的 `z`, `r`, `c` 来跳过冗余匹配：

#### a. i 不在 r 的覆盖范围内

- 即 `i > r`；
- 无法使用任何历史信息，直接暴力扩展。

#### b. i 在 r 内，且对称点 `j = i - c` 的 `z[j]` 没越界

- 即 `z[j] < r - i + 1`；
- 表示 `z[j]` 完全落在当前大扩展区间内；
- 直接令 `z[i] = z[j]`。

#### c. i 在 r 内，且 `z[j]` 超过了边界

- 即 `z[j] > r - i + 1`；
- 最多能用到的就是 `r - i + 1`；
- 所以直接令 `z[i] = r - i + 1`。

#### d. i 在 r 内，且 `z[j]` 恰好撞到右边界

- 即 `z[j] == r - i + 1`；
- 表示不确定右边能不能继续扩展，需要从 r 开始继续比较；
- 从 `r` 之后的位置手动暴力扩展。

> 实际代码中，上述四种情况通过一个 while 循环统一处理。

------

### 时间复杂度分析

- 每个字符最多被右端扩展一次；
- 总复杂度为：**O(n)**





# [3031. 将单词恢复初始状态所需的最短时间 II](https://leetcode.cn/problems/minimum-time-to-revert-word-to-initial-state-ii/)


难度：困难

标签：Z函数

题意和思路：



## 题意简化

给定一个字符串 `word` 和整数 `k`，你每秒执行如下操作：

- **删除前 k 个字符**
- **在末尾添加任意 k 个字符**

问：**最少需要几秒，才能让字符串再次等于初始的 `word`？**
 你必须至少执行一次操作。

------

## 解题思路

### 本质转换

你每次做的操作：删掉前缀、补上后缀，其实是一个**“整体右移”**的过程。
 执行 `t` 次操作，实际就是让：

```
word[k * t:] + ???  ==  word
```

我们要求 `word` 的某个后缀 `word[k * t:]` 和原串的前缀相同（能“拼回”原始串），那么就能复原。

------

### Z 函数的应用

我们构造原串的 Z 数组：

- `z[i]` 表示：`word[i:]` 与 `word` 前缀的最长公共前缀长度；
- 如果存在某个 `i = t * k` 满足：`z[i] == n - i`，说明从位置 `i` 开始的后缀等于前缀 → 可复原！

------

### 扫描过程

- 从 `i = k, 2k, 3k, ...` 开始遍历；
- 找最小的 `t` 满足：`z[t * k] == n - t * k`；
- 没找到就说明只好走满，就是把原串全部删掉：`⌈n / k⌉` 步，强行转回原串。







```python
# Constants
N = int(2e5 + 10)
M = int(20)
INF = int(1e12)
OFFSET = int(100)
MOD = int(1e9 + 7)

# Set recursion limit
setrecursionlimit(int(2e9))
class Arr:
    array = staticmethod(lambda x=0, size=N: [x() if callable(x) else x for _ in range(size)])
    array2d = staticmethod(lambda x=0, rows=N, cols=M: [Arr.array(x, cols) for _ in range(rows)])
    graph = staticmethod(lambda size=N: [[] for _ in range(size)])


class Math:
    max = staticmethod(lambda a, b: a if a > b else b)
    min = staticmethod(lambda a, b: a if a < b else b)


class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        z = Arr.array(0, n)

        c, r = 1, 1
        z[0] = n

        for i in range(1, n):
            length = 0
            if r > i:
                length = Math.min(r - i, z[i - c])
            else:
                length = 0
            
            while i + length < n and word[length] == word[i + length]:
                length += 1
            
            if i + length > r:
                c = i
                r = i + length
            z[i] = length
        
        i = 1
        while i * k < n:
            if z[i * k] == n - i * k:
                return i
            i += 1

        return (n + k - 1) // k
```
