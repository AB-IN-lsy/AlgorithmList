# 数位 DP

## [统计特殊整数](https://leetcode.cn/problems/count-special-integers/description/)

**难度：** 中等
**标签：** 数位 DP、位运算、记忆化搜索

------

### **题意与思路：**

| 要点         | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| **特殊整数** | 十进制表示下，各数位互不相同的正整数。                       |
| **目标**     | 统计区间 **[1, n]** 内的特殊整数个数。                       |
| **核心套路** | **数位 DP**（Digit DP）：枚举“填哪一位”，同时用位掩码 `mask` 记录哪些数字已经出现过。 |

> **为什么用数位 DP？**
>
> - `n ≤ 10^9`，直接暴力到 10⁹ 会超时；
> - 每一位只有 10 种选择，而位掩码最多 2¹⁰=1024 种状态，DP 状态空间可控。

------

### 数位DP模板v1

#### 4 个经典数位 DP 维度

| 维度         | 解释                                                   | 在代码中的变量 |
| ------------ | ------------------------------------------------------ | -------------- |
| **i**        | 当前处理到的下标（高 → 低）                            | `i`            |
| **mask**     | 已用数字集合（位掩码）                                 | `mask`         |
| **is_limit** | 前缀是否已跟 `n` 对齐；若是，则当前位最大只能填 `s[i]` | `is_limit`     |
| **is_num**   | 前面是否已放过数字；若否，还在“前导空格”阶段           | `is_num`       |

------

#### 状态转移图示

```
        ┌── not is_num ──► 允许“跳过”该位（保持 mask）
        │
状态(i, mask, is_limit, is_num)
        │
        └── 枚举 d = down..up ─►
                 └─ 若 d 未在 mask 中出现：进入
                 状态(i+1,
                       mask | (1<<d),
                       is_limit && (d==up),
                       True)
```

- `down`: 如果已放过数字可为 0；否则必须 ≥ 1（避免前导 0）。
- `up`: 若 `is_limit` 为真则 `= s[i]`，否则 9。

终点：`i == len(s)`，只要 `is_num` 为真就算 **1 种方案**。

------

#### **代码讲解（逐段拆解）**

```python
class Solution:
    def countSpecialNumbers(self, n: int) -> int:
        s = str(n)                        # 化成字符串，方便逐位访问
```

`lru_cache` 缓存 4 元组状态，避免指数级重复计算。

```python
        @lru_cache(None)
        def dfs(i: int, mask: int, is_limit: bool, is_num: bool) -> int:
            if i == len(s):              # 所有位处理完
                return int(is_num)       # 若至少放过一个数字 → 合法方案 1，否则 0
```

1. **可选“跳过”当前位**（仍未放数字）
    仅在 `is_num == False` 才允许：

```python
            res = 0
            if not is_num:
                res = dfs(i + 1, mask, False, False)
```

1. **枚举要放的数字 `d`**：

```python
            up   = int(s[i]) if is_limit else 9
            down = 0 if is_num else 1    # 避免前导 0
            for d in range(down, up + 1):
                if (mask >> d) & 1:      # d 已经出现，跳过
                    continue
                res += dfs(
                    i + 1,
                    mask | (1 << d),     # 标记 d 已用
                    is_limit and d == up,# 仍贴边？
                    True                 # 已放数字
                )
            return res
```

入口参数：`dfs(0, 0, True, False)`

- 位索引从 0 开始（最高位），
- 初始掩码 0（没用过任何数字），
- `is_limit=True`（最高位必须 ≤ `n` 对应位），
- `is_num=False`（还没放数字）。

时间复杂度 **O( 位数 × 2¹⁰ × 10 ) ≈ O(30000)**，足够通过。





```python
class Solution:
    def countSpecialNumbers(self, n: int) -> int:
        # 将整数 n 转换为字符串表示，方便逐位处理。
        s = str(n)

        @lru_cache(None)  # 使用缓存机制来避免重复计算，提高效率。
        def dfs(i, mask, is_limit, is_num):
            """
            使用深度优先搜索（DFS）和动态规划计算特殊数字的数量。

            参数:
                i (int): 当前处理的位数（在字符串 s 中的索引）。
                mask (int): 位掩码，用于表示目前已经使用的数字。每个位代表一个数字，位被设置为1表示该数字已使用。集合和二进制的转换关系
                is_limit (bool): 表示前面填的数字是否都是 n 对应位上的，如果为 true，那么当前位至多为 int(s[i])，否则至多为 9
                is_num (bool): 表示当前是否已经形成了一个有效的数字（避免前导零）。

            返回值:
                int: 从当前状态开始的有效特殊数字的数量。
            """
            # 基础情况：如果已经处理完所有位。
            if i == len(s):
                # 如果已经形成了一个有效的数字（is_num 为 True），返回 1；否则返回 0。
                return int(is_num)
            
            # 初始化当前状态下的结果。
            res = 0
            
            # 如果还没有形成有效数字，可以选择跳过当前位。
            if not is_num:
                # 递归处理下一位，不形成数字。
                res = dfs(i + 1, mask, False, False)

            # 确定当前位的上限。
            # 如果 is_limit 为 True，当前位的数字不能超过 s[i]；
            # 否则，当前位可以是 0 到 9 之间的任意数字。
            up = int(s[i]) if is_limit else 9
            
            # 确定当前位的下限。
            # 如果已经形成了一个数字，当前位可以从 0 开始；
            # 否则，为了避免前导零，当前位只能从 1 开始。
            down = 0 if is_num else 1
            
            # 尝试当前位的所有可能数字。
            for d in range(down, up + 1):
                # 检查当前数字 d 是否已经使用过（即 mask 中相应的位是否已设置）。
                if not 1 << d & mask:
                    # 递归处理下一位，更新位掩码和限制条件。
                    res += dfs(
                        i + 1, 
                        mask | 1 << d,  # 将当前数字 d 加入位掩码。
                        is_limit and d == up,  # 如果 d 达到上限，更新 is_limit。
                        True  # 现在已经形成了一个有效数字。
                    )

            return res

        # 从第一位开始 DFS，位掩码为空，限制条件由 n 决定，尚未形成数字。
        return dfs(0, 0, True, False)
```





### 数位DP模板v2



> **这个题需要考虑前缀0，是因为如果不考虑的话，代码会把前缀0也算使用过的整数0，这样会少结果**





```python
dfs(i, mask, limit_low, limit_high, is_num)
```


含义是：

- i：当前填到哪一位
- mask：当前已经使用过的数字（用二进制表示）

- **limit_low：当前是否贴着下界**（只可能在 i==0 时 relevant）
- **limit_high：是否贴着上界 n**
- is_num：前面是否填过有效数字（前导 0 阶段为 False）



一定要记住定义！！

比如这里

#### 为什么是 `limit_low=True` 和 `limit_high=False`？

```python
res += dfs(i + 1, mask, True, False, False)
```

我们解释每个参数：

##### `limit_low=True`：

- 原因：你选择「不填当前这一位」，相当于继续保持贴着 low 的限制。
- 因为 low_s[i] 是 0，所以你“跳过”这一位的行为相当于默认继续贴着 low 的边界，合法。
- 而且要一定保证low_s[i] 是 0，这样才能选到前导0

##### `limit_high=False`：

- 原因：你一旦「不填」当前位，就没有办法限制这一位之后的数字必须小于等于 high 的当前位。

- 举例说明：

  假设 high = 345 → high_s = [3,4,5]，当前 i=0，你不填第 0 位，那么后续可以填任何 `[0~9]`，不再受高位控制。

  所以，**跳过一位后必然 break 掉 high 的限制**。

##### `is_num=False`：

- 表示你仍然处于“还没开始正式填数”的阶段。





```python
class Solution:
    def countSpecialNumbers(self, n: int) -> int:
        m = len(str(n))
        high_s = list(map(int, str(n)))
        low_s = list(map(int, str(1).zfill(m)))

        @cache
        def dfs(i: int, mask: int, limit_low: bool, limit_high: bool, is_num: bool) -> int:
            if i == m:
                return int(is_num)

            res = 0
            if not is_num and low_s[i] == 0:
                res += dfs(i + 1, mask, True, False, False)
            
            lo = low_s[i] if limit_low else 0
            hi = high_s[i] if limit_high else 9

            d0 = 1 - int(is_num) # 如果前面没填数字，至少得从1开始
            for d in range(max(lo, d0), hi + 1):
                if not (mask & 1 << d):
                    res += dfs(
                        i + 1,
                        mask | 1 << d,
                        limit_low and d == lo,
                        limit_high and d == hi,
                        True
                    )
            
            return res

        return dfs(0, 0, True, True, False)
```



## **统计区间内恰好包含 k 个数字 0 的整数个数**

**难度：** 中等（典型数位 DP）

**标签：** 数位 DP、记忆化搜索、计数问题

------

### **题意与思路：**

> **题目**
>  给定三个正整数 `low (≥ 1)`、`high` 和 `k`，统计闭区间 `[low, high]` 中 **十进制表示里** 恰好出现 `k` 个 **数字 0** 的整数个数。
>
> - “前导 0” 不计入数字位，也就不会算作一个 0。
> - `1 ≤ low ≤ high ≤ 10^9`（或更大都行，算法与范围无关）。

------

#### **解题核心：数位 DP**

1. **补齐长度，方便按位对齐**

   - 把 `high` 转成字符串再转成数位数组 `high_s`。

   - 用 `zfill(len(high_s))` 给 `low` 补前导 0，得到与 `high_s` 等长的 `low_s`。

     ```python
     n = len(str(high))  # 确定最高位长度（标准数位DP的入口）
     high_s = list(map(int, str(high)))
     low_s  = list(map(int, str(low).zfill(n))) # 统一位数
     ```

2. **状态设计**   `dfs(i, cnt0, is_num, limit_low, limit_high)`

   - `i`         ：当前处理到的位（0 — n-1，左→右）
   - `cnt0`   ：已统计到的“非前导 0” 数量
   - `is_num` ：前面是否已经出现过非 0 位（= True 表示已经正式开始计数）
   - `limit_low` / `limit_high`：当前是否仍与下界/上界贴边

3. **转移**   枚举当前位可以填的数字 `d ∈ [lo, hi]`：

   ```python
   lo = low_s[i]  if limit_low  else 0
   hi = high_s[i] if limit_high else 9
   for d in range(lo, hi + 1):
       next_is_num = is_num or d != 0
       next_cnt0   = cnt0 + (1 if d == 0 and next_is_num else 0)
       ...
   ```

4. **终止条件**   到达 `i == n` 时，若 `cnt0 == k` 且已真正填过数字 (`is_num == True`) 即计数 1。

   > 由于题干保证 `low ≥ 1`，区间里不会出现孤零零的数字 0，因此不必单独特判 “整体为 0” 的情况；如果以后需支持 `low = 0`，只要在终止态另加一行判断即可。

5. **记忆化**   `functools.cache`（或 `lru_cache(None)`) 记下五维状态，整体复杂度 `O(位数 × cnt0 × 2 × 2)`，最多几十万状态，轻松通过。

------

### **代码：**



重点！！



>        上下界的限定，请勿乱动！！
>
>        如果对数位还有其它约束，应当只在下面的 for 循环做限制，不应修改 lo 或 hi



```python
        res = 0
        lo = low_s[i]  if limit_low  else 0
        hi = high_s[i] if limit_high else 9
```







```python
from functools import cache

def count_exact_k_zeros(low: int, high: int, k: int) -> int:
    """统计区间 [low, high] 内十进制中恰好包含 k 个 0 的整数个数（low ≥ 1）。"""
    # 1. 数位拆分并补齐
    n = len(str(high))  # 确定最高位长度（标准数位DP的入口）
    high_s = list(map(int, str(high)))
    low_s  = list(map(int, str(low).zfill(n))) # 统一位数

    @cache
    def dfs(i: int, cnt0: int, is_num: bool, limit_low: bool, limit_high: bool) -> int:
        # 剪枝：0 已超标
        if cnt0 > k:
            return 0
        # 递归到底：看是否满足“填过数字 且 0 的数量刚好为 k”
        if i == n:
            return int(cnt0 == k and is_num)
		
        # 上下界的限定，请勿乱动！！
        # 如果对数位还有其它约束，应当只在下面的 for 循环做限制，不应修改 lo 或 hi
        res = 0
        lo = low_s[i]  if limit_low  else 0
        hi = high_s[i] if limit_high else 9

        for d in range(lo, hi + 1):
            next_is_num = is_num or d != 0 # 一旦你填了一个非 0（比如 d=1~9），就意味着「开始填有效数字了」，此后 is_num = True
            res += dfs(
                i + 1,
                cnt0 + (1 if d == 0 and next_is_num else 0),  # 前导 0 不计数，只有在你已经开始填数字，且当前这一位是 0 的时候，才去加 cnt0 + 1。
                next_is_num,
                limit_low  and d == lo,
                limit_high and d == hi,
            )
        return res

    return dfs(0, 0, False, True, True)
```

------

### **注意事项 / 易错点**

1. **前导 0 不计入**：必须用 `is_num` 标记「是否已经开始遇到非 0 位」。
2. **上下界贴边**：`limit_low`/`limit_high` 只有当前位等于界限数位时才继续保持为 `True`。
3. **`low` 补 0 对齐**：避免在 DFS 中额外维护位差。
4. **`low = 0` 特判**（本题不需要）：若区间可能含 0 且 `k == 1`，应在终止态允许 `is_num == False` 并把 0 计入答案。
5. **记忆化五维状态**：`i × cnt0 × is_num × limit_low × limit_high`，位数 ≤ 10，不会爆内存。



------

## **[数字 1 的个数](https://leetcode.cn/problems/number-of-digit-one/description/)**

**难度：** 简单

**标签：** 数位 DP、计数、高位分析

------

### **题意与思路：**

> **题目**
>  给定一个整数 `n`，统计所有 `0 ≤ x ≤ n` 的整数中，数字 `1` 出现的总次数。

------

#### 解法：数位 DP（Digit Dynamic Programming）

我们按**每一位**枚举填的数字，记录所有可能构成的数字中，`1` 出现的次数总和。

------

#### 状态设计：`dfs(i, cnt1, limit_low, limit_high)`

- `i`: 当前正在处理第几位（从左往右，第 `i` 位）
- `cnt1`: 到当前位为止，已经出现了多少个数字 `1`
- `limit_low`: 当前是否贴着下界（0 的每一位）
- `limit_high`: 当前是否贴着上界（n 的每一位）

------

#### 状态转移

枚举当前位可以填的数字 `d ∈ [lo, hi]`：

- 如果当前贴边，就不能超过 `n` 的第 `i` 位
- 每次填一个数字 `d`，如果 `d == 1`，就把 `cnt1 + 1` 传下去

------

#### 终止条件

当 `i == n`，说明所有位都填完了，**此时返回的是目前累计的 `cnt1` 数值**。

也就是说，这里不是判断“是否合法”，而是直接返回累计答案。

------

### 注意事项

| 细节          | 说明                                                         |
| ------------- | ------------------------------------------------------------ |
| 数位补齐      | 为了对齐 `low=0` 和 `high=n`，我们手动用 `zfill()` 把 0 补齐到与 n 等长 |
| 不需要 is_num | 因为统计的是“所有数字中出现的 1 的总数”，不需要处理前导 0 的排除逻辑 |
| 记忆化        | 使用 `@cache` 修饰器避免重复计算                             |





### 如果是二进制？

按二进制位做 Digit DP



> **把整个数一位位转成“二进制位”，然后枚举每一位是 0/1**



下面这个题是

#### 1 到 N 中有多少个数满足其二进制表示中恰好有 K 个 1



```python
n, k = IO.read()
high_s = list(map(int, bin(n)[2:]))
m = len(high_s)
low_s = [0] * m


@lru_cache(None)
def dfs(i: int, cnt: int, limit_low: bool, limit_high: bool) -> int:
    if cnt > k:
        return 0
    if i == m:
        return int(cnt == k)

    res = 0
    lo = low_s[i] if limit_low else 0
    hi = high_s[i] if limit_high else 1

    for d in range(lo, hi + 1):
        res += dfs(
            i + 1,
            cnt + d,
            limit_low and d == lo,
            limit_high and d == hi,
        )
    return res


ans = dfs(0, 0, True, True)
print(ans)
dfs.cache_clear()
```





------

### **代码：**

```python
from functools import cache

class Solution:
    def countDigitOne(self, n: int) -> int:  
        m = len(str(n))  # 先获取位数
        high_s = list(map(int, str(n)))
        low_s = list(map(int, str(0).zfill(m)))

        @cache
        def dfs(i: int, cnt1: int, limit_low: bool, limit_high: bool) -> int:
            if i == m:
                return cnt1  # 所有位都填完，返回当前累计的1的个数

            res = 0
            lo = low_s[i] if limit_low else 0
            hi = high_s[i] if limit_high else 9

            for d in range(lo, hi + 1):
                res += dfs(
                    i + 1,
                    cnt1 + (1 if d == 1 else 0),
                    limit_low and d == lo,
                    limit_high and d == hi
                )
            return res

        return dfs(0, 0, True, True)
```



------

## [神奇数]([16届蓝桥杯14天国特冲刺营 - 神奇数 - 蓝桥云课](https://www.lanqiao.cn/courses/51805/learning/?id=4072900&compatibility=false))

**难度：** 中等偏上

**标签：** 数位 DP、枚举优化、取模优化、状态压缩

------

### **题意与思路：**

#### 题目描述

给定两个整数 `l` 和 `r`，统计区间 `[l, r]` 中有多少个 **“神奇数字”**。

**神奇数字定义如下：**

- 是一个正整数；
- 它的十进制表示中最后一位不为 0；
- 并且这个数字中所有数位的和，**能被最后一位整除**。

------

#### 例子解释

- 例如 `132`：数位和为 1+3+2=6，最后一位是 2，6%2==0 ✔️
- 而 `131`：1+3+1=5，最后一位是 1，5%1==0 ✔️
- 而 `123`：1+2+3=6，最后一位是 3，6%3==0 ✔️
- `124`：1+2+4=7，最后一位 4，7%4 != 0 ❌

------

### 解题思路：数位 DP + 枚举最后一位

**挑战：** 判断 `(数位和 - 最后一位) % 最后一位 == 0`，貌似不能提前知道最后一位。

#### 做法优化

1. **枚举 `last ∈ [1, 9]`**：把“最后一位”单独拿出来固定，变成外层枚举；

2. 然后构造 DP，统计前 `m-1` 位所有和为 `sum_` 的合法前缀，最后拼上 `last`，满足：

   ```
   (sum_ + last - last) % last == 0
   ⟺ sum_ % last == 0
   ```

3. 由于 `last` 固定，因此只需维护 `sum % last`，状态范围很小。

#### 状态定义 `dfs(i, sum_mod, last, limit_low, limit_high)`

- `i`: 当前填到第几位
- `sum_`: 当前前缀和对 `last` 取模（压缩状态）
- `last`: 枚举的最后一位（固定）
- `limit_low`, `limit_high`: 是否贴着边界

#### 特殊注意点：

- **终止条件为 `i == m - 1`**，即只剩最后一位未填，此时判断 `last` 是否在允许范围 `[lo, hi]`，并检查 `sum_ % last == 0`；
- 不能忘记 **`limit_low` / `limit_high` 对最后一位也要判断**，否则会计算出超出下界 / 上界的数字，造成错误答案！

------

### 重点附加知识：为什么不用 `%2520`

数位和是否能被 `last ∈ [1, 9]` 整除，其本质是判断 `(sum % last == 0)`。当你不能提前确定 `last`，例如需要动态在递归中出现时，必须保存多个模值状态。

这时就要引入技巧：

预先对 `sum % LCM(1..9) = 2520` 取模，**能同时保留所有 `sum % d` 的信息**（压缩为一个状态值），常用于一些动态整除判断的题目。

**但在这题中**，我们枚举了 `last`，每轮只处理一个模，因此完全不需要 `%2520`。

------

### 最终代码

```python
from functools import lru_cache

MOD = 998244353

l, = IO.read()
r, = IO.read()

m = len(str(r))
high_s = list(map(int, str(r)))
low_s  = list(map(int, str(l).zfill(m)))  # 统一补齐位数

@lru_cache(None)
def dfs(i: int, sum_: int, last: int, limit_low: bool, limit_high: bool) -> int:
    if i == m - 1:  # 最后一位固定为 last，判断是否合法
        lo = low_s[i] if limit_low else 0
        hi = high_s[i] if limit_high else 9
        return int(lo <= last <= hi and sum_ == 0)

    res = 0
    lo = low_s[i] if limit_low else 0
    hi = high_s[i] if limit_high else 9

    for d in range(lo, hi + 1):
        res += dfs(
            i + 1,
            (sum_ + d) % last,
            last,
            limit_low and d == lo,
            limit_high and d == hi
        ) % MOD
    return res % MOD

total = 0
for last in range(1, 10):
    dfs.cache_clear()
    total += dfs(0, 0, last, True, True) % MOD
print(total % MOD)
```

