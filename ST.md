# ST表（RMQ）

预处理从初始化开始，第一层直接存储原数组，每一层的区间长度逐步翻倍，从 $2$ 开始，每次合并上一层中两个相邻区间的结果构造新层。

遍历范围用 $n - 2 \times i + 1$ 保证区间不越界，每次用聚合函数（如最小值、最大值或位运算）合并左区间和右区间的值。

最终，通过多层递归构建一个完整的 Sparse Table，时间复杂度为 $O(n \log n)$。

记住关键点：“第一层是原数组，区间翻倍递归合并，右端防越界，层层叠加完成表”。









> `st_[i][j]` 表示从下标 j 开始，长度为 2^i 的区间的聚合结果（比如最大值）
>
> **和常见的定义维度相反了！！是为了更好的写板子！**



> `pre_[j]` 表示从下标 j 开始，长度为 i 的聚合值
>
> `pre_[j + i]` 表示从下标 j + i 开始，长度为 i 的聚合值
>
> （因上一层定义的区间长度为i，这个pre相当于第二维）
>
> `func(pre_[j], pre_[j + i])` 就表示：从 j 开始，长度为 2 * i 的聚合值



> 一共可以构建 `n - 2*i + 1` 个，避免越界（比如要 [j, j+2i-1] 完整落在数组里）







```python
class SparseTable:
    def __init__(self, data: List, func=lambda x, y: x | y):
        """Initialize the Sparse Table with the given data and function."""
        self.func = func
        self.st_ = [list(data)]
        i, n = 1, len(self.st_[0])
        while 2 * i <= n:
            pre_ = self.st_[-1]
            self.st_.append([func(pre_[j], pre_[j + i]) for j in range(n - 2 * i + 1)])
            i <<= 1

    def query(self, begin: int, end: int) -> int:
        """Query the combined result over the interval [begin, end] using O(1)."""
        lg = (end - begin + 1).bit_length() - 1
        return self.func(self.st_[lg][begin], self.st_[lg][end - (1 << lg) + 1])
```




## [选数异或](https://www.acwing.com/problem/content/description/4648/)

难度：困难

标签：DP、ST、**异或**、**离线莫队**

题意和思路：

**题意**：

给定一个长度为 $n$ 的数组 $A = [A_1, A_2, \dots, A_n]$ 和一个非负整数 $x$，同时给定 $m$ 次查询。每次查询提供一个区间 $[l, r]$，需要判断在该区间内是否存在两个不同的数，使得它们的异或值等于 $x$。

### **思路：**

#### **思路 1：ST 表 + 预处理**

问题可以简化为：对于每个区间 $[l, r]$，判断是否存在一个数 $A[i]$ 和一个数 $A[j]$，其中 $A[j] = A[i] \oplus x$，且它们在这个区间内。

> 对于每个查询 [l,r]，判断是否存在两个数 A[i],A[j] 满足：A[i]⊕A[j]=k，**等价于**：A[j]=A[i]⊕k

1. **预处理：**

   - 我们需要快速查找区间 $[l, r]$ 内，满足 $A[j] = A[i] \oplus x$ 的两个数。

   - 构建一个辅助数组 $w[i]$，定义为：

     - $w[i]$ 表示 $A[i] \oplus x$ 最近一次出现的下标（如果不存在则为 0）

     - 也就是说，**$w[i]$** 是记录我们 **之前看到的 $A[j]$** 中，是否有满足：
       $$
       A[j] = A[i] \oplus x
       $$

   | 下标 $i$ | $A[i]$ | $A[i] \oplus x$ | 目标值 `target` | `last_occurrence`    | $w[i]$ |
   | -------- | ------ | --------------- | --------------- | -------------------- | ------ |
   | 1        | 1      | 0               | 0               | `{}`                 | 0      |
   | 2        | 2      | 3               | 3               | `{1: 1}`             | 0      |
   | 3        | 3      | 2               | 2               | `{1: 1, 2: 2}`       | 2      |
   | 4        | 2      | 3               | 3               | `{1: 1, 2: 2, 3: 3}` | 3      |
   | 5        | 1      | 0               | 0               | `{1: 1, 2: 4, 3: 3}` | 1      |

   第 3 个元素 `3`，它的异或 $3 \oplus 1 = 2$，而 `2` 在上一次出现的位置是 **下标 2**，所以 $w[3] = 2$。

   第 5 个元素 `1`，它的异或 $1 \oplus 1 = 0$，而 `0` 在上一次出现的位置是 **下标 1**，所以 $w[5] = 1$。

2. **构建 ST 表：**

   - ST 表支持 **区间最大值查询**，时间复杂度为 $O(\log n)$。
   - 对于每个查询 $[l, r]$，我们需要查找 $w[i]$ 的最大值，如果该最大值 **大于等于 $l$**，则表示存在满足条件的数对，否则不存在。
     - 如果 $w[i] \geq l$，说明当前 $A[i]$ 的异或目标值 $A[i] \oplus x$ **在区间 $[l, r]$ 内出现过**，且出现的位置为 $w[i]$，因此存在满足异或条件的两个不同元素。

3. **查询过程：**

   - 构建 ST 表的复杂度为 $O(n \log n)$。
   - 每次查询的复杂度为 $O(\log n)$。
   - 总复杂度：$O(n \log n + m \log n)$。



```python
def process_queries(n: int, m: int, x: int, arr: List[int], queries: List[List[int]]) -> List[str]:
    last_occ = {}
    w = [0] * n

    for i in range(n):
        target = arr[i] ^ x
        w[i] = last_occ.get(target, 0)
        last_occ[arr[i]] = i + 1

    st = SparseTable(w, max)

    for l, r in queries:
        max_w = st.query(l - 1, r - 1)
        if max_w >= l:
            print("yes")
        else:
            print("no")

if __name__ == "__main__":
    n, m, x = IO.read()
    arr = IO.read_list()
    queries = [IO.read_list() for _ in range(m)]
    process_queries(n, m, x, arr, queries)
```



------

#### **思路 2：离线莫队算法**

**为什么可以用莫队？**

- 莫队算法是一种离线处理区间查询的问题，通过排序查询，优化查询顺序，降低复杂度。
- 在本题中，每个查询可以视作一个区间异或问题，通过莫队算法按块排序，将查询按左端点块编号排序，再按右端点排序，优化查询顺序。

**实现细节：**

1. **离线排序：**
   - 按照左端点 $\frac{l}{\sqrt{n}}$ 排序，如果左端点所在块相同，则按右端点 $r$ 排序。
2. **数据结构：**
   - 使用哈希表维护当前区间内元素出现次数。
   - 当遍历到一个新元素 $A[i]$，检查 $A[i] \oplus x$ 是否已经出现过。如果出现过，说明存在满足条件的数对。
3. **时间复杂度：**
   - 时间复杂度为 $O((n + m) \sqrt{n})$，其中 $n$ 是数组长度，$m$ 是查询数量。

---

## [子数组按位与值为 K 的数目](https://leetcode.cn/problems/number-of-subarrays-with-and-value-of-k/description/)

难度：困难

标签：DP、ST、二分

题意和思路：

**题意**：

给定一个整数数组 $nums$ 和一个整数 $k$，需要返回满足条件的子数组个数，其中每个子数组中所有元素按位与（AND）的结果等于 $k$。

**思路：**

### ST表

利用稀疏表（Sparse Table）和二分查找的方法。稀疏表用于快速计算区间的按位与结果，二分查找用于确定满足按位与结果等于 $k$ 的子数组范围，从而优化查找过程。

1. **稀疏表构建：**
   - 初始化稀疏表 `st`，其中 `st[i][j]` 表示 **起点为 $i$、长度为 $2^j$ 的子数组的按位与结果**。
   - 构建时采用按位与操作：
   - 稀疏表支持 **$O(1)$ 时间复杂度** 的区间查询，用于快速获取区间 `[i, r]` 的 AND 结果。
2. **单调性转换：**

    - 按位与 (AND) 操作具有 **单调递减性**：
      - 随着子数组长度的增加，AND 结果只会 **保持不变或减小**，不会增大。
    - 为了利用 `bisect` 函数在单调递增序列上查找，我们将 **按位与结果取负数**，从而将原本 **单调递减序列转换为单调递增序列**
3. **二分查找过程：**

    - **固定起点 $i$**，在区间 `[i, n)` 中进行二分查找，查找 **符合 AND = $k$ 的子数组范围**：
      
      1. **左边界查找：**
      
         - 使用 `bisect_left` 在 **负数序列** 中查找 **第一个等于 `-k` 的位置**，即：
      
           ```python
           l = bisect_left(range(i, n), -k, key=lambda r: -st.query(i, r))
           ```
      2. **右边界查找：**
      
         - 使用 `bisect_right` 在 **负数序列** 中查找 **第一个大于 `-k` 的位置**，即：
      
           ```python
           r = bisect_right(range(i, n), -k, key=lambda r: -st.query(i, r))
           ```
      
    - **子数组个数计算：**
      - 子数组个数为 $r - l$，即 **从 `l` 到 `r-1` 的所有子数组** 均满足 AND = $k$。





```python
class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        st = Std.SparseTable(nums, func=lambda x, y: x & y)
        
        ans =  0
        n = len(nums)
        for i in range(n):
            l = bisect_left(range(i, n), -k, key=lambda r: -st.query(i, r))
            r = bisect_right(range(i, n), -k, key=lambda r: -st.query(i, r))
            ans += r - l
        
        return ans
```





****





### 为什么 log Trick 对“计数”行不通？

| 用途                               | 最短 / 最小差值                   | 计数                                                         |
| ---------------------------------- | --------------------------------- | ------------------------------------------------------------ |
| 关心的只是 **某个区间值** 是否更优 | ✔ 可以原地覆盖 (丢失原值也没问题) | ✘ 需要 **子数组 multiplicity**，一旦把 `nums[j]` 覆盖，就无法区分 “同一个 AND 值由多少个不同 j 产生” → 计数会被合并丢失 |
| 提前 `break` 不会漏答案            | ✔ 只要当前区间值不再变化即可      | ✘ 计数场景下，值虽然不变但 **仍可能来自多个更短起点**，提前退出就漏掉这些子数组 |

### 滑动窗口 + 哈希压缩转移（状态压缩）

#### **核心思想：**

- 我们**从左往右扫描数组**，每次处理当前元素 `nums[i]`。
- 维护一个哈希表 `cur`：它记录了**所有以 `i-1` 为结尾的子数组**的 AND 值及其出现次数。
- 对于当前的 `nums[i]`，我们做两件事：
  1. 把它自己视作一个新子数组 `[i]`，AND 值就是它本身。
  2. 把所有旧子数组扩展一位，把 `nums[i]` 加在后面，用 `val & nums[i]` 得到新 AND 值。

这保证了我们会 **不重不漏地枚举出所有子数组 `[j..i]`**。





```python
from collections import Counter

class Solution:
    def countSubarraysAND(self, nums: list[int], k: int) -> int:
        ans = 0
        cur = Counter()  # cur[val] 表示以 i-1 为结尾的子数组中 AND == val 的数量

        for x in nums:
            nxt = Counter()
            nxt[x] += 1  # 当前数自己作为新子数组 [i]
            for val, cnt in cur.items():
                nxt[val & x] += cnt  # 所有旧子数组 [j..i-1] 扩展到 [j..i]
            ans += nxt[k]  # 把所有 AND == k 的子数组加入答案
            cur = nxt      # 当前处理完，成为下一轮的“旧值”
        
        return ans
```





## [边界元素是最大值的子数组数目](https://leetcode.cn/problems/find-the-number-of-subarrays-where-boundary-elements-are-maximum/description/)

**难度：** 中等
**标签：** 值域分桶 / 下标分段、稀疏表 RMQ、单调栈

------

### 题意

给定正整数数组 `nums`（下标 `0 … n-1`）。
 统计有多少个子数组满足

1. 子数组的 **首元素** 与 **尾元素** 相同；
2. 且这个相同元素恰好是 **该子数组的最大值**。

返回所有满足条件的子数组个数。

------

### 解法一：下标分段 + 稀疏表 RMQ

| 步骤         | 解释                                                         |
| ------------ | ------------------------------------------------------------ |
| ① 预处理 RMQ | 用稀疏表 `Std.SparseTable(nums, max)` 支持 `O(1)` 求任意区间最大值 |
| ② 逐值建桶   | `index_list_dict[val]` 存放 **当前正在延伸的连续块** 中，`val` 的出现下标 |
| ③ 维护连续块 | 枚举下标 `i`： - 取该值最近一次出现位置 `pre`； - 若区间 `[pre, i]` 的最大值仍是 `val` (用 RMQ 判)，说明块未断，可继续收集； - 否则块被“更大值”切断，**统计上一块答案** `len*(len+1)/2`，清空后重新开始 |
| ④ 收尾       | 枚举完后，对每个桶剩余的下标块再统计一次                     |

> **为什么是 `len\*(len+1)/2`？**
>  对于一块合法下标序列 `p₀ < p₁ < … < p_{len-1}`，任意选 `p_s ≤ p_t` 作为首尾都能组成满足条件的子数组；可重复选同一个下标（`s=t`）表示长度为 1 的子数组，组合数即 `len·(len+1)/2`。

```python
class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        st = Std.SparseTable(nums, Math.max)       # O(1) RMQ
        buckets = defaultdict(list)                # val -> 当前块的下标
        ans = 0

        for i, v in enumerate(nums):
            bucket = buckets[v]
            if bucket:                             # v 在当前块已出现
                pre = bucket[-1]
                if st.query(pre, i) == v:          # 仍未被“大值”截断
                    bucket.append(i)
                else:                              # 被截断，结算旧块
                    n = len(bucket)
                    ans += n * (n + 1) // 2
                    bucket.clear()
                    bucket.append(i)
            else:                                  # 首次出现
                bucket.append(i)

        for bucket in buckets.values():            # 收尾
            n = len(bucket)
            ans += n * (n + 1) // 2
        return ans
```

- **时间** `O(n log n)`（建表） + `O(n)`（遍历）
- **空间** `O(n log n)`（稀疏表）

------

### 解法二：单调栈（官方题解做法）

把「首、尾都是最大值」转成在线计数问题。

#### 核心栈 invariant

- 维护 **单调不增** 栈 `st = [[val, cnt]]`
  - `val` = 栈元素值
  - `cnt` = 栈顶元素值在当前后缀出现的次数

当读到新数 `x` 时：

1. **弹栈**：`while x > st[-1][0]` → 说明栈顶比 `x` 小，被 `x`“截断”
2. **如果相等**：
   - 新增贡献 `cnt`（即以这些栈顶出现位置作为 “首” 的子数组数量）
   - `cnt++`（把 `x` 也归到同块里）
3. **如果更小**：`st.append([x,1])` 新开一块
4. **单元素子数组** 永远合法 → 初始答案 `ans = n`

```python
class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        INF = 10**18
        st = [[INF, 0]]          # 哨兵
        ans = len(nums)          # 所有长度 1 子数组
        for x in nums:
            while x > st[-1][0]:
                st.pop()
            if x == st[-1][0]:
                ans += st[-1][1]     # 首尾同为 x 的新子数组
                st[-1][1] += 1
            else:
                st.append([x, 1])
        return ans
```

- **时间** `O(n)`（每元素仅入/出栈一次）
- **空间** `O(n)`（最坏全递增）

------

### 代码注解：解法二的关键行

| 行号                     | 代码                                                      | 作用 |
| ------------------------ | --------------------------------------------------------- | ---- |
| `while x > st[-1][0]`    | 弹掉比 `x` 小的段，保证栈单调不增                         |      |
| `if x == st[-1][0]`      | `x` 可以与当前块合并                                      |      |
| `ans += st[-1][1]`       | 以 **块内任意旧位置作“首”**、`x` 作“尾”形成的新合法子数组 |      |
| `st[-1][1] += 1`         | 把 `x` 的出现次数合并到块里                               |      |
| `else: st.append([x,1])` | 新值更小，单独开一块                                      |      |
