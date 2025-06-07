

# BFS DFS 排列组合

## [拖拉机](https://www.acwing.com/problem/content/2021/)

难度：中等

标签：**双端队列广搜**、最短路

题意和思路：

边权只有 0 和 1 的最短路问题：适用于边权只有两种情况（0 和 1）的特殊图。双端队列广搜的核心思想：使用双端队列代替堆优化 Dijkstra 算法：当边权为 0 时，将新点加入队头。当边权为 1 时，将新点加入队尾。每个点出队时只处理一次，但可能多次入队以更新最小权值。

------

## [正方形数组的数目](https://www.acwing.com/problem/content/description/4522/)

难度：中等

标签：DFS、**组合**、**剪枝**

题意和思路：

在组合问题中，为避免重复解，针对相同数字需要排序后剪枝，保证相同数字按固定顺序使用，剪枝条件是

* “**当前数字与前一个相同，且前一个未被使用时跳过**”。

总计复杂度：O(n×n!)，如果 n=10：排列数量达到约 **360万**，大多数情况下会超时，如果良好剪枝了，就不会了。

前提：先得给原数组排序。

同题目[40. 组合总和 II](https://leetcode.cn/problems/combination-sum-ii/)（**跳过重复元素**：**同一层跳过相同数字**，防止重复子集）

这一行代码通常出现在 **组合类问题 (Combination)** 中，用于 **跳过重复元素，避免生成重复组合**。

```python
if j > i and nums[j] == nums[j - 1]: continue
```



比如[1, 1, 2, 2, 3]，获得了1,1,2了之后，return。现在的状态是获取了1,1，下标在i=3开始，但是i=3的2和前一个相等，所以continue。





### **为什么需要 `if i > 0 and A[i] == A[i - 1] and not used[i - 1]:`？**

`and not used[i - 1]` 是 **跳过重复元素的关键条件**，它的作用是：

1. **确保当前元素是重复元素：** `A[i] == A[i - 1]`
   - 如果当前元素和前一个元素相同，说明它们是 **重复元素**。
2. **确保当前元素不是第一个未使用的重复元素：** `not used[i - 1]`
   - 如果 `used[i - 1]` 为 `False`，说明 **前一个相同元素还没有被选中**，因此 **当前元素不能被选中**，否则会生成重复排列。





```python
import math

def count_square_permutations(A):
    # 预处理完全平方数
    max_sum = 2 * 10**9
    squares = set()
    i = 0
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1

    # 判断是否是完全平方数
    def is_square(num):
        return num in squares

    # DFS + 剪枝
    def dfs(path, used):
        if len(path) == len(A):  # 所有元素都已使用
            return 1
        
        count = 0
        for i in range(len(A)):
            # 跳过已使用的元素
            if used[i]:
                continue
            
            # 跳过重复元素
            if i > 0 and A[i] == A[i - 1] and not used[i - 1]:
                continue

            # 判断是否满足条件
            if not path or is_square(path[-1] + A[i]):
                used[i] = True
                path.append(A[i])
                count += dfs(path, used)  # 递归
                path.pop()
                used[i] = False
        return count

    # 对数组排序以便跳过重复
    A.sort()
    used = [False] * len(A)
    return dfs([], used)

# 输入示例
n = int(input())
A = list(map(int, input().split()))
print(count_square_permutations(A))

```





------

## [全排列 II](https://leetcode.cn/problems/permutations-ii/)

难度：中等

标签：DFS、**全排列**

题意和思路：

题意：给定一个可包含重复数字的序列 `nums` ，按任意顺序 返回所有不重复的全排列。

思路：

1. 和上面题差不多，关键在 `if vis[j] or (j > 0 and nums[j] == nums[j - 1] and not vis[j - 1]): continue`。
   - 如果 nums[j] 已填入排列，continue。
   - 如果 nums[j] 和前一个数 nums[j-1] 相等，且 nums[j-1] 没填入排列，continue。



```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        ans = []
        path = [0] * n
        vis = [0] * n
        nums.sort()

        def dfs(i: int) -> None:
            if i == n:
                ans.append(path.copy())
                return 
            
            for j in range(len(vis)):
                if vis[j] or (j > 0 and nums[j] == nums[j - 1] and not vis[j - 1]):
                    continue
                path[i] = nums[j]
                vis[j] = 1
                dfs(i + 1)
                vis[j] = 0
        
        dfs(0)
        return ans
```







1. 另一个思路，用集合来做

   **原代码：** 每次递归都 **完整遍历 `nums` 数组**，通过 `vis` 数组控制元素使用状态，并用 `continue` 跳过重复元素，依赖索引进行控制。

   **优化代码：** 每次递归 **只遍历唯一元素集合 `set(nums)`**，通过 `cnt` 计数器控制元素剩余使用次数，从而 **避免重复路径生成**，无需显式跳过重复元素，逻辑更简洁。

   ```python
   class Solution:
       def permuteUnique(self, nums: List[int]) -> List[List[int]]:
           cnt = Counter(nums)
           unique_nums = list(cnt.keys())
           n = len(nums)
           ans = []
           path = [0] * n
   
           def dfs(i: int) -> None:
               if i == n:
                   ans.append(path.copy())
                   return
               
               for num in unique_nums:
                   if cnt[num] > 0:
                       cnt[num] -= 1
                       path[i] = num
                       dfs(i + 1)
                       cnt[num] += 1
           dfs(0)
           return ans
   ```

   





****

## **排列 vs 组合 - DFS 模板详解**

在 **排列** 和 **组合** 问题中，DFS 的使用方法是有明显区别的。
它们在 **是否遍历全部元素** 和 **是否标记 `used` 数组** 方面的策略不同。





![image-20250408154221217](images\image-20250408154221217.png)



------

### **1. 排列 (Permutation)：**

#### **定义：**

- 排列是指 **顺序有关** 的组合，即 `[1, 2]` 和 `[2, 1]` 是两个不同的排列。

**典型排列问题：**

- 找出 `[1, 2, 3]` 的所有排列。

**DFS 模板：**

- **每次递归时，都会遍历所有元素。**
- **`used` 数组用于标记哪些元素已被使用。**

#### **DFS 代码模板：**

```python
def dfs_permutation(path, used, nums):
    # 终止条件：路径长度等于原数组长度
    if len(path) == len(nums):
        print(path)  # 输出排列
        return
    
    for i in range(len(nums)):
        # 跳过已使用的元素
        if used[i]:
            continue
        
        # 选择当前元素
        used[i] = True
        path.append(nums[i])
        dfs_permutation(path, used, nums)
        # 回溯
        path.pop()
        used[i] = False

def generate_permutations(nums):
    used = [False] * len(nums)
    dfs_permutation([], used, nums)

generate_permutations([1, 2, 3])
```

**输出：**

```
[1, 2, 3]
[1, 3, 2]
[2, 1, 3]
[2, 3, 1]
[3, 1, 2]
[3, 2, 1]
```

------

### **排列中的剪枝：**

当数组存在 **重复元素** 时，排列问题中就需要进行 **剪枝**。



核心目的是：**在同一层递归里，只允许“相同数值”的第一个未被使用的元素进入分支，从而消除重复解。**



#### 为什么能避免重复？

- **同一层（同一深度）下标 `i` 左边的元素都还没有被固定**。
  - 如果 `nums[i] == nums[i-1]` 且 `used[i-1]` 是 `False`，意味着前一个相同元素并没有出现在当前路径里。
  - 若此时你却拿 `nums[i]` 去递归，就会得到一条与“拿 `nums[i-1]` 再拿 `nums[i]`”互换顺序但数值完全一样的路径——这正是我们要去掉的重复。

#### **优化版：**

```python
def dfs_permutation(path, used, nums):
    if len(path) == len(nums):
        print(path)
        return
    
    for i in range(len(nums)):
        # 跳过已使用的元素
        if used[i]:
            continue
        
        # 跳过重复元素
        if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
            continue

        used[i] = True
        path.append(nums[i])
        dfs_permutation(path, used, nums)
        path.pop()
        used[i] = False

def generate_permutations(nums):
    nums.sort()  # 排序方便跳过重复元素
    used = [False] * len(nums)
    dfs_permutation([], used, nums)

generate_permutations([1, 1, 2])
```

**输出：**

```
[1, 1, 2]
[1, 2, 1]
[2, 1, 1]
```

------

### **2. 组合 (Combination)：**

#### **定义：**

- 组合是指 **顺序无关** 的子集，即 `[1, 2]` 和 `[2, 1]` 视为 **同一个组合**。

**典型组合问题：**

- 从 `[1, 2, 3]` 中选出 **两个元素的组合**。

**DFS 模板：**

- **每次递归时，起点是上一层的下一个元素**，即 `start`。
- **`used` 数组不需要使用**，因为每个元素 **只会被访问一次**。

#### **DFS 代码模板：**

```python
def dfs_combination(start, path, nums, k):
    # 终止条件：路径长度等于 k
    if len(path) == k:
        print(path)
        return
    
    for i in range(start, len(nums)):
        path.append(nums[i])
        dfs_combination(i + 1, path, nums, k)  # 下一个元素从 `i + 1` 开始
        path.pop()

def generate_combinations(nums, k):
    dfs_combination(0, [], nums, k)

generate_combinations([1, 2, 3], 2)
```

**输出：**

```
[1, 2]
[1, 3]
[2, 3]
```

------

### **组合中的剪枝：**

- 如果数组中有 **重复元素**，需要 **跳过重复组合**。

#### **优化版：**

```python
def dfs_combination(start, path, nums, k):
    if len(path) == k:
        print(path)
        return
    
    for i in range(start, len(nums)):
        # 跳过重复元素
        if i > start and nums[i] == nums[i - 1]:
            continue

        path.append(nums[i])
        dfs_combination(i + 1, path, nums, k)
        path.pop()

def generate_combinations(nums, k):
    nums.sort()
    dfs_combination(0, [], nums, k)

generate_combinations([1, 1, 2], 2)
```

**输出：**

```
[1, 1]
[1, 2]
[2, 2]
```

------

### **对比总结：**

|                 | **排列 (Permutation)**                                       | **组合 (Combination)**                              |
| --------------- | ------------------------------------------------------------ | --------------------------------------------------- |
| **顺序**        | 顺序不同视为不同                                             | 顺序无关                                            |
| **遍历方式**    | 每次递归 **从 0 遍历到 n**                                   | 每次递归 **从 `start` 开始**                        |
| **`used` 数组** | 需要使用，用于标记元素是否使用                               | 不需要                                              |
| **剪枝条件**    | `if used[i]: continue` + `if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]: continue` | `if i > start and nums[i] == nums[i - 1]: continue` |

------

### **如何判断当前题目用排列还是组合？**

1. **题目要求顺序相关？**
   - 如果 **顺序相关**（如 `1, 2` 和 `2, 1` 是不同解），那么 **用排列**，并且需要 `used` 数组。
2. **题目要求顺序无关？**
   - 如果 **顺序无关**（如 `[1, 2]` 和 `[2, 1]` 视为同一解），那么 **用组合**，并且 **不需要 `used` 数组**，只需要 `start` 标记起点。





## [公交路线](https://leetcode.cn/problems/bus-routes/description/?envType=daily-question&envId=2024-09-17)

**难度：** 困难
**标签：** BFS、图论、哈希表

------

### **题意**

- `routes[i]` 给出第 `i` 辆公交车循环经过的站点序列。
- 你从车站 `source` 出发（起始时不在任何车上），要到达 `target` 站点。
- **每次上车都算一次“换乘”**，求最少换乘次数；若到不了返回 `-1`。
  - 如果 `source == target`，答案是 `0`（无需上车）。

------

### **思路**

#### 1. 站点 → 公交车的倒排索引

先用哈希表 `stop_to_buses[stop] = [bus1, bus2, …]` 记录**每个站点能上哪些公交车**。

- 建表时间 = 所有 `routes` 元素数的总和 `O(∑len(route))`。

#### 2. BFS（最短换乘次数）

- **节点**：把 **“车站”** 当作 BFS 状态；
- **一步操作**：在当前站点 `x`
  1. 扫描所有经过 `x` 的公交车 `bus`；
  2. 坐上 `bus`，可一次性抵达它路线里的所有站 `y`；
  3. 这些站 `y` 的换乘距离 = `dis[x] + 1`（因为刚刚新上了一辆车）；
  4. 为避免重复，上过的 **公交车** 标记为已访问即可（`routes[bus] = None`）。
- 这样每条公交线路最多扫描一次，整体时间仍然 `O(∑len(route))`。
- 把“公交车”当作一次 **批量连边** 的中介：BFS 时上车→一次扩散到该车所有站点，并且给公交车打已访问标记，从而 `O(∑len(route))` 时间就能求出最少换乘次数。

#### 3. 终止条件

- BFS 过程中若弹出站点 == `target`，直接返回其距离；
- 队列空而未到达 → 返回 `-1`。

> **核心降复杂度技巧**：
>  与其让“公交车站 × 站”暴力连边，不如在 BFS 中 **一次性把整辆车的所有站扩散**，并给公交车打“已用”标记，避免同一路线被重复枚举。

------

### **代码**

```python
from typing import List
from collections import defaultdict, deque

class Solution:
    def numBusesToDestination(
        self, routes: List[List[int]], source: int, target: int
    ) -> int:
        # 倒排索引：站点 -> 能上的公交车编号列表
        stop_to_buses = defaultdict(list)
        for bus_id, route in enumerate(routes):
            for stop in route:
                stop_to_buses[stop].append(bus_id)

        if source == target:          # 原地就到
            return 0
        if source not in stop_to_buses or target not in stop_to_buses:
            return -1                 # 起点/终点无人车到达

        # BFS 初始化
        dist = {source: 0}            # 站点 -> 最少换乘次数
        q = deque([source])

        while q:
            cur_stop = q.popleft()
            cur_dis  = dist[cur_stop]

            # 遍历所有经过 cur_stop 的公交车
            for bus_id in stop_to_buses[cur_stop]:
                if routes[bus_id] is None:          # 该公交车已被遍历过
                    continue
                # 坐上这辆车，可到达其路线中的所有站
                for nxt_stop in routes[bus_id]:
                    if nxt_stop not in dist:        # 站点没访问过
                        dist[nxt_stop] = cur_dis + 1
                        if nxt_stop == target:      # 提前命中目标
                            return dist[nxt_stop]
                        q.append(nxt_stop)
                routes[bus_id] = None               # 标记整辆车已处理

        return -1                                   # 无法到达
```



