# Trie

## 简要解释每个函数的作用

### `__init__`

初始化一个 Trie 节点，包含：

- `_children_`：子节点的字典（字符 -> TrieNode）
- `_cost`：该节点代表单词的最小花费（默认为 INF）
- `_is_end_of_word`：是否是一个完整单词的终点
- `_sid`：节点对应的字符串 ID，默认 -1，表示未分配

------

### `add(word: str, cost: int) -> int`

**向 Trie 中插入一个单词**，并设置它的最小 cost 和唯一 sid：

- 如果路径中节点不存在，则创建
- 每次插入时更新末尾节点的 `_cost`
- 若该末尾节点是第一次插入，则赋予一个自增的唯一 `_sid`

------

### `search(word: str) -> List[List]`

**查找字符串所有的前缀路径**，并返回这些路径对应的：

- 前缀长度
- 最小 cost
- 唯一 sid

例如对 `"apple"`：

- 返回形如 `[[1, ..., ...], [2, ..., ...], ..., [5, ..., ...]]`，每个元素代表前缀 `"a"`、`"ap"`、...、`"apple"` 对应的元信息

------

### `search_exact(word: str) -> int`

**精确查找某个完整单词**，如果是有效的词终点，返回其 `cost`，否则返回 `INF`。





```python
class TrieNode:
    """TrieNode class can quickly process string prefixes, a common feature used in applications like autocomplete and spell checking."""
    _sid_cnt = 0  # sid counter, representing string index starting from 0

    def __init__(self):
        """Initialize children dictionary and cost. The trie tree is a 26-ary tree."""
        self._children_ = {}
        self._cost = INF
        self._is_end_of_word = False  # Flag to indicate end of word
        self._sid = -1  # Unique ID for the node, -1 if not assigned

    def add(self, word: str, cost: int) -> int:
        """Add a word to the trie with the associated cost and return a unique ID."""
        node = self
        for c in word:
            if c not in node._children_:
                node._children_[c] = Std.TrieNode()
            node = node._children_[c]
        node._cost = Math.min(node._cost, cost)
        node._is_end_of_word = True  # Mark the end of the word
        if node._sid < 0:
            node._sid = self._sid_cnt
            self._sid_cnt += 1
        return node._sid

    def search(self, word: str) -> List[List]:
        """Search for prefixes of 'word' in the trie and return their lengths, costs, and sids.

        Collects ALL prefix lengths and their associated costs and sids!! 
        Valid matches are those where node.cost != INF and node.sid != -1.
        """
        node = self
        ans = []
        for i, c in enumerate(word):
            if c not in node._children_:
                break
            node = node._children_[c]
            ans.append([i + 1, node._cost, node._sid])  # i + 1 to denote length from start
        return ans

    def search_exact(self, word: str) -> int:
        """Search for the exact word in the trie and return its cost or unique ID."""
        node = self
        for c in word:
            if c not in node._children_:
                return INF
            node = node._children_[c]
        return node._cost if node._is_end_of_word else INF
```



## [转换字符串的最小成本 II](https://leetcode.cn/problems/minimum-cost-to-convert-string-ii/description/)

难度：困难

标签：Floyd、Trie、记忆化搜索

题意和思路：

**题意**：给定两个字符串 $source$ 和 $target$，它们长度相同且由小写字母组成，还有两个数组 $original$ 和 $changed$，以及一个整数数组 $cost$，其中 $cost[i]$ 表示将 $original[i]$ 替换为 $changed[i]$ 的费用。需要通过最小化操作费用将 $source$ 转换为 $target$，并满足以下条件：

1. 不同操作选择的子串不能在 $source$ 中重叠。
2. 相同位置的两次操作必须选择相同的子串。

如果转换不可行，则返回 $-1$。



**思路**：

1. **字符串转换的最短路径**: 使用 Floyd 算法优化字符串距离计算。通过将 $original[i]$ 到 $changed[i]$ 转换表示为图的边，计算所有字符串之间的最短距离。（Floyd优化 `if dis[i][k] == inf: continue`）
2. **字典树 (Trie)**: 建立 Trie 结构，用于将字符串转换为整数下标，便于在图中查询。Trie 还支持快速搜索字符串的所有前缀，以找到可替换子串的位置。需要额外判断前缀的合法性并记录每个子串的可能位置。
3. **记忆化搜索**: 从字符串的第 0 位开始递归处理，用 DFS 判断当前是否可以替换子串。如果可以替换，则递归处理下一个位置。若子串相同且符合转换条件，则对应最短路径为 $0$，也一并处理。





```python
class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        trie = Std.TrieNode()
        edges = []
        for u, v, w in zip(original, changed, cost):
            x, y = trie.add(u, 0), trie.add(v, 0)
            edges.append((x, y, w))

        floyd = Std.Floyd(trie.sid_cnt)
        for u, v, w in edges:
            floyd.add_edge(u, v, w)

        n = len(source)
        floyd.floyd()

        @lru_cache(None)
        def dfs(l: int):
            if l >= n:
                return 0
            res = INF
            if source[l] == target[l]:
                res = dfs(l + 1)

            for (len_, _, x), (_, _, y) in zip(trie.search(source[l:]), trie.search(target[l:])):
                if x != -1 and y != -1:
                    res = Math.min(res, floyd.get_dist(x, y) + dfs(l + len_))
            return res

        ans = dfs(0)
        return ans if ans != INF else -1
```

