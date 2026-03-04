from tqdm.auto import tqdm

class Trie:
    """Trie类定义必须存在，以便pickle能够重建对象"""
    def __init__(self, sequences: list[list[int]] = None):
        self.trie = {}
        if sequences: 
            self.build_from_sequences(sequences)

    def add(self, seq: list[int]):
        node = self.trie
        for item in seq:
            if item not in node: 
                node[item] = {}
            node = node[item]

    def build_from_sequences(self, sequences: list[list[int]]):
        for seq in tqdm(sequences, desc="Constructing Trie object", leave=False): 
            self.add(seq)

    def merge(self, other_trie_dict: dict):
        for key, value in other_trie_dict.items():
            if key not in self.trie: 
                self.trie[key] = value
            else: 
                self._recursive_merge(self.trie[key], value)

    def _recursive_merge(self, node1, node2):
        for key, value in node2.items():
            if key not in node1: 
                node1[key] = value
            else: 
                self._recursive_merge(node1[key], value)

    def get(self, prefix: list[int]) -> list[int]:
        node = self.trie
        for item in prefix:
            if item not in node: 
                return []
            node = node[item]
        return list(node.keys())