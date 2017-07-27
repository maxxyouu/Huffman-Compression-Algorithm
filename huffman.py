"""
Code for compressing and decompressing using Huffman compression.
"""

from nodes import HuffmanNode, ReadNode

# ====================
# Helper functions for manipulating bytes


def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str

    >>> byte_to_bits(14)
    '00001110'
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])


def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """
    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])


# ====================
# Functions for compression
def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency1

    @param bytes text: a bytes object
    @rtype: dict{int,int}

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    freq_dict = {}
    for byte in text:
        freq_dict[byte] = freq_dict.get(byte, 0) + 1
    return freq_dict


def custom_sort(freq_dict, rev=False):
    """sort the freq_dict by its value in ascending order and return
     a list of tuple

    note1: each tuple in the list represents: (int, frequency of value)
    note2: This is a helper function for huffman_tree and improve_tree

    @type freq_dict: dict{int: int}|list[tuple]
    @type rev: bool
            if true, sort the list in descending order
    @rtype: list[tuple]

    >>> custom_sort({2: 9, 6: 3}) == [(6, 3), (2, 9)]
    True
    >>> custom_sort([(2, 8), (3, 2)]) == [(3, 2), (2, 8)]
    True
    """
    if isinstance(freq_dict, dict):
        list_ = list(freq_dict.items())
    else:
        list_ = freq_dict
    return sorted(list_, key=lambda elements: elements[1], reverse=rev)


def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode

    >>> t = huffman_tree({2: 6, 3: 4})
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> t == result1 or t == result2
    True
    """
    # reference: https://en.wikipedia.org/wiki/Huffman_coding
    node_freq = [(HuffmanNode(s), f) for s, f in custom_sort(freq_dict)]
    while len(node_freq) > 1:
        n1 = node_freq.pop(0)
        n2 = node_freq.pop(0)
        node_freq.append((HuffmanNode(None, n1[0], n2[0]), n1[1] + n2[1]))
        node_freq = custom_sort(node_freq)  # sort the list again
    return node_freq[-1][0]  # return the root of the tree


def get_codes(tree):
    """ Return a dict mapping symbols from tree rooted at HuffmanNode to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {2: '1', 3: '0'}
    True
    """
    def _get_codes_helper(t, code=''):
        return ({sym: code for sym, code in
                 list(_get_codes_helper(t.left, code + '0').items()) +
                 list(_get_codes_helper(t.right, code + '1').items())}
                if not t.is_leaf() else {t.symbol: code})
    return _get_codes_helper(tree)


def number_nodes(tree):
    """ Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    counter = 0

    def _number_codes_helper(_tree):
        nonlocal counter
        if not _tree.is_leaf():
            _number_codes_helper(_tree.left)
            _number_codes_helper(_tree.right)
            _tree.number = counter
            counter += 1
    _number_codes_helper(tree)


def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    """
    total_bits, total_symbols = 0, 0
    dict_ = get_codes(tree)
    for key in freq_dict:
        total_bits += freq_dict[key] * len(dict_[key])
        total_symbols += freq_dict[key]
    return total_bits / total_symbols


def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mappings from symbols to codes
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    total_bits = ''.join([codes[symbol] for symbol in text])
    while len(total_bits) % 8 != 0:  # fill up the 0s
        total_bits += '0'
    return bytes([bits_to_byte(byte) for byte in
                  [total_bits[i:i + 8] for i in range(0, len(total_bits), 8)]])


def tree_to_bytes(tree):
    """ Return a bytes representation of the tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    """
    result = []
    def _tree_to_bytes_helper(tree, storage):
        if not tree.is_leaf():
            _tree_to_bytes_helper(tree.left, storage)
            _tree_to_bytes_helper(tree.right, storage)
            if tree.left.is_leaf():
                storage.extend([0, tree.left.symbol])
            else:
                storage.extend([1, tree.left.number])  # symbol is None
            if tree.right.is_leaf():
                storage.extend([0, tree.right.symbol])
            else:
                storage.extend([1, tree.right.number])  # symbol is None
    _tree_to_bytes_helper(tree, result)
    return bytes(result)


def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])


def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer that we want to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")


def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file whose contents we want to compress
    @param str out_file: output file, where we store our compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)
    tree = huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
              size_to_bytes(len(text)))
    result += generate_compressed(text, codes)

    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in the list.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)  # doctest: +NORMALIZE_WHITESPACE
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None),\
    HuffmanNode(12, None, None)), HuffmanNode(None, HuffmanNode(5, None, None),\
    HuffmanNode(7, None, None)))
    """
    root = HuffmanNode()
    def _generate_tree_general_helper(tree, nodes, node):
        if node.l_type == 0:
            tree.left = HuffmanNode(node.l_data)
        else:
            tree.left = HuffmanNode()
            _generate_tree_general_helper(tree.left, nodes, nodes[node.l_data])
        if node.r_type == 0:
            tree.right = HuffmanNode(node.r_data)
        else:
            tree.right = HuffmanNode()
            _generate_tree_general_helper(tree.right, nodes, nodes[node.r_data])
    _generate_tree_general_helper(root, node_lst, node_lst[root_index])
    return root


def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that the list represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2) # doctest: +NORMALIZE_WHITESPACE
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), \
    HuffmanNode(7, None, None)), \
    HuffmanNode(None, HuffmanNode(10, None, None), HuffmanNode(12, None, None)))
    """
    root = HuffmanNode()
    nodes = node_lst[:root_index]
    def _tree_postorder_helper(tree, nodes, node, guide):
        if node.r_type == 0:  # a leaf: display data
            tree.right = HuffmanNode(node.r_data)
        else:
            tree.right = HuffmanNode()
            index = guide.pop()
            _tree_postorder_helper(tree.right, nodes, nodes[index], guide)
        if node.l_type == 0:  # a leaf: display data
            tree.left = HuffmanNode(node.l_data)
        else:
            tree.left = HuffmanNode()
            index = guide.pop()
            _tree_postorder_helper(tree.left, nodes, nodes[index], guide)
    _tree_postorder_helper(root, nodes, node_lst[root_index],
                           [i for i in range(len(nodes))])
    return root


def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: how many bytes to decompress from text.
    @rtype: bytes

    >>> text = bytes([2, 3, 2])
    >>> freq = make_freq_dict(text)
    >>> tree = huffman_tree(freq)
    >>> codes = get_codes(tree)
    >>> compressed = generate_compressed(text, codes)
    >>> text == generate_uncompressed(tree, compressed, len(text))
    True
    """
    bits = ''.join([byte_to_bits(byte) for byte in text])  # a string of bits
    codes_symbol = {code: sym for sym, code in get_codes(tree).items()}
    bytes_, s, e = [], 0, 0
    while len(bytes_) != size:
        while not bits[s:e] in codes_symbol:
            e += 1
        bytes_.append(codes_symbol[bits[s:e]])
        s = e
    return bytes([int(x) for x in bytes_])


def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.

    @param bytes buf: a bytes object
    @rtype: list[ReadNode]

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i + 1]
        r_type = buf[i + 2]
        r_data = buf[i + 3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_size(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(generate_uncompressed(tree, text, size))


# ====================
# Other functions

def improve_tree(tree, freq_dict):
    """ Improve the tree as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to freq_dict.

    @param HuffmanNode tree: Huffman tree rooted at 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    >>> right = HuffmanNode(None, HuffmanNode(101), \
    HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    reference = custom_sort(freq_dict, rev=True)  # sort the dict in ascend
    operation_list = [tree]  # root node of the entire tree
    i = 0  # use this as a index
    # reference: course notes: the website under->March 3rd->general_tree_code
    # http://www.teach.cs.toronto.edu/~csc148h/winter/danny_lectures.html
    while len(operation_list) > 0:  # use level order traversal to swap value
        subtree = operation_list.pop(0)
        for child in [subtree.left, subtree.right]:
            if child:
                if child.is_leaf():  # if it is a leaf, swap value
                    child.symbol = reference[i][0]
                    i += 1
                else:  # otherwise, append the sub root to the list
                    operation_list.append(child)


if __name__ == "__main__":
    import python_ta

    python_ta.check_all(config="huffman_pyta.txt")
    import doctest
    doctest.testmod()
    import time

    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds."
              .format(fname, time.time() - start))
