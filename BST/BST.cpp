#include <iostream>
using std::cin;
using std::cout;
template <typename K, typename V> struct Node {
  K key;                    // sorted by key
  V val;                    // associated data
  Node<K, V> *left, *right; // left and right substrees
  int N;                    // number of nodes in subtree
  Node(K _key, V _val, int _N) : key{_key}, val{_val}, left{nullptr}, right{nullptr}, N{_N} {}
};

template <typename K, typename V> class BST {
private:
  Node<K, V> *root;

public:
  BST() : root{nullptr} {}

  // Return true if this symbol table is empty.
  bool isEmpty() { return size() == 0; }

  // Return the number of key-value pairs in this symbol table.
  int size() { return size(root); }

  // Return number of key-value pairs in BST rooted at x
  int size(Node<K, V> *x) {
    if (x == nullptr) {
      return 0;
    } else {
      return x->N;
    }
  }

  // Does this symbol table contain the given key?
  bool contains(K key) { return get(key) != nullptr; }

  // Return the node associated with the given key
  Node<K, V> *get(K key) { return get(root, key); }

  // Insert the specified key-value pair into the symbol table, overwriting the
  // old value with the new value if the symbol table is already contains the
  // specified key.
  void put(K key, V val) { root = put(root, key, val); }

private:
  Node<K, V> *get(Node<K, V> *x, K key) {
    if (x == nullptr)
      return nullptr;
    if (key < x->key) {
      return get(x->left, key);
    } else if (key > x->key) {
      return get(x->right, key);
    } else {
      return x;
    }
  }

  Node<K, V> *put(Node<K, V> *x, K key, V val) {
    if (x == nullptr) {
      return new Node<K, V>{key, val, 1};
    }
    if (key < x->key) {
      x->left = put(x->left, key, val);
    } else if (key > x->key) {
      x->right = put(x->right, key, val);
    } else {
      x->val = val;
    }
    x->N = 1 + size(x->left) + size(x->right);
    return x;
  }
};

int main() {
  BST<char, int> bst;
  char c;
  int count = 0;
  while (cin >> c) {
    bst.put(c, count++);
  }
  return 0;
}
