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
public:
  BST() : root{nullptr} {}
  ~BST() {
    cleanup(root);
  }
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

  // Return the node contains the smallest key
  Node<K, V> *min() {
    if (isEmpty()) { return nullptr; }
    return min(root);
  }

  // Return the node contains the smallest key
  Node<K, V> *max() {
    if (isEmpty()) { return nullptr; }
    return max(root);
  }

  // Does this binary tree satisfy symmetric order?
  bool isBST() {
    return isBST(root, nullptr, nullptr);
  }

private:
  Node<K, V> *root;

  void cleanup(Node<K, V> *node) {
    if (node == nullptr) {
      return;
    }
    cleanup(node->left);
    cleanup(node->right);
    delete node;
  }

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

  Node<K, V> *min(Node<K, V> *x) {
    if (x->left == nullptr) { return x; }
    else { return min(x->left); }
  }

  Node<K, V> *max(Node<K, V> *x) {
    if (x->right == nullptr) { return x; }
    else { return min(x->right); }
  }

  bool isBST(Node<K, V> *x, Node<K, V> *node_min, Node<K, V> *node_max) {
    if (x == nullptr) { return true; }
    if (node_min != nullptr && x->key <= node_min->key) { return false; }
    if (node_max != nullptr && x->key >= node_max->key) { return false; }
    return isBST(x->left, node_min, x) && isBST(x->right, x, node_max);
  }
};

int main() {
  BST<char, int> bst;
  char c;
  int count = 0;
  while (cin >> c) {
    bst.put(c, count++);
  }
  cout << bst.isBST() << "\n";
  return 0;
}
