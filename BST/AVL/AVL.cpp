#include <iostream>
#include <algorithm>
#include <vector>
using std::vector;
using std::cout;
struct Node {
  int key;
  Node *left;
  Node *right;
  int height;
  int size;
  int count; // for replicate nodes
  // New node is at the leaf
  Node(int _key): key{_key}, left{nullptr}, right{nullptr}, height{1}, size{1}, count{1} {}
};

class AVL {
public:
  AVL(): root{nullptr} {}

  ~AVL() {
    cleanup(root);
  }

  void insert(int key) { root = insert(root, key); }

  void preOrder() {
    preOrder(root);
    cout << "\n";
  }
private:
  Node *root;

  void cleanup(Node *node) {
    if (node == nullptr) {
      return;
    }
    cleanup(node->left);
    cleanup(node->right);
    delete node;
  }

  int getHeight(Node *node) {
    return node != nullptr ? node->height : 0;
  }

  int getSize(Node *node) {
    return node != nullptr ? node->size : 0;
  }

  int getBalance(Node *node) {
    return node != nullptr ? (getHeight(node->left) - getHeight(node->right)) : 0;
  }

  Node *rightRotate(Node *parent) {
    Node *child = parent->left;

    parent->left = child->right;
    child->right = parent;

    // Update height
    // Important: update old parent first since it is child now
    parent->height = 1 + std::max(getHeight(parent->left), getHeight(parent->right));
    child->height = 1 + std::max(getHeight(child->left), getHeight(child->right));

    // Update size
    // Important: update old parent first since it is child now
    parent->size = parent->count + getSize(parent->left) + getSize(parent->right);
    child->size = child->count + getSize(child->left) + getSize(child->right);

    // New root is child
    return child;
  }

  Node *leftRotate(Node *parent) {
    Node *child = parent->right;

    parent->right = child->left;
    child->left = parent;

    // Update height
    // Important: update old parent first since it is child now
    parent->height = 1 + std::max(getHeight(parent->left), getHeight(parent->right));
    child->height = 1 + std::max(getHeight(child->left), getHeight(child->right));

    // Update size
    // Important: update old parent first since it is child now
    parent->size = parent->count + getSize(parent->left) + getSize(parent->right);
    child->size = child->count + getSize(child->left) + getSize(child->right);

    // New root is child
    return child;
  }

  Node *insert(Node *node, int key) {
    // Normal insertion of binary search tree
    if (node == nullptr) {
      return new Node(key);
    }
    if (key < node->key) {
      node->left = insert(node->left, key);
    } else if (key > node->key) {
      node->right = insert(node->right, key);
    } else {
      ++node->count;
      ++node->size;
      return node;
    }

    // Update height
    node->height = std::max(getHeight(node->left), getHeight(node->right)) + 1;

    // Update size
    node->size = getSize(node->left) + getSize(node->right) + node->count;

    // Get current balance
    int balance = getBalance(node);

    // Left Left case
    if (balance > 1 && key < node->left->key) {
      return rightRotate(node);
    }

    // Left Right case
    if (balance > 1 && key > node->left->key) {
      node->left = leftRotate(node->left);
      return rightRotate(node);
    }

    // Right Right case
    if (balance < -1 && key > node->right->key) {
      return leftRotate(node);
    }

    // Right Left case
    if (balance < -1 && key < node->right->key) {
      node->right = rightRotate(node->right);
      return leftRotate(node);
    }

    return node;
  }

  // Preorder printing
  void preOrder(Node *node) {
    if (node != nullptr) {
      cout << node->key << " ";
      preOrder(node->left);
      preOrder(node->right);
    }
  }
};

int main() {
  vector<int> v {26,78,27,100,33,67,90,23,66,5,38,7,35,23,52,22,83,51,98,69,81,32,78,28,94,13,2,97,3,76,99,51,9,21,84,66,65,36,100,41};
  AVL avl;
  for (int i = v.size() - 1; i >=0; --i) {
    avl.insert(v[i]);
  }

  avl.preOrder();
  return 0;
}
