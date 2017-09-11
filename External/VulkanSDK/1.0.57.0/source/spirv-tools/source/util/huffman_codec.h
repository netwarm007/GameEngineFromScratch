// Copyright (c) 2017 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Contains utils for reading, writing and debug printing bit streams.

#ifndef LIBSPIRV_UTIL_HUFFMAN_CODEC_H_
#define LIBSPIRV_UTIL_HUFFMAN_CODEC_H_

#include <algorithm>
#include <cassert>
#include <functional>
#include <queue>
#include <iomanip>
#include <map>
#include <memory>
#include <ostream>
#include <sstream>
#include <stack>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace spvutils {

// Used to generate and apply a Huffman coding scheme.
// |Val| is the type of variable being encoded (for example a string or a
// literal).
template <class Val>
class HuffmanCodec {
  struct Node;

 public:
  // Creates Huffman codec from a histogramm.
  // Histogramm counts must not be zero.
  explicit HuffmanCodec(const std::map<Val, uint32_t>& hist) {
    if (hist.empty()) return;

    // Heuristic estimate.
    all_nodes_.reserve(3 * hist.size());

    // The queue is sorted in ascending order by weight (or by node id if
    // weights are equal).
    std::vector<Node*> queue_vector;
    queue_vector.reserve(hist.size());
    std::priority_queue<Node*, std::vector<Node*>,
        std::function<bool(const Node*, const Node*)>>
            queue(LeftIsBigger, std::move(queue_vector));

    // Put all leaves in the queue.
    for (const auto& pair : hist) {
      Node* node = CreateNode();
      node->val = pair.first;
      node->weight = pair.second;
      assert(node->weight);
      queue.push(node);
    }

    // Form the tree by combining two subtrees with the least weight,
    // and pushing the root of the new tree in the queue.
    while (true) {
      // We push a node at the end of each iteration, so the queue is never
      // supposed to be empty at this point, unless there are no leaves, but
      // that case was already handled.
      assert(!queue.empty());
      Node* right = queue.top();
      queue.pop();

      // If the queue is empty at this point, then the last node is
      // the root of the complete Huffman tree.
      if (queue.empty()) {
        root_ = right;
        break;
      }

      Node* left = queue.top();
      queue.pop();

      // Combine left and right into a new tree and push it into the queue.
      Node* parent = CreateNode();
      parent->weight = right->weight + left->weight;
      parent->left = left;
      parent->right = right;
      queue.push(parent);
    }

    // Traverse the tree and form encoding table.
    CreateEncodingTable();
  }

  // Prints the Huffman tree in the following format:
  // w------w------'x'
  //        w------'y'
  // Where w stands for the weight of the node.
  // Right tree branches appear above left branches. Taking the right path
  // adds 1 to the code, taking the left adds 0.
  void PrintTree(std::ostream& out) {
    PrintTreeInternal(out, root_, 0);
  }

  // Traverses the tree and prints the Huffman table: value, code
  // and optionally node weight for every leaf.
  void PrintTable(std::ostream& out, bool print_weights = true) {
    std::queue<std::pair<Node*, std::string>> queue;
    queue.emplace(root_, "");

    while (!queue.empty()) {
      const Node* node = queue.front().first;
      const std::string code = queue.front().second;
      queue.pop();
      if (!node->right && !node->left) {
        out << node->val;
        if (print_weights)
            out << " " << node->weight;
        out << " " << code << std::endl;
      } else {
        if (node->left)
          queue.emplace(node->left, code + "0");

        if (node->right)
          queue.emplace(node->right, code + "1");
      }
    }
  }

  // Returns the Huffman table. The table was built at at construction time,
  // this function just returns a const reference.
  const std::unordered_map<Val, std::pair<uint64_t, size_t>>&
      GetEncodingTable() const {
    return encoding_table_;
  }

  // Encodes |val| and stores its Huffman code in the lower |num_bits| of
  // |bits|. Returns false of |val| is not in the Huffman table.
  bool Encode(const Val& val, uint64_t* bits, size_t* num_bits) {
    auto it = encoding_table_.find(val);
    if (it == encoding_table_.end())
      return false;
    *bits = it->second.first;
    *num_bits = it->second.second;
    return true;
  }

  // Reads bits one-by-one using callback |read_bit| until a match is found.
  // Matching value is stored in |val|. Returns false if |read_bit| terminates
  // before a code was mathced.
  // |read_bit| has type bool func(bool* bit). When called, the next bit is
  // stored in |bit|. |read_bit| returns false if the stream terminates
  // prematurely.
  bool DecodeFromStream(const std::function<bool(bool*)>& read_bit, Val* val) {
    Node* node = root_;
    while (true) {
      assert(node);

      if (node->left == nullptr && node->right == nullptr) {
        *val = node->val;
        return true;
      }

      bool go_right;
      if (!read_bit(&go_right))
        return false;

      if (go_right)
        node = node->right;
      else
        node = node->left;
    }

    assert (0);
    return false;
  }

 private:
  // Huffman tree node.
  struct Node {
    Val val = Val();
    uint32_t weight = 0;
    // Ids are issued sequentially starting from 1. Ids are used as an ordering
    // tie-breaker, to make sure that the ordering (and resulting coding scheme)
    // are consistent accross multiple platforms.
    uint32_t id = 0;
    Node* left = nullptr;
    Node* right = nullptr;
  };

  // Returns true if |left| has bigger weight than |right|. Node ids are
  // used as tie-breaker.
  static bool LeftIsBigger(const Node* left, const Node* right) {
    if (left->weight == right->weight) {
      assert (left->id != right->id);
      return left->id > right->id;
    }
    return left->weight > right->weight;
  }

  // Prints subtree (helper function used by PrintTree).
  static void PrintTreeInternal(std::ostream& out, Node* node, size_t depth) {
    if (!node)
      return;

    const size_t kTextFieldWidth = 7;

    if (!node->right && !node->left) {
      out << node->val << std::endl;
    } else {
      if (node->right) {
        std::stringstream label;
        label << std::setfill('-') << std::left << std::setw(kTextFieldWidth)
              << node->right->weight;
        out << label.str();
        PrintTreeInternal(out, node->right, depth + 1);
      }

      if (node->left) {
        out << std::string(depth * kTextFieldWidth, ' ');
        std::stringstream label;
        label << std::setfill('-') << std::left << std::setw(kTextFieldWidth)
              << node->left->weight;
        out << label.str();
        PrintTreeInternal(out, node->left, depth + 1);
      }
    }
  }

  // Traverses the Huffman tree and saves paths to the leaves as bit
  // sequences to encoding_table_.
  void CreateEncodingTable() {
    struct Context {
      Context(Node* in_node, uint64_t in_bits, size_t in_depth)
          :  node(in_node), bits(in_bits), depth(in_depth) {}
      Node* node;
      // Huffman tree depth cannot exceed 64 as histogramm counts are expected
      // to be positive and limited by numeric_limits<uint32_t>::max().
      // For practical applications tree depth would be much smaller than 64.
      uint64_t bits;
      size_t depth;
    };

    std::queue<Context> queue;
    queue.emplace(root_, 0, 0);

    while (!queue.empty()) {
      const Context& context = queue.front();
      const Node* node = context.node;
      const uint64_t bits = context.bits;
      const size_t depth = context.depth;
      queue.pop();

      if (!node->right && !node->left) {
        auto insertion_result = encoding_table_.emplace(
            node->val, std::pair<uint64_t, size_t>(bits, depth));
        assert(insertion_result.second);
        (void)insertion_result;
      } else {
        if (node->left)
          queue.emplace(node->left, bits, depth + 1);

        if (node->right)
          queue.emplace(node->right, bits | (1ULL << depth), depth + 1);
      }
    }
  }

  // Creates new Huffman tree node and stores it in the deleter array.
  Node* CreateNode() {
    all_nodes_.emplace_back(new Node());
    all_nodes_.back()->id = next_node_id_++;
    return all_nodes_.back().get();
  }

  // Huffman tree root.
  Node* root_ = nullptr;

  // Huffman tree deleter.
  std::vector<std::unique_ptr<Node>> all_nodes_;

  // Encoding table value -> {bits, num_bits}.
  // Huffman codes are expected to never exceed 64 bit length (this is in fact
  // impossible if frequencies are stored as uint32_t).
  std::unordered_map<Val, std::pair<uint64_t, size_t>> encoding_table_;

  // Next node id issued by CreateNode();
  uint32_t next_node_id_ = 1;
};

}  // namespace spvutils

#endif  // LIBSPIRV_UTIL_HUFFMAN_CODEC_H_
