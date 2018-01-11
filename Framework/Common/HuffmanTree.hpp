#include <memory>
#include <vector>
#include "portable.hpp"

namespace My {
    template<typename T>
    class HuffmanNode {
        protected:
            T m_Value;
            std::shared_ptr<HuffmanNode<T>> m_pLeft;
            std::shared_ptr<HuffmanNode<T>> m_pRight;
            bool m_isLeaf = false;
        public:
            HuffmanNode() = default;
            HuffmanNode(T value) : m_Value(value), m_isLeaf(true) {};
            ~HuffmanNode() = default;
            HuffmanNode(HuffmanNode&) = default;
            HuffmanNode(HuffmanNode&&) = default;
            HuffmanNode& operator=(HuffmanNode&) = default;
            HuffmanNode& operator=(HuffmanNode&&) = default;
            bool IsLeaf(void) const { return m_isLeaf; };
            void SetLeft(std::shared_ptr<HuffmanNode> pNode) { m_pLeft = pNode; };
            void SetRight(std::shared_ptr<HuffmanNode> pNode) { m_pRight = pNode; };
            const std::shared_ptr<HuffmanNode<T>> GetLeft(void) const { return m_pLeft; };
            const std::shared_ptr<HuffmanNode<T>> GetRight(void) const { return m_pRight; };
            void SetValue(T value) { m_Value = value; m_isLeaf = true; };
            T GetValue() const { return m_Value; };
    };

    template<typename T>
    class HuffmanTree {
        protected:
            std::shared_ptr<HuffmanNode<T>> m_pRoot;

        private:
            void recursiveDump(const std::shared_ptr<HuffmanNode<T>>& pNode, std::string bit_stream)
            {
                if (pNode) {
                    if (pNode->IsLeaf()) {
                        printf("%20s | %x\n", bit_stream.c_str(), pNode->GetValue());
                    } else {
                        recursiveDump(pNode->GetLeft(), bit_stream + "0");
                        recursiveDump(pNode->GetRight(), bit_stream + "1");
                    }
                }
            }

        public:
            HuffmanTree() { m_pRoot = std::make_shared<HuffmanNode<uint8_t>>(); };

            size_t PopulateWithHuffmanTable(const uint8_t num_of_codes[16], const uint8_t* code_values)
            {
                int num_symbo = 0;
                for (int i = 0; i < 16; i++)
                {
                    num_symbo += num_of_codes[i];
                }
                const uint8_t* pCodeValueEnd = code_values + num_symbo - 1;

                std::queue<std::shared_ptr<HuffmanNode<uint8_t>>> node_queue;
                bool found_bottom = false;

                for (int i = 15; i >= 0; i--) {
                    int l = num_of_codes[i];
                    if (!found_bottom)
                    {
                        if (l == 0)
                        {
                            // simply move to upper layer
                            continue;
                        } else {
                            found_bottom = true;
                        }
                    }

                    auto childrenCount = node_queue.size();

                    if(l) {
                        // create leaf node for level i
                        pCodeValueEnd = pCodeValueEnd - l;
                            const uint8_t* pCodeValue = pCodeValueEnd + 1;

                        for (int j = 0; j < l; j++) {
                            auto pNode = std::make_shared<HuffmanNode<uint8_t>>(*pCodeValue++);
                            node_queue.push(pNode);
                        }
                    }

                    // create non-leaf node and append children
                    while (childrenCount > 0) {
                        auto pNode = std::make_shared<HuffmanNode<uint8_t>>();
                        auto pLeftNode = node_queue.front();
                        node_queue.pop();
                        pNode->SetLeft(pLeftNode);
                        childrenCount--;

                        if (childrenCount > 0)
                        {
                            auto pRightNode = node_queue.front();
                            node_queue.pop();
                            pNode->SetRight(pRightNode);
                            childrenCount--;
                        }

                        node_queue.push(pNode);
                    }
                }

                // now append to the root node
                assert(node_queue.size() <= 2 && node_queue.size() > 0);
                auto pLeftNode = node_queue.front();
                node_queue.pop();
                m_pRoot->SetLeft(pLeftNode);
                if (!node_queue.empty())
                {
                    auto pRightNode = node_queue.front();
                    node_queue.pop();
                    m_pRoot->SetRight(pRightNode);
                }

                return num_symbo;
            }

            T DecodeSingleValue(const uint8_t* encoded_stream, const size_t encoded_stream_length,
                    size_t* byte_offset, uint8_t* bit_offset)
            {
                T res = 0;
                std::shared_ptr<HuffmanNode<T>> pNode = m_pRoot;
                for (size_t i = *byte_offset; i < encoded_stream_length; i++)
                {
                    uint8_t data = encoded_stream[i];
                    for (int j = *bit_offset; j < 8; j++)
                    {
                        uint8_t bit = (data & (0x1 << (7 - j))) >> (7 - j);
                        if (bit == 0) { // left child
                            pNode = pNode->GetLeft();
                        } else {                     // right child
                            pNode = pNode->GetRight();
                        }

                        assert(pNode);

                        if (pNode->IsLeaf()) {
                            // move pointers to next bit
                            if (j == 7) {
                                *bit_offset = 0;
                                *byte_offset = i + 1;
                            } else {
                                *bit_offset = j + 1;
                                *byte_offset = i;
                            }
                            res = pNode->GetValue();
                            return res;
                        }
                    }
                    *bit_offset = 0;
                }

                // decode failed
                *byte_offset = -1;
                *bit_offset = -1;

                return res;
            }

            std::vector<T> Decode(const uint8_t* encoded_stream, const size_t encoded_stream_length)
            {
                std::vector<T> res;
                std::shared_ptr<HuffmanNode<T>> pNode = m_pRoot;
                for (size_t i = 0; i < encoded_stream_length; i++)
                {
                    uint8_t data = encoded_stream[i];
                    for (int j = 0; j < 8; j++)
                    {
                        uint8_t bit = (data & (0x1 << (7 - j))) >> (7 - j);
                        if (bit == 0) { // left child
                            pNode = pNode->GetLeft();
                        } else {                     // right child
                            pNode = pNode->GetRight();
                        }

                        assert(pNode);

                        if (pNode->IsLeaf()) {
                            res.push_back(pNode->GetValue());
                            pNode = m_pRoot;
                        }
                    }
                }

                return res;
            }

            void Dump()
            {
                std::string bit_stream;
                recursiveDump(m_pRoot, bit_stream);
                std::cout << std::endl;
            }
    };
}

