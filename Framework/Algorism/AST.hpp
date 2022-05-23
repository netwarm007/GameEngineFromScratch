#include "Tree.hpp"
#include <variant>

namespace My {
    template <typename T, class...Args>
    class ASTNode : public TreeNode {
        protected:
            T m_Value;

        private:
            ASTNode() = delete;
            void AppendChild(std::shared_ptr<TreeNode>&& sub_node) override {};

        public:
            explicit ASTNode(Args&&... args) : m_Value(std::forward<Args>(args)...) { m_Children.resize(2); }
            void SetLeft(std::shared_ptr<TreeNode> pNode) { m_Children.front() = pNode; }
            void SetRight(std::shared_ptr<TreeNode> pNode) { m_Children.back() = pNode; }
            [[nodiscard]] const std::shared_ptr<TreeNode> GetLeft() const {
                return m_Children.front();
            }
            [[nodiscard]] const std::shared_ptr<TreeNode> GetRight() const {
                return m_Children.back();
            }
            [[nodiscard]] T GetValue() const { return m_Value; }
        
        protected:
            void dump(std::ostream& out) const override { out << m_Value; }
    };
}