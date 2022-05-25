#include <cassert>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>
#include "Tree.hpp"

namespace My {
    enum class AST_NODE_TYPE {
        NONE,
        ENUM, 
        NAMESPACE,
        STRUCT,
        TABLE
    };
    
    class ASTNode : public TreeNode {
        protected:
            std::string m_Idn;

        protected:
            ASTNode() { m_Children.resize(2); }
            void AppendChild(std::shared_ptr<TreeNode>&& sub_node) override { assert(false); };

        public:
            void SetLeft(std::shared_ptr<ASTNode> pNode) { m_Children.front() = pNode; }
            void SetRight(std::shared_ptr<ASTNode> pNode) { m_Children.back() = pNode; }

            [[nodiscard]] const std::shared_ptr<ASTNode> GetLeft() const {
                return std::dynamic_pointer_cast<ASTNode>(m_Children.front());
            }
            [[nodiscard]] const std::shared_ptr<ASTNode> GetRight() const {
                return std::dynamic_pointer_cast<ASTNode>(m_Children.back());
            }
    };

    using ASTNodeRef = ASTNode*;
    std::map<std::string, ASTNodeRef> global_symbol_table;

    template <typename T, typename V>
    using ASTPair  = std::pair<T, V>;

    template <typename T>
    using ASTList = std::vector<T>;

    template<typename T, typename U>
    std::ostream& operator<<(std::ostream& s, const ASTPair<T, U>& v) 
    {
        s.put('(');
        char comma[3] = {'\0', ' ', '\0'};
        s << comma << v.first;
        comma[0] = ',';
        s << comma << v.second;
        return s << ')';
    }

    template<typename T>
    std::ostream& operator<<(std::ostream& s, const ASTList<T>& v) 
    {
        s.put('[');
        char comma[3] = {'\0', ' ', '\0'};
        for (const auto& e : v) {
            s << comma << e;
            comma[0] = ',';
        }
        return s << ']';
    }

    template <AST_NODE_TYPE T, typename V, class...Args>
    class ASTNodeT : public ASTNode {
        private:
            V m_Value;
            AST_NODE_TYPE node_type = T;

        public:
            explicit ASTNodeT(const char* idn, Args&&... args) : m_Value(std::forward<Args>(args)...) { m_Idn = idn; }
            [[nodiscard]] V GetValue() const { return m_Value; }
        
        protected:
            void dump(std::ostream& out) const override { out << "IDN:\t" << m_Idn << std::endl << "Value:\t" << m_Value; }
    };

    template <AST_NODE_TYPE T>
    class ASTNodeT<T, void> : public ASTNode {
        private:
            AST_NODE_TYPE node_type = T;

        private:
            ASTNodeT() = delete;

        public:
            explicit ASTNodeT(const char* idn) { m_Idn = idn; }
        
        protected:
            void dump(std::ostream& out) const override { out << "IDN:\t" << m_Idn; }
    };

    using ASTNodeNone =
            ASTNodeT<AST_NODE_TYPE::NONE,        void>;
    template <class...Args>
    using ASTNodeNameSpace =
            ASTNodeT<AST_NODE_TYPE::NAMESPACE,   std::string,                                      Args...>;
    template <class...Args>
    using ASTNodeEnum =
            ASTNodeT<AST_NODE_TYPE::ENUM,        ASTList<std::string>,                             Args...>;
    template <class...Args>
    using ASTNodeStruct =
            ASTNodeT<AST_NODE_TYPE::STRUCT,      ASTList<ASTPair<std::string, std::string>>,       Args...>;
    template <class...Args>
    using ASTNodeTable =
            ASTNodeT<AST_NODE_TYPE::TABLE,       ASTList<ASTPair<std::string, std::string>>,       Args...>;

} // namespace My

