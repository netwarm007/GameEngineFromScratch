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
    
    class ASTNode;

    using ASTNodeRef = std::shared_ptr<ASTNode>;

    class ASTNode : public TreeNode {
        protected:
            std::string m_Idn;
            AST_NODE_TYPE node_type;

        protected:
            ASTNode() { m_Children.resize(2); }
            void AppendChild(std::shared_ptr<TreeNode>&& sub_node) override { assert(false); };

        public:
            AST_NODE_TYPE GetNodeType() const { return node_type; }
            void SetLeft(ASTNodeRef pNode) { m_Children.front() = pNode; }
            void SetRight(ASTNodeRef pNode) { m_Children.back() = pNode; }


            [[nodiscard]] const ASTNodeRef GetLeft() const {
                return std::dynamic_pointer_cast<ASTNode>(m_Children.front());
            }
            [[nodiscard]] const ASTNodeRef GetRight() const {
                return std::dynamic_pointer_cast<ASTNode>(m_Children.back());
            }
    };

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
        public:
            using value_type = V;

        private:
            ASTNodeT() = delete;
            V m_Value;

        public:
            explicit ASTNodeT(const char* idn, Args&&... args) : m_Value(std::forward<Args>(args)...) { node_type = T; m_Idn = idn; }
            [[nodiscard]] V GetValue() const { return m_Value; }
        
        protected:
            void dump(std::ostream& out) const override { out << "IDN:\t" << m_Idn << std::endl << "Value:\t" << m_Value; }
    };

    template <AST_NODE_TYPE T>
    class ASTNodeT<T, void> : public ASTNode {
        public:
            using value_type = void;

        private:
            ASTNodeT() = delete;

        public:
            explicit ASTNodeT(const char* idn) { node_type = T; m_Idn = idn; }
        
        protected:
            void dump(std::ostream& out) const override { out << "IDN:\t" << m_Idn; }
    };

    template <class...Args>
    using ASTNodeNone =
            ASTNodeT<AST_NODE_TYPE::NONE,        void>;

    using ASTNodeNoneValueType = ASTNodeNone<>::value_type;

    template <class...Args>
    using ASTNodeNameSpace =
            ASTNodeT<AST_NODE_TYPE::NAMESPACE,   std::string,                                      Args...>;

    using ASTNodeNameSpaceValueType = ASTNodeNameSpace<>::value_type;

    template <class...Args>
    using ASTNodeEnum =
            ASTNodeT<AST_NODE_TYPE::ENUM,        ASTList<ASTPair<std::string, int32_t>>,           Args...>;

    using ASTNodeEnumValueType = ASTNodeEnum<>::value_type;

    template <class...Args>
    using ASTNodeStruct =
            ASTNodeT<AST_NODE_TYPE::STRUCT,      ASTList<ASTPair<std::string, std::string>>,       Args...>;

    using ASTNodeStructValueType = ASTNodeStruct<>::value_type;

    template <class...Args>
    using ASTNodeTable =
            ASTNodeT<AST_NODE_TYPE::TABLE,       ASTList<ASTPair<std::string, std::string>>,       Args...>;

    using ASTNodeTableValueType = ASTNodeTable<>::value_type;

    template <template<class...> class T, class...Args>
    ASTNodeRef make_ASTNodeRef(const char* idn, Args&&... args) {
        return std::make_shared<T<Args...>>(idn, std::forward<Args>(args)...);
    }
} // namespace My


