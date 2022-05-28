#pragma once
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
        PRIMITIVE,
        ENUM, 
        NAMESPACE,
        STRUCT,
        TABLE,
        ATTRIBUTE,
        ROOTTYPE
    };
    
    class ASTNode;

    using ASTNodeRef = std::shared_ptr<ASTNode>;

    using ASTAttr       = std::string;

    template <typename T, typename V>
    using ASTPair       = std::pair<T, V>;

    using ASTFieldDecl  = ASTPair<std::string, ASTNodeRef>;
    using ASTEnumItemDecl  = ASTPair<std::string, int64_t>;

    template <typename T>
    using ASTList       = std::vector<T>;

    using ASTFieldList  = ASTList<ASTFieldDecl>;
    using ASTEnumItems  = ASTList<ASTEnumItemDecl>;
    using ASTAttrList   = ASTList<ASTAttr>;

    class ASTNode : public TreeNode {
        public:
            using IDN_TYPE = std::string;

        protected:
            IDN_TYPE m_Idn;
            AST_NODE_TYPE node_type;

        protected:
            ASTNode() { m_Children.resize(2); }
            void AppendChild(std::shared_ptr<TreeNode>&& sub_node) override { assert(false); };

            void dump(std::ostream& out) const override { 
                out << m_Idn;
                out << "Type:\t";
                switch(node_type) {
                case AST_NODE_TYPE::NONE:
                    out << "NONE";
                    break;
                case AST_NODE_TYPE::PRIMITIVE:
                    out << "PRIMITIVE";
                    break;
                case AST_NODE_TYPE::ENUM:
                    out << "ENUM";
                    break;
                case AST_NODE_TYPE::NAMESPACE:
                    out << "NAMESPACE";
                    break;
                case AST_NODE_TYPE::STRUCT:
                    out << "STRUCT";
                    break;
                case AST_NODE_TYPE::TABLE:
                    out << "TABLE";
                    break;
                case AST_NODE_TYPE::ATTRIBUTE:
                    out << "ATTRIBUTE";
                    break;
                case AST_NODE_TYPE::ROOTTYPE:
                    out << "ROOTTYPE";
                    break;
                default:
                    assert(0);
                }

                out << std::endl;
            }

        public:
            IDN_TYPE      GetIDN()      const { return m_Idn; }
            AST_NODE_TYPE GetNodeType() const { return node_type; }
            void SetLeft(ASTNodeRef pNode) { m_Children.front() = pNode; }
            void SetRight(ASTNodeRef pNode) { m_Children.back() = pNode; }


            [[nodiscard]] ASTNodeRef GetLeft() const {
                return std::dynamic_pointer_cast<ASTNode>(m_Children.front());
            }
            [[nodiscard]] ASTNodeRef GetRight() const {
                return std::dynamic_pointer_cast<ASTNode>(m_Children.back());
            }
    };


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
    };

    template <AST_NODE_TYPE T>
    class ASTNodeT<T, void> : public ASTNode {
        public:
            using value_type = void;

        private:
            ASTNodeT() = delete;

        public:
            explicit ASTNodeT(const char* idn) { node_type = T; m_Idn = idn; }
    };

    template <class...Args>
    using ASTNodeNone =
            ASTNodeT<AST_NODE_TYPE::NONE,        void>;

    template <class...Args>
    using ASTNodePrimitive =
            ASTNodeT<AST_NODE_TYPE::PRIMITIVE,   void>;

    template <class...Args>
    using ASTNodeNameSpace =
            ASTNodeT<AST_NODE_TYPE::NAMESPACE,   std::string,           Args...>;

    template <class...Args>
    using ASTNodeEnum =
            ASTNodeT<AST_NODE_TYPE::ENUM,        ASTEnumItems,          Args...>;

    template <class...Args>
    using ASTNodeStruct =
            ASTNodeT<AST_NODE_TYPE::STRUCT,      ASTFieldList,          Args...>;

    template <class...Args>
    using ASTNodeTable =
            ASTNodeT<AST_NODE_TYPE::TABLE,       ASTFieldList,          Args...>;

    template <class...Args>
    using ASTNodeAttribute =
            ASTNodeT<AST_NODE_TYPE::ATTRIBUTE,        void>;

    template <class...Args>
    using ASTNodeRootType =
            ASTNodeT<AST_NODE_TYPE::ROOTTYPE,        void>;


    // Factory
    template <template<class...> class T, class...Args>
    ASTNodeRef make_ASTNodeRef(const char* idn, Args&&... args) {
        return std::make_shared<T<Args...>>(idn, std::forward<Args>(args)...);
    }

    // Utility Functions
    extern std::map<std::string, ASTNodeRef> global_symbol_table;

    static inline std::pair<bool, ASTNode::IDN_TYPE> findRootType() {
        auto it = global_symbol_table.find(static_cast<ASTNode::IDN_TYPE>("[root type]"));
        if (it != global_symbol_table.end()) {
            assert(it->second);
            return std::make_pair(true, it->second->GetIDN());
        }
        return std::make_pair(false, nullptr);
    }

    static inline std::pair<bool, ASTNodeRef> findSymbol(ASTNode::IDN_TYPE idn) {
        auto it = global_symbol_table.find(idn);
        if (it != global_symbol_table.end()) {
            return std::make_pair(true, it->second);
        }
        return std::make_pair(false, nullptr);
    }

} // namespace My


