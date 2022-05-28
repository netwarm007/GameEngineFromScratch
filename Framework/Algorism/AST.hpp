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
                out << "|{Type|";
                switch(node_type) {
                case AST_NODE_TYPE::NONE:
                    out << "AST_NODE_TYPE::NONE" << std::endl;
                    break;
                case AST_NODE_TYPE::PRIMITIVE:
                    out << "AST_NODE_TYPE::PRIMITIVE" << std::endl;
                    break;
                case AST_NODE_TYPE::ENUM:
                    out << "AST_NODE_TYPE::ENUM" << std::endl;
                    break;
                case AST_NODE_TYPE::NAMESPACE:
                    out << "AST_NODE_TYPE::NAMESPACE" << std::endl;
                    break;
                case AST_NODE_TYPE::STRUCT:
                    out << "AST_NODE_TYPE::STRUCT" << std::endl;
                    break;
                case AST_NODE_TYPE::TABLE:
                    out << "AST_NODE_TYPE::TABLE" << std::endl;
                    break;
                case AST_NODE_TYPE::ATTRIBUTE:
                    out << "AST_NODE_TYPE::ATTRIBUTE" << std::endl;
                    break;
                case AST_NODE_TYPE::ROOTTYPE:
                    out << "AST_NODE_TYPE::ROOTTYPE" << std::endl;
                    break;
                default:
                    assert(0);
                }

                out << "}";
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

    static inline std::ostream& operator<<(std::ostream& s, const ASTFieldDecl& v)
    {
        s << '{' << v.first << '\t';
        s << "|{";
        assert(v.second);
        s << *v.second << "}}";
        return s;
    }

    static inline std::ostream& operator<<(std::ostream& s, const ASTEnumItemDecl& v)
    {
        s << '{' << v.first << '\t';
        s << "|{" << v.second;
        s << "}}";
        return s;
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

    static inline std::ostream& operator<<(std::ostream& s, const ASTFieldList& v)
    {
        bool first = true;
        s << "{";
        for (const auto& e : v) {
            if (first) { 
                first = false; 
            } else {
                s << '|'; 
            }
            s << e << std::endl;
        }
        s << "}";
        return s;
    }

    static inline std::ostream& operator<<(std::ostream& s, const ASTEnumItems& v)
    {
        bool first = true;
        s << "{";
        for (const auto& e : v) {
            if (first) { 
                first = false; 
            } else {
                s << '|'; 
            }
            s << e << std::endl;
        }
        s << "}";
        return s;
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
            void dump(std::ostream& out) const override { 
                ASTNode::dump(out);
                out << "|{Value|" << m_Value << "}"; 
            }
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


