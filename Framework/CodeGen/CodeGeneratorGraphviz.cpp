#include "CodeGenerator.hpp"

using namespace My;

static void dump_node(std::ostream& out, const ASTNodeRef& ref);

template<typename T, typename U>
static std::ostream& operator<<(std::ostream& s, const ASTPair<T, U>& v) 
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
    s << '{' << std::get<0>(v);
    s << "|{";
    assert(std::get<1>(v));
    dump_node(s, std::get<1>(v));
    s << "}}";
    return s;
}

static inline std::ostream& operator<<(std::ostream& s, const ASTEnumItemDecl& v)
{
    s << '{' << v.first;
    s << "|{" << v.second;
    s << "}}";
    return s;
}

template<typename T>
static std::ostream& operator<<(std::ostream& s, const ASTList<T>& v)
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
        s << e;
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
        s << e;
    }
    s << "}";
    return s;
}

static void dump_node(std::ostream& out, const ASTNodeRef& ref) {
    out << ref->GetIDN();
    out << "|{Type|";
    switch(ref->GetNodeType()) {
    case AST_NODE_TYPE::NONE:
        out << "NONE}";
        break;
    case AST_NODE_TYPE::PRIMITIVE:
        out << "PRIMITIVE}";
        break;
    case AST_NODE_TYPE::ENUM:
        out << "ENUM}";
        out << "|{Value|";
        out << std::dynamic_pointer_cast<ASTNodeEnum<ASTEnumItems>>(ref)->GetValue();
        out << '}';
        break;
    case AST_NODE_TYPE::NAMESPACE:
        out << "NAMESPACE}";
        break;
    case AST_NODE_TYPE::STRUCT:
        out << "STRUCT}";
        out << "|{Value|";
        out << std::dynamic_pointer_cast<ASTNodeStruct<ASTFieldList>>(ref)->GetValue();
        out << '}';
        break;
    case AST_NODE_TYPE::TABLE:
        out << "TABLE}";
        out << "|{Value|";
        out << std::dynamic_pointer_cast<ASTNodeTable<ASTFieldList>>(ref)->GetValue();
        out << '}';
        break;
    case AST_NODE_TYPE::ATTRIBUTE:
        out << "ATTRIBUTE}";
        break;
    case AST_NODE_TYPE::ROOTTYPE:
        out << "ROOTTYPE}";
        break;
    default:
        assert(0);
    }
}

// private
void CodeGenerator::generateGraphvizDot(std::ostream& out, const ASTNodeRef& ref) {
    auto idn = ref->GetIDN();
    out << "digraph ";
    out << idn;
    out << " {rankdir=LR node [shape=record]; " << idn << " [label=\"";

    dump_node(out, ref);

    out << "\"];}";
}
