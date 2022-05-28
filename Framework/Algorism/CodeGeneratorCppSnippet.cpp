#include "CodeGenerator.hpp"

using namespace My;

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

// private
void CodeGenerator::generateCppSnippet(std::ostream& out, const ASTNodeRef& ref) {
    auto idn = ref->GetIDN();
    out << "digraph ";
    out << idn;
    out << " {rankdir=LR node [shape=record]; " << idn << " [label=\"" << std::endl;
    out << *ref;
    out << "\"];}" << std::endl;
}
