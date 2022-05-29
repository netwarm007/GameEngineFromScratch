#include "CodeGenerator.hpp"

using namespace My;

class headers_stream    : public std::stringstream { using std::stringstream::stringstream; };
class values_stream     : public std::stringstream { using std::stringstream::stringstream; };
class reflect_stream    : public std::stringstream { using std::stringstream::stringstream; };
class function_stream   : public std::stringstream { using std::stringstream::stringstream; };

extern ASTNodeRef ast_root;

// statics
static int indent_val = 0;

static inline std::string indent() {
    return std::string(4 * indent_val, ' ');
}

static const std::map<ASTNode::IDN_TYPE, const char*> prilimitive_types = {
        { "byte",   "char"      },
        { "short",  "int16_t"   },
        { "ushort", "uint16_t"  },
        { "bool",   "bool"      },
        { "int",    "int32_t"   },
        { "uint",   "uint32_t"  },
        { "float",  "float"     },
        { "double", "double"    }
};

static inline std::ostream& operator<<(values_stream& s, const ASTFieldDecl& v)
{
    assert(v.second);
    auto type_idn = v.second->GetIDN();

    // convert prilimitive types
    auto it = prilimitive_types.find(type_idn);
    if (it != prilimitive_types.end()) {
        type_idn = it->second;
    }

    s << indent() << type_idn << '\t';
    s << v.first << ';' << std::endl;
    return s;
}

static const std::map<ASTNode::IDN_TYPE, const char*> imgui_commands = {
        { "byte",   "ImGui::InputScalar( \"%s\", ImGuiDataType_S8, &%s );"},
        { "short",  "ImGui::InputScalar( \"%s\", ImGuiDataType_S16, &%s );"},
        { "ushort", "ImGui::InputScalar( \"%s\", ImGuiDataType_U16, &%s );"},
        { "bool",   "ImGui::Checkbox( \"%s\", &%s );"},
        { "int",    "ImGui::InputScalar( \"%s\", ImGuiDataType_S32, &%s );"},
        { "uint",   "ImGui::InputScalar( \"%s\", ImGuiDataType_U32, &%s );"},
        { "float",  "ImGui::InputFloat( \"%s\", &%s );"},
        { "double", "ImGui::InputDouble( \"%s\", &%s );"}
};

static inline std::ostream& operator<<(reflect_stream& s, const ASTFieldDecl& v)
{
    assert(v.second);
    auto type_idn = v.second->GetIDN();
    switch(v.second->GetNodeType())
    {
    case AST_NODE_TYPE::NONE:
        break;
    case AST_NODE_TYPE::PRIMITIVE:
        {
            auto it = imgui_commands.find(type_idn);
            if (it != imgui_commands.end()) {
                char buf[128];
                snprintf(buf, 128, it->second, v.first.c_str(), v.first.c_str());
                s << indent() << buf;
            }
        }
        break;
    case AST_NODE_TYPE::ENUM:
        s << indent() << type_idn << "::Enum " << v.first << ';' << std::endl;
        s << indent() << "ImGui::Combo( \"" << v.first << "\", (int32_t*)&" << v.first << ", " << type_idn << "::s_value_names, " << type_idn << "::Count );" << std::endl;
        break;
    case AST_NODE_TYPE::NAMESPACE:
        break;
    case AST_NODE_TYPE::STRUCT:
        s << indent() << type_idn << '\t' << v.first << ';' << std::endl;
        s << indent() << "ImGui::Text(\"" << v.first << "\");" << std::endl;
        s << indent() << v.first << ".reflectMembers();" << std::endl;
        break;
    case AST_NODE_TYPE::TABLE:
        s << indent() << type_idn << '\t' << v.first << ';' << std::endl;
        s << indent() << "ImGui::Text(\"" << v.first << "\");" << std::endl;
        s << indent() << v.first << ".reflectMembers();" << std::endl;
        break;
    case AST_NODE_TYPE::ATTRIBUTE:
        break;
    case AST_NODE_TYPE::ROOTTYPE:
        break;
    default:
        assert(0);
    }
    return s;
}

static inline std::ostream& operator<<(headers_stream& s, const ASTFieldDecl& v)
{
    assert(v.second);
    auto type_idn = v.second->GetIDN();
    switch(v.second->GetNodeType())
    {
    case AST_NODE_TYPE::NONE:
        break;
    case AST_NODE_TYPE::PRIMITIVE:
        break;
    case AST_NODE_TYPE::ENUM:
    case AST_NODE_TYPE::STRUCT:
    case AST_NODE_TYPE::TABLE:
        s << "#include \"" << type_idn << ".hpp\"";
        CodeGenerator::AppendGenerationSource(type_idn);
        break;
    case AST_NODE_TYPE::NAMESPACE:
        break;
    case AST_NODE_TYPE::ATTRIBUTE:
        break;
    case AST_NODE_TYPE::ROOTTYPE:
        break;
    default:
        assert(0);
    }
    return s;
}

static inline std::ostream& operator<<(values_stream& s, const ASTFieldList& v)
{
    for (const auto& e : v) {
        s << e << std::endl;
    }
    return s;
}

static inline std::ostream& operator<<(reflect_stream& s, const ASTFieldList& v)
{
    for (const auto& e : v) {
        s << e << std::endl;
    }
    return s;
}

static inline std::ostream& operator<<(headers_stream& s, const ASTFieldList& v)
{
    for (const auto& e : v) {
        s << e << std::endl;
    }
    return s;
}

static inline std::ostream& operator<<(values_stream& s, const ASTEnumItemDecl& v)
{
    s << v.first << " = " << v.second;
    return s;
}

static inline std::ostream& operator<<(reflect_stream& s, const ASTEnumItemDecl& v)
{
    s << '\"' << v.first << '\"';
    return s;
}

static inline std::ostream& operator<<(values_stream& s, const ASTEnumItems& v)
{
    bool first = true;
    for (const auto& e : v) {
        if (first) { 
            first = false; 
        } else {
            s << "," << std::endl; 
        }
        s << indent(); 
        s << e;
    }
    s << std::endl;
    return s;
}

static inline std::ostream& operator<<(reflect_stream& s, const ASTEnumItems& v)
{
    bool first = true;
    for (const auto& e : v) {
        if (first) { 
            first = false; 
        } else {
            s << "," << std::endl; 
        }
        s << indent(); 
        s << e;
    }
    s << std::endl;
    return s;
}

// private
void CodeGenerator::generateEnumCpp(std::ostream& out, const ASTNodeRef& ref) {
    if (!nameSpace.empty()) {
        out << indent() << "namespace " << nameSpace << " {" << std::endl;
        indent_val++;
    }

    out << indent() << "namespace " << ref->GetIDN() << " {" << std::endl;

    indent_val++;

    values_stream vs (std::ios_base::out);
    vs << indent() << "int Count = ";
    vs << std::dynamic_pointer_cast<ASTNodeEnum<ASTEnumItems>>(ref)->GetValue().size() << ";" << std::endl;

    vs << indent() << "enum Enum { " << std::endl;
    indent_val++;
    vs << std::dynamic_pointer_cast<ASTNodeEnum<ASTEnumItems>>(ref)->GetValue();
    indent_val--;
    vs << indent() << "};" << std::endl;

    reflect_stream rs (std::ios_base::out);
    function_stream fs (std::ios_base::out);
    if (genRelfectionCode) {
        rs << indent() << "static const char* s_value_names[] = {" << std::endl;
        indent_val++;
        rs << std::dynamic_pointer_cast<ASTNodeEnum<ASTEnumItems>>(ref)->GetValue();
        indent_val--;
        rs << indent() << "};" << std::endl;

        fs << indent() << "static const char* ToString( Enum e ) {" << std::endl;
        indent_val++;
        fs << indent() << "return s_value_names[(int)e];" << std::endl;
        indent_val--;
        fs << indent() << "}" << std::endl;
    }

    out << vs.str();
    out << std::endl;
    out << rs.str();
    out << std::endl;
    out << fs.str();

    indent_val--;
    out << indent() << "} // namespace " << ref->GetIDN() << std::endl;

    if (!nameSpace.empty()) {
        indent_val--;
        out << indent() << "} // namespace " << nameSpace << std::endl;
    }
}

void CodeGenerator::generateStructCpp(std::ostream& out, const ASTNodeRef& ref) {
    headers_stream hs (std::ios_base::out);
    hs << std::dynamic_pointer_cast<ASTNodeStruct<ASTFieldList>>(ref)->GetValue();

    out << hs.str() << std::endl;

    if (!nameSpace.empty()) {
        out << indent() << "namespace " << nameSpace << " {" << std::endl;
        indent_val++;
    }

    out << indent() << "struct " << ref->GetIDN() << " {" << std::endl;

    indent_val++;

    values_stream vs (std::ios_base::out);
    vs << std::dynamic_pointer_cast<ASTNodeStruct<ASTFieldList>>(ref)->GetValue();

    reflect_stream rs (std::ios_base::out);
    if (genRelfectionCode) {
        rs << indent() << "void reflectMembers() {" << std::endl;
        indent_val++;
        rs << std::dynamic_pointer_cast<ASTNodeStruct<ASTFieldList>>(ref)->GetValue();
        indent_val--;
        rs << indent() << "}" << std::endl;
    }

    function_stream fs (std::ios_base::out);
    if (genGuiBindCode) {
        fs << indent() << "reflectUI() {" << std::endl;
        indent_val++;
        fs << indent() << "ImGui::Begin(\"" << ref->GetIDN() << "\");" << std::endl;
        fs << indent() << "reflectMembers();" << std::endl;
        fs << indent() << "ImGui::End();" << std::endl;
        indent_val--;
        fs << indent() << "}" << std::endl;
    }

    out << vs.str();
    out << std::endl;
    out << rs.str();
    out << std::endl;
    out << fs.str();

    indent_val--;
    out << indent() << "};" << std::endl;
 
    if (!nameSpace.empty()) {
        indent_val--;
        out << indent() << "} // namespace " << nameSpace << std::endl;
    }
}

void CodeGenerator::generateTableCpp(std::ostream& out, const ASTNodeRef& ref) {
    headers_stream hs (std::ios_base::out);
    hs << std::dynamic_pointer_cast<ASTNodeTable<ASTFieldList>>(ref)->GetValue();

    out << hs.str() << std::endl;

    if (!nameSpace.empty()) {
        out << indent() << "namespace " << nameSpace << " {" << std::endl;
        indent_val++;
    }

    out << indent() << "struct " << ref->GetIDN() << " {" << std::endl;

    indent_val++;

    values_stream vs (std::ios_base::out);
    vs << std::dynamic_pointer_cast<ASTNodeTable<ASTFieldList>>(ref)->GetValue();

    reflect_stream rs (std::ios_base::out);
    if (genRelfectionCode) {
        rs << indent() << "void reflectMembers() {" << std::endl;
        indent_val++;
        rs << std::dynamic_pointer_cast<ASTNodeTable<ASTFieldList>>(ref)->GetValue();
        indent_val--;
        rs << indent() << "}" << std::endl;
    }

    function_stream fs (std::ios_base::out);
    if (genGuiBindCode) {
        fs << indent() << "reflectUI() {" << std::endl;
        indent_val++;
        fs << indent() << "ImGui::Begin(\"" << ref->GetIDN() << "\");" << std::endl;
        fs << indent() << "reflectMembers();" << std::endl;
        fs << indent() << "ImGui::End();" << std::endl;
        indent_val--;
        fs << indent() << "}" << std::endl;
    }

    out << vs.str();
    out << std::endl;
    out << rs.str();
    out << std::endl;
    out << fs.str();

    indent_val--;
    out << indent() << "};" << std::endl;
 
    if (!nameSpace.empty()) {
        indent_val--;
        out << indent() << "} // namespace " << nameSpace << std::endl;
    }
}

void CodeGenerator::generateCppSnippet(std::ostream& out, const ASTNodeRef& ref) {
    indent_val = 0;

    switch(ref->GetNodeType()) {
    case AST_NODE_TYPE::NONE:
        break;
    case AST_NODE_TYPE::PRIMITIVE:
        break;
    case AST_NODE_TYPE::ENUM:
        generateEnumCpp(out, ref);
        break;
    case AST_NODE_TYPE::NAMESPACE:
        break;
    case AST_NODE_TYPE::STRUCT:
        generateStructCpp(out, ref);
        break;
    case AST_NODE_TYPE::TABLE:
        generateTableCpp(out ,ref);
        break;
    case AST_NODE_TYPE::ATTRIBUTE:
        break;
    case AST_NODE_TYPE::ROOTTYPE:
        break;
    default:
        assert(0);
    }
}
