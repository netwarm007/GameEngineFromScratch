#include "CodeGenerator.hpp"

using namespace My;

class values_stream     : public std::stringstream { using std::stringstream::stringstream; };
class reflect_stream    : public std::stringstream { using std::stringstream::stringstream; };
class function_stream   : public std::stringstream { using std::stringstream::stringstream; };

// statics
static uint indent_val = 0;

static inline std::string indent() {
    return std::string(4 * indent_val, ' ');
}

static inline std::ostream& operator<<(values_stream& s, const ASTFieldDecl& v)
{
    assert(v.second);
    s << indent() << v.second->GetIDN() << '\t';
    s << v.first << ';' << std::endl;
    return s;
}

static const std::map<std::string, const char*> imgui_commands = {
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
    switch(v.second->GetNodeType())
    {
    case AST_NODE_TYPE::NONE:
        break;
    case AST_NODE_TYPE::PRIMITIVE:
        {
            auto it = imgui_commands.find(v.second->GetIDN());
            if (it != imgui_commands.end()) {
                char buf[128];
                snprintf(buf, 128, it->second, v.first.c_str(), v.first.c_str());
                s << indent() << buf;
            }
        }
        break;
    case AST_NODE_TYPE::ENUM:
        s << indent() << v.second->GetIDN() << "::Enum " << v.first << ';' << std::endl;
        s << indent() << "ImGui::Combo( \"" << v.first << "\", (int32_t*)&" << v.first << ", " << v.second->GetIDN() << "::s_value_names, " << v.second->GetIDN() << "::Count );" << std::endl;
        break;
    case AST_NODE_TYPE::NAMESPACE:
        break;
    case AST_NODE_TYPE::STRUCT:
        s << indent() << v.second->GetIDN() << '\t' << v.first << ';' << std::endl;
        s << indent() << "ImGui::Text(\"" << v.first << "\");" << std::endl;
        s << indent() << v.first << ".reflectMembers();" << std::endl;
        break;
    case AST_NODE_TYPE::TABLE:
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
    rs << indent() << "static const char* s_value_names[] = {" << std::endl;
    indent_val++;
    rs << std::dynamic_pointer_cast<ASTNodeEnum<ASTEnumItems>>(ref)->GetValue();
    indent_val--;
    rs << indent() << "};" << std::endl;

    function_stream fs (std::ios_base::out);
    fs << indent() << "static const char* ToString( Enum e ) {" << std::endl;
    indent_val++;
    fs << indent() << "return s_value_names[(int)e];" << std::endl;
    indent_val--;
    fs << indent() << "}" << std::endl;

    indent_val--;

    out << indent() << vs.str();
    out << std::endl;
    out << indent() << rs.str();
    out << std::endl;
    out << indent() << fs.str();
    out << "} // namespace " << ref->GetIDN() << std::endl;
}

void CodeGenerator::generateStructCpp(std::ostream& out, const ASTNodeRef& ref) {
    out << indent() << "struct " << ref->GetIDN() << " {" << std::endl;

    indent_val++;

    values_stream vs (std::ios_base::out);
    vs << std::dynamic_pointer_cast<ASTNodeStruct<ASTFieldList>>(ref)->GetValue();

    reflect_stream rs (std::ios_base::out);
    rs << indent() << "void reflectMembers() {" << std::endl;
    indent_val++;
    rs << std::dynamic_pointer_cast<ASTNodeStruct<ASTFieldList>>(ref)->GetValue();
    indent_val--;
    rs << indent() << "}" << std::endl;

    function_stream fs (std::ios_base::out);
    fs << indent() << "reflectUI() {" << std::endl;
    indent_val++;
    fs << indent() << "ImGui::Begin(" << ref->GetIDN() << "\");" << std::endl;
    fs << indent() << "reflectMembers();" << std::endl;
    fs << indent() << "ImGui::End();" << std::endl;
    indent_val--;
    fs << indent() << "}" << std::endl;

    indent_val--;

    out << indent() << vs.str();
    out << std::endl;
    out << indent() << rs.str();
    out << std::endl;
    out << indent() << fs.str();
    out << "};" << std::endl;
}

void CodeGenerator::generateTableCpp(std::ostream& out, const ASTNodeRef& ref) {
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
