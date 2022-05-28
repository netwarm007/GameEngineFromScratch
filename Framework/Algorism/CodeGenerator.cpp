#include "CodeGenerator.hpp"

using namespace My;

// public
void CodeGenerator::GenerateCode(std::ostream& out, const ASTNodeRef& ref, CODE_GENERATION_TYPE type) {
    assert(ref);
    if (ref->GetNodeType() == AST_NODE_TYPE::NAMESPACE)
        nameSpace = ref->GetIDN();

    switch (type) {
        case CODE_GENERATION_TYPE::GRAPHVIZ_DOT:
            generateGraphvizDot(out, ref);
            break;
        case CODE_GENERATION_TYPE::CPP_SNIPPET:
            generateCppSnippet(out, ref);
            break;
        default:
            assert(0);
    }
}
