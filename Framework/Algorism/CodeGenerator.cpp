#include "CodeGenerator.hpp"

using namespace My;

// public
CodeGenerator::GENERATION_STATUS CodeGenerator::GetGenerationStatus(ASTNode::IDN_TYPE idn) {
    GENERATION_STATUS status = GENERATION_STATUS::NONE;

    auto it = task_list.find(idn);
    if (it != task_list.end()) {
        status = it->second;
    }

    return status;
}

void CodeGenerator::AppendGenerationSource(ASTNode::IDN_TYPE idn) {
    auto it = task_list.find(idn);
    if (it == task_list.end()) {
        task_list[idn] = GENERATION_STATUS::WAITING;
    }
}

ASTNodeRef CodeGenerator::NextWaitingASTNode() {
    // Now check the task list and recursively generate any missing files
    for (auto& task : task_list) {
        if (task.second == GENERATION_STATUS::WAITING) {
            return global_symbol_table[task.first];
        }
    }

    return nullptr;
}

void CodeGenerator::ResetStatus() {
    task_list.clear();
}

void CodeGenerator::GenerateCode(std::ostream& out, const char* name_space, const ASTNodeRef& ref, CODE_GENERATION_TYPE type) {
    assert(ref);
    auto idn = ref->GetIDN();

    nameSpace = name_space;

    auto status = GetGenerationStatus(idn);
    switch (status) {
    case GENERATION_STATUS::NONE:
    case GENERATION_STATUS::WAITING:
        task_list[idn] = GENERATION_STATUS::GENERATING;
        break;
    case GENERATION_STATUS::GENERATING:
    case GENERATION_STATUS::GENERATED:
        return;
    default:
        assert(0);
    }


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

    task_list[idn] = GENERATION_STATUS::GENERATED;
}

std::map<std::string, CodeGenerator::GENERATION_STATUS> CodeGenerator::task_list;
