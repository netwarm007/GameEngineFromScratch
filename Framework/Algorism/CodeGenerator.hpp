#pragma once
#include <iostream>
#include <sstream>
#include "AST.hpp"

namespace My {
    class CodeGenerator {
        public:
            enum class CODE_GENERATION_TYPE {
                GRAPHVIZ_DOT,
                CPP_SNIPPET
            };

            enum class GENERATION_STATUS {
                NONE,
                WAITING,
                GENERATING,
                GENERATED
            };

            void GenerateCode(std::ostream& out, const ASTNodeRef& ref, CODE_GENERATION_TYPE type);

            static GENERATION_STATUS GetGenerationStatus(ASTNode::IDN_TYPE idn);
            static void AppendGenerationSource(ASTNode::IDN_TYPE idn);
            static ASTNodeRef NextWaitingASTNode();
            static void ResetStatus();

        private:
            void generateGraphvizDot(std::ostream& out, const ASTNodeRef& ref);
            void generateCppSnippet(std::ostream& out, const ASTNodeRef& ref);

            void generateEnumCpp(std::ostream& out, const ASTNodeRef& ref);
            void generateStructCpp(std::ostream& out, const ASTNodeRef& ref);
            void generateTableCpp(std::ostream& out, const ASTNodeRef& ref);

        private:
            std::string nameSpace;
            bool genRelfectionCode  = true;
            bool genGuiBindCode     = true;

            static std::map<std::string, GENERATION_STATUS> task_list;
    };
}
