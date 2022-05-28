#pragma once
#include <iostream>
#include "AST.hpp"

namespace My {
    class CodeGenerator {
        public:
            enum class CODE_GENERATION_TYPE {
                GRAPHVIZ_DOT,
                CPP_HEADER
            };

            void GenerateCode(std::ostream& out, const ASTNodeRef& ref, CODE_GENERATION_TYPE type);

        private:
            void generateGraphvizDot(std::ostream& out, const ASTNodeRef& ref);
            void generateCppSnippet(std::ostream& out, const ASTNodeRef& ref);

        private:
            bool genRelfectionCode  = false;
            bool genGuiBindCode     = false;
    };
}
