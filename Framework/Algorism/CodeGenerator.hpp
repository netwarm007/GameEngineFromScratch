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

            void GenerateCode(std::ostream& out, const ASTNodeRef& ref, CODE_GENERATION_TYPE type);

        private:
            void generateGraphvizDot(std::ostream& out, const ASTNodeRef& ref);
            void generateCppSnippet(std::ostream& out, const ASTNodeRef& ref);

            void generateEnumCpp(std::ostream& out, const ASTNodeRef& ref);
            void generateStructCpp(std::ostream& out, const ASTNodeRef& ref);
            void generateTableCpp(std::ostream& out, const ASTNodeRef& ref);

        private:
            std::string nameSpace;
            bool genRelfectionCode  = false;
            bool genGuiBindCode     = false;
    };
}
