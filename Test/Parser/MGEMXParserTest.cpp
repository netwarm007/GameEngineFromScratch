#include "AST.hpp"
#include "MGEMX.scanner.generated.hpp"
#include "MGEMX.parser.generated.hpp"
#include "AssetLoader.hpp"
#include "CodeGenerator.hpp"

namespace My {
    std::map<ASTNode::IDN_TYPE, ASTNodeRef> global_symbol_table;

    ASTNodeRef ast_root = make_ASTNodeRef<ASTNodeNone>( "AST ROOT" );
}  // namespace My

using namespace My;

static void parse(const char* file) {
    yyscan_t scanner;
    AssetLoader assetLoader;
    auto fp = (FILE*) assetLoader.OpenFile(file, 
                                            AssetLoader::AssetOpenMode::MY_OPEN_TEXT);
    yylex_init(&scanner);
    yyset_in(fp, scanner);
    My::MGEMXParser parser(scanner);
    std::cout.precision(10);
    parser.parse();
    yylex_destroy(scanner);
}

int main() {
    global_symbol_table = {
        { "byte",   make_ASTNodeRef<ASTNodePrimitive>( "byte" ) },
        { "short",  make_ASTNodeRef<ASTNodePrimitive>( "short" ) },
        { "ushort", make_ASTNodeRef<ASTNodePrimitive>( "ushort" ) },
        { "bool",   make_ASTNodeRef<ASTNodePrimitive>( "bool" ) },
        { "int",    make_ASTNodeRef<ASTNodePrimitive>( "int" ) },
        { "uint",   make_ASTNodeRef<ASTNodePrimitive>( "uint" ) },
        { "float",  make_ASTNodeRef<ASTNodePrimitive>( "float" ) },
        { "double", make_ASTNodeRef<ASTNodePrimitive>( "double" ) },
        { "Vector3f", make_ASTNodeRef<ASTNodePrimitive>( "Vector3f" ) }
    };

    parse("Schema/RenderDefinitions.fbs");

    bool                result;
    ASTNode::IDN_TYPE   idn;
    ASTNodeRef          ref;

    // find start point
    std::tie(result, idn) = findRootType();
    if(!result) {
        std::cerr << "no root type defined!" << std::endl;
        return -1;
    }

    // find the symbol and dump
    std::tie(result, ref) = findSymbol(idn);
    if(!result) {
        fprintf(stderr, "Can not find symbol with IDN{%s}!\n", idn.c_str());
        return -1;
    }

    std::cerr << std::string(3, '\n');
    std::cerr << "\x1b[7m解析完成，输出数据结构：\x1b[0m" << std::endl;

    CodeGenerator generator;
    generator.GenerateCode(std::cout, "My", ref, CodeGenerator::CODE_GENERATION_TYPE::GRAPHVIZ_DOT);

    return 0;
}
