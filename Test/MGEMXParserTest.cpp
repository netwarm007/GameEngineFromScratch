#include <initializer_list>
#include "AST.hpp"
#include "MGEMX.scanner.generated.hpp"
#include "MGEMX.parser.generated.hpp"
#include "AssetLoader.hpp"

namespace My {
    AssetLoader* g_pAssetLoader = new AssetLoader();

    std::map<std::string, ASTNodeRef> global_symbol_table;

    ASTNodeRef ast_root = make_ASTNodeRef<ASTNodeNone>( "AST ROOT" );
}  // namespace My

using namespace My;

static void parse(const char* file) {
    yyscan_t scanner;
    auto fp = (FILE*) g_pAssetLoader->OpenFile(file, 
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
        { "double", make_ASTNodeRef<ASTNodePrimitive>( "double" ) }
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

    if (ref) {
        std::cout << "digraph ";
        std::cout << idn;
        std::cout << " {rankdir=LR node [shape=record]; Pipeline [label=\"" << std::endl;
        std::cout << *ref << std::endl;
        std::cout << "\"];}" << std::endl;
    }

    return 0;
}
