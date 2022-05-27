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
        { "byte",   nullptr },
        { "short",  nullptr },
        { "ushort", nullptr },
        { "bool",   nullptr },
        { "int",    nullptr },
        { "uint",   nullptr },
        { "float",  nullptr },
        { "double", nullptr }
    };

    parse("Schema/RenderDefinitions.fbs");

    bool                result;
    ASTNode::IDN_TYPE   idn;
    ASTNodeRef          ref;

    std::tie(result, idn) = findRootType();
    if(!result) {
        std::cout << "no root type defined!" << std::endl;
        return -1;
    }

    std::tie(result, ref) = findSymbol(idn);
    if(!result) {
        printf("Can not find symbol with IDN{%s}!\n", idn.c_str());
        return -1;
    }

    std::cout << *ref << std::endl;

    return 0;
}
