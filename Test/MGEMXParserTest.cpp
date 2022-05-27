#include <initializer_list>
#include "AST.hpp"
#include "MGEMX.scanner.generated.hpp"
#include "MGEMX.parser.generated.hpp"
#include "AssetLoader.hpp"

namespace My {
    AssetLoader* g_pAssetLoader = new AssetLoader();

    std::map<std::string, ASTNodeRef> global_symbol_table;

    ASTNodeRef ast_root = make_ASTNodeRef<ASTNodeNone>( "ROOT" );
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
    ASTNodeRef primitive_type_nodes[8];
    global_symbol_table = {
        { "byte",   primitive_type_nodes[0] },
        { "short",  primitive_type_nodes[1] },
        { "ushort", primitive_type_nodes[2] },
        { "bool",   primitive_type_nodes[3] },
        { "int",    primitive_type_nodes[4] },
        { "uint",   primitive_type_nodes[5] },
        { "float",  primitive_type_nodes[6] },
        { "double", primitive_type_nodes[7] }
    };
    parse("Schema/RenderDefinitions.fbs");

    global_symbol_table = {
        { "byte",   primitive_type_nodes[0] },
        { "short",  primitive_type_nodes[1] },
        { "ushort", primitive_type_nodes[2] },
        { "bool",   primitive_type_nodes[3] },
        { "int",    primitive_type_nodes[4] },
        { "uint",   primitive_type_nodes[5] },
        { "float",  primitive_type_nodes[6] },
        { "double", primitive_type_nodes[7] }
    };
    parse("Schema/RenderDefinitions.fbs");

    std::cout << *ast_root << std::endl;

    return 0;
}
