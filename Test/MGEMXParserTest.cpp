#include "AssetLoader.hpp"
#include "MGEMX.scanner.generated.hpp"
#include "MGEMX.parser.generated.hpp"

namespace My {
AssetLoader* g_pAssetLoader = new AssetLoader();
}  // namespace My

using namespace My;

int main() {
    yyscan_t scanner;
    auto fp = (FILE*) g_pAssetLoader->OpenFile("Schema/RenderDefinitions.fbs", 
                                            AssetLoader::AssetOpenMode::MY_OPEN_TEXT);
    yylex_init(&scanner);
    yyset_in(fp, scanner);
    My::MGEMXParser parser(scanner);
    std::cout.precision(10);
    parser.parse();
    yylex_destroy(scanner);

    return 0;
}