#include <iostream>
#include <fstream>
#include "CodeGenerator.hpp"
#include "MGEMX.scanner.generated.hpp"
#include "MGEMX.parser.generated.hpp"
#include "AssetLoader.hpp"

using namespace My;

namespace My {
    std::map<ASTNode::IDN_TYPE, ASTNodeRef> global_symbol_table =
    {
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

    ASTNodeRef ast_root = make_ASTNodeRef<ASTNodeNone>( "AST ROOT" );
}  // namespace My

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

int main(int argc, char** argv) {
    if (argc > 2) {
        parse(argv[1]);
    } else {
        parse("Schema/RenderDefinitions.fbs");
    }

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

    // generate main data structure
    std::ofstream source(ref->GetIDN() + ".hpp");
    generator.GenerateCode(source, "My::RenderGraph", ref, CodeGenerator::CODE_GENERATION_TYPE::CPP_SNIPPET);
    source.close();

    // now check if any other dependencies and generate them
    auto next_ref = generator.NextWaitingASTNode();
    while (next_ref) {
        source.open(next_ref->GetIDN() + ".hpp");
        generator.GenerateCode(source, "My::RenderGraph", next_ref, CodeGenerator::CODE_GENERATION_TYPE::CPP_SNIPPET);
        source.close();
        next_ref = generator.NextWaitingASTNode();
    }
}