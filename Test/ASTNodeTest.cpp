#include <iostream>
#include <string>
#include <memory>
#include <initializer_list>

#include "AST.hpp"

using namespace My;

int main(int argc, char* argv[])
{
    auto root   = make_ASTNodeRef<ASTNodeNone>( "module" );

    auto ns     = make_ASTNodeRef<ASTNodeNameSpace, const char*>( 
                                                        "namespace", "My" );

    auto enum_  = make_ASTNodeRef<ASTNodeEnum, ASTList<std::string>> ( 
        "Enum", 
        { "Apple", "Banana", "Cherry", "Donut", "Egg" }
    );

    auto struct_= make_ASTNodeRef<ASTNodeStruct, ASTList<ASTPair<std::string, std::string>>> ( 
        "Struct", 
        { {"field1", "int"}, {"field2", "double"}, {"field3", "string"} }
    );

    auto table = make_ASTNodeRef<ASTNodeTable, ASTList<ASTPair<std::string, std::string>>> ( 
        "Table",
        { {"field1", "uuid"}, {"field2", "Texture"}, {"field3", "Shader"} }
    );

    root->SetLeft(ns);
    ns->SetLeft(enum_);
    enum_->SetRight(struct_);
    struct_->SetRight(table);

    std::cout << *root << std::endl;

    return 0;
}
