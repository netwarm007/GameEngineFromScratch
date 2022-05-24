#include <iostream>
#include <string>
#include <memory>
#include <initializer_list>

#include "AST.hpp"

using namespace My;

int main(int argc, char* argv[])
{
    auto root   = std::make_shared<ASTNodeNone>( "module" );

    auto ns     = std::make_shared<ASTNodeNameSpace<const char*>>( "namespace", "My" );

    ASTList<std::string> enum_items { "A", "B", "C", "D", "E" };
    auto enum_  = std::make_shared<ASTNodeEnum<typeof(enum_items)>> ( "Enum", std::move(enum_items) );

    ASTList<ASTPair<std::string, std::string>> field_list { {"field1","int"}, {"field2","double"}, {"field3","string"} };
    auto struct_= std::make_shared<ASTNodeStruct<typeof(field_list)>>( "Struct", std::move(field_list) );

    ASTList<ASTPair<std::string, std::string>> record_list { {"field1","uuid"}, {"field2","Texture"}, {"field3","Shader"} };
    auto table  = std::make_shared<ASTNodeTable<typeof(record_list)>>( "Table", std::move(record_list) );

    root->SetLeft(ns);
    ns->SetLeft(enum_);
    enum_->SetRight(struct_);
    struct_->SetRight(table);

    std::cout << *root << std::endl;

    return 0;
}
