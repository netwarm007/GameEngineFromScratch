#include <iostream>
#include <string>
#include <memory>

#include "AST.hpp"

using namespace My;

int main(int argc, char* argv[])
{
    // next statement equivalent to: std::shared_ptr<ASTNode> root = std::static_pointer_cast<ASTNode>(std::make_shared<ASTNodeT<AST_NODE_TYPE::NONE, void>, const char*>( "module"));
    auto root   = make_ASTNodeRef<ASTNodeNone>( "MODULE" );

    auto ns     = make_ASTNodeRef<ASTNodeNameSpace, const char*>( 
                                                        "My", "https://www.chenwenli.com" );

    auto enum_  = make_ASTNodeRef<ASTNodeEnum, ASTEnumItems> ( 
        "Foods", 
        { {"Apple", 1}, {"Banana", 2}, {"Cherry", 3}, {"Donut", 4}, {"Egg", 5} }
    );

    /* next statement equivalent to:
    std::shared_ptr<ASTNode> enum_ = std::static_pointer_cast<ASTNode>(std::make_shared<ASTNodeT<AST_NODE_TYPE::ENUM, std::vector<std::string>, std::vector<std::string>>, const char*, std::vector<std::string>>(
     "Foods",
     { "Apple", "Banana", "Cherry", "Donut", "Egg" }
    ));
    */
    auto struct_= make_ASTNodeRef<ASTNodeStruct, ASTFieldList> (
        "Struct", 
        { {"field1", "int"}, {"field2", "double"}, {"field3", "string"} }
    );

    auto table = make_ASTNodeRef<ASTNodeTable, ASTFieldList> ( 
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
