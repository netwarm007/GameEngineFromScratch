#include <iostream>
#include <string>
#include <memory>

#include "AST.hpp"

using namespace My;

int main(int argc, char* argv[])
{
    // next statement equivalent to: std::shared_ptr<ASTNode> root = std::static_pointer_cast<ASTNode>(std::make_shared<ASTNodeT<AST_NODE_TYPE::NONE, void>, const char*>( "module"));
    auto root   = make_ASTNodeRef<ASTNodeNone>( "module" );

    auto ns     = make_ASTNodeRef<ASTNodeNameSpace, const char*>( 
                                                        "namespace", "My" );

    auto enum_  = make_ASTNodeRef<ASTNodeEnum, ASTNodeEnumValueType> ( 
        "Enum", 
        { "Apple", "Banana", "Cherry", "Donut", "Egg" }
    );

    /* next statement equivalent to:
    std::shared_ptr<ASTNode> enum_ = std::static_pointer_cast<ASTNode>(std::make_shared<ASTNodeT<AST_NODE_TYPE::ENUM, std::vector<std::string>, std::vector<std::string>>, const char*, std::vector<std::string>>(
     "Enum",
     { "Apple", "Banana", "Cherry", "Donut", "Egg" }
    ));
    */
    auto struct_= make_ASTNodeRef<ASTNodeStruct, ASTNodeStructValueType> (
        "Struct", 
        { {"field1", "int"}, {"field2", "double"}, {"field3", "string"} }
    );

    auto table = make_ASTNodeRef<ASTNodeTable, ASTNodeTableValueType> ( 
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
