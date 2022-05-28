#include <functional>
#include <iostream>
#include <memory>
#include <string>

#include "AST.hpp"

using namespace My;

void travelAndPrint(ASTNodeRef node) {
    assert(node);
    std::cout << *node << std::endl;
    auto left = node->GetLeft();
    if (left)
        travelAndPrint(left);
    auto right = node->GetRight();
    if (right)
        travelAndPrint(right);
};

int main(int argc, char* argv[])
{
    auto root   = make_ASTNodeRef<ASTNodeNone>( "MODULE" );

    auto ns     = make_ASTNodeRef<ASTNodeNameSpace, const char*>( 
                                                        "My", "https://www.chenwenli.com" );

    auto enum_  = make_ASTNodeRef<ASTNodeEnum, ASTEnumItems> ( 
        "Foods", 
        { {"Apple", 1}, {"Banana", 2}, {"Cherry", 3}, {"Donut", 4}, {"Egg", 5} }
    );

    auto struct_= make_ASTNodeRef<ASTNodeStruct, ASTFieldList> (
        "Struct", 
        { {"field1", enum_}, {"field2", enum_}, {"field3", enum_} }
    );

    auto table = make_ASTNodeRef<ASTNodeTable, ASTFieldList> ( 
        "Table",
        { {"field1", enum_}, {"field2", struct_}, {"field3", enum_} }
    );

    root->SetLeft(ns);
    ns->SetLeft(enum_);
    enum_->SetRight(struct_);
    struct_->SetRight(table);


    travelAndPrint(root);

    return 0;
}
