#include <iostream>
#include <string>
#include <memory>

#include "AST.hpp"

using namespace My;

int main(int argc, char* argv[])
{
    using ASTNodeString = ASTNode<std::string, const char*>;
    using ASTNodeInt    = ASTNode<int64_t, int64_t>;
    using ASTNodeFloat  = ASTNode<float, float>;
    auto root = std::make_shared<ASTNodeString>("root node");
    auto left = std::make_shared<ASTNodeInt>(32);
    auto right = std::make_shared<ASTNodeFloat>(3.14f);

    root->SetLeft(left);
    root->SetRight(right);

    std::cout << *root << std::endl;

    return 0;
}