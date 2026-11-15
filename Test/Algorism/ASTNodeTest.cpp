#include <iostream>
#include <map>
#include <sstream>

#include "AST.hpp"

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::cerr << "FAILED: " #cond " at line " << __LINE__ << "\n"; \
            return 1;                                                       \
        }                                                                   \
    } while (0)

namespace My {
std::map<ASTNode::IDN_TYPE, ASTNodeRef> global_symbol_table;
}

using namespace My;

int main() {
    auto none = make_ASTNodeRef<ASTNodeNone>("root");
    auto primitive = make_ASTNodeRef<ASTNodePrimitive>("int");
    auto name_space =
        make_ASTNodeRef<ASTNodeNameSpace, const char*>("namespace", "https://example.com");
    auto enum_node = make_ASTNodeRef<ASTNodeEnum, ASTEnumItems>(
        "enum", ASTEnumItems{{"first", 1}, {"second", 2}});
    auto structure = make_ASTNodeRef<ASTNodeStruct, ASTFieldList>(
        "struct", ASTFieldList{{"field", primitive, true}});
    auto table = make_ASTNodeRef<ASTNodeTable, ASTFieldList>(
        "table", ASTFieldList{{"column", enum_node, false}});
    auto attribute = make_ASTNodeRef<ASTNodeAttribute>("attribute");
    auto root_type = make_ASTNodeRef<ASTNodeRootType>("root type");
    auto namespace_value = std::dynamic_pointer_cast<ASTNodeNameSpace<const char*>>(name_space);
    auto enum_value = std::dynamic_pointer_cast<ASTNodeEnum<ASTEnumItems>>(enum_node);
    auto structure_value = std::dynamic_pointer_cast<ASTNodeStruct<ASTFieldList>>(structure);
    auto table_value = std::dynamic_pointer_cast<ASTNodeTable<ASTFieldList>>(table);

    CHECK(none->GetNodeType() == AST_NODE_TYPE::NONE);
    CHECK(primitive->GetNodeType() == AST_NODE_TYPE::PRIMITIVE);
    CHECK(name_space->GetNodeType() == AST_NODE_TYPE::NAMESPACE);
    CHECK(enum_node->GetNodeType() == AST_NODE_TYPE::ENUM);
    CHECK(structure->GetNodeType() == AST_NODE_TYPE::STRUCT);
    CHECK(table->GetNodeType() == AST_NODE_TYPE::TABLE);
    CHECK(attribute->GetNodeType() == AST_NODE_TYPE::ATTRIBUTE);
    CHECK(root_type->GetNodeType() == AST_NODE_TYPE::ROOTTYPE);
    CHECK(none->GetIDN() == "root");
    CHECK(namespace_value && namespace_value->GetValue() == "https://example.com");
    CHECK(enum_value && enum_value->GetValue().size() == 2);
    CHECK(std::get<1>(enum_value->GetValue()[1]) == 2);
    CHECK(std::get<0>(structure_value->GetValue()[0]) == "field");
    CHECK(std::get<1>(structure_value->GetValue()[0]) == primitive);
    CHECK(std::get<2>(structure_value->GetValue()[0]));
    CHECK(std::get<0>(table_value->GetValue()[0]) == "column");
    CHECK(!std::get<2>(table_value->GetValue()[0]));

    none->SetLeft(name_space);
    none->SetRight(table);
    CHECK(none->GetLeft() == name_space);
    CHECK(none->GetRight() == table);
    none->SetLeft(nullptr);
    CHECK(!none->GetLeft());

    const std::pair<ASTNodeRef, const char*> typed_nodes[] = {
        {none, "AST_NODE_TYPE::NONE"},
        {primitive, "AST_NODE_TYPE::PRIMITIVE"},
        {enum_node, "AST_NODE_TYPE::ENUM"},
        {name_space, "AST_NODE_TYPE::NAMESPACE"},
        {structure, "AST_NODE_TYPE::STRUCT"},
        {table, "AST_NODE_TYPE::TABLE"},
        {attribute, "AST_NODE_TYPE::ATTRIBUTE"},
        {root_type, "AST_NODE_TYPE::ROOTTYPE"},
    };
    for (const auto& [node, type_name] : typed_nodes) {
        std::ostringstream output;
        output << *node;
        CHECK(output.str().find(type_name) != std::string::npos);
    }

    global_symbol_table.clear();
    auto [root_found_before_insert, missing_root] = findRootType();
    CHECK(!root_found_before_insert);
    CHECK(missing_root.empty());
    auto [symbol_found, missing_symbol] = findSymbol("missing");
    CHECK(!symbol_found);
    CHECK(!missing_symbol);

    global_symbol_table["[root type]"] = root_type;
    global_symbol_table["int"] = primitive;
    auto [root_found, root_idn] = findRootType();
    CHECK(root_found);
    CHECK(root_idn == "root type");
    auto [found_symbol, found_node] = findSymbol("int");
    CHECK(found_symbol);
    CHECK(found_node == primitive);

    return 0;
}
