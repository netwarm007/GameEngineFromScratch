#pragma once
#include <iostream>
#include <list>

namespace My {
class TreeNode {
   protected:
    TreeNode* m_Parent;
    std::list<std::shared_ptr<TreeNode>> m_Children;

   protected:
    virtual void dump(std::ostream& out) const {};

   public:
    virtual ~TreeNode() = default;

    virtual void AppendChild(std::shared_ptr<TreeNode>&& sub_node) {
        sub_node->m_Parent = this;
        m_Children.push_back(std::move(sub_node));
    }

    friend std::ostream& operator<<(std::ostream& out, const TreeNode& node) {
        static thread_local int32_t indent = 0;
        indent++;

        out << std::string(indent, ' ') << "Tree Node" << std::endl;
        out << std::string(indent, ' ') << "----------" << std::endl;
        node.dump(out);
        out << std::endl;

        for (const std::shared_ptr<TreeNode>& sub_node : node.m_Children) {
            out << *sub_node << std::endl;
        }

        indent--;

        return out;
    }
};
}  // namespace My