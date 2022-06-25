#pragma once
#include <iostream>
#include <list>

namespace My {
class TreeNode {
   protected:
    TreeNode* m_Parent = nullptr;
    std::list<std::shared_ptr<TreeNode>> m_Children;

   protected:
    virtual void dump(std::ostream& out) const {
        out << "Tree Node" << std::endl;
        out << "----------" << std::endl;
    };

   public:
    virtual ~TreeNode() = default;

    virtual void AppendChild(std::shared_ptr<TreeNode>&& sub_node) {
        sub_node->m_Parent = this;
        m_Children.push_back(std::move(sub_node));
    }

    friend std::ostream& operator<<(std::ostream& out, const TreeNode& node) {
        node.dump(out);
        out << std::endl;

        return out;
    }
};
}  // namespace My
