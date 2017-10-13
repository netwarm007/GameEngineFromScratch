/*
	OpenDDL Library Software License
	==================================

	OpenDDL Library, version 2.0
	Copyright 2014-2017, Eric Lengyel
	All rights reserved.

	The OpenDDL Library is free software published on the following website:

		http://openddl.org/

	Redistribution and use in source and binary forms, with or without modification,
	are permitted provided that the following conditions are met:

	1. Redistributions of source code must retain the entire text of this license,
	comprising the above copyright notice, this list of conditions, and the following
	disclaimer.
	
	2. Redistributions of any modified source code files must contain a prominent
	notice immediately following this license stating that the contents have been
	modified from their original form.

	3. Redistributions in binary form must include attribution to the author in any
	listing of credits provided with the distribution. If there is no listing of
	credits, then attribution must be included in the documentation and/or other
	materials provided with the distribution. The attribution must be exactly the
	statement "This software contains the OpenDDL Library by Eric Lengyel" (without
	quotes) in the case that the distribution contains the original, unmodified
	OpenDDL Library, or it must be exactly the statement "This software contains a
	modified version of the OpenDDL Library by Eric Lengyel" (without quotes) in the
	case that the distribution contains a modified version of the OpenDDL Library.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
	ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
	WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
	IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
	INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
	NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
	PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
	WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
	ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
	POSSIBILITY OF SUCH DAMAGE.
*/


#include "ODDLTree.h"


using namespace ODDL;


TreeBase::~TreeBase()
{
	PurgeSubtree();

	if (superNode)
	{
		superNode->RemoveSubnode(this);
	}
}

TreeBase *TreeBase::GetRootNode(void)
{
	TreeBase *root = this;
	for (;;)
	{
		TreeBase *node = root->superNode;
		if (!node)
		{
			break;
		}

		root = node;
	}

	return (root);
}

const TreeBase *TreeBase::GetRootNode(void) const
{
	const TreeBase *root = this;
	for (;;)
	{
		const TreeBase *node = root->superNode;
		if (!node)
		{
			break;
		}

		root = node;
	}

	return (root);
}

bool TreeBase::Successor(const TreeBase *node) const
{
	TreeBase *super = node->superNode;
	while (super)
	{
		if (super == this)
		{
			return (true);
		}

		super = super->superNode;
	}

	return (false);
}

TreeBase *TreeBase::GetLeftmostNode(void)
{
	TreeBase *node = this;
	for (;;)
	{
		TreeBase *subnode = node->firstSubnode;
		if (!subnode)
		{
			break;
		}

		node = subnode;
	}

	return (node);
}

const TreeBase *TreeBase::GetLeftmostNode(void) const
{
	const TreeBase *node = this;
	for (;;)
	{
		const TreeBase *subnode = node->firstSubnode;
		if (!subnode)
		{
			break;
		}

		node = subnode;
	}

	return (node);
}

TreeBase *TreeBase::GetRightmostNode(void)
{
	TreeBase *node = this;
	for (;;)
	{
		TreeBase *subnode = node->lastSubnode;
		if (!subnode)
		{
			break;
		}

		node = subnode;
	}

	return (node);
}

const TreeBase *TreeBase::GetRightmostNode(void) const
{
	const TreeBase *node = this;
	for (;;)
	{
		const TreeBase *subnode = node->lastSubnode;
		if (!subnode)
		{
			break;
		}

		node = subnode;
	}

	return (node);
}

TreeBase *TreeBase::GetNextNode(const TreeBase *node) const
{
	TreeBase *next = node->GetFirstSubnode();
	if (!next)
	{
		for (;;)
		{
			if (node == this)
			{
				break;
			}

			next = node->nextNode;
			if (next)
			{
				break;
			}

			node = node->superNode;
		}
	}

	return (next);
}

TreeBase *TreeBase::GetPreviousNode(const TreeBase *node)
{
	if (node == this)
	{
		return (nullptr);
	}

	TreeBase *prev = node->prevNode;
	if (!prev)
	{
		return (node->superNode);
	}

	return (prev->GetRightmostNode());
}

const TreeBase *TreeBase::GetPreviousNode(const TreeBase *node) const
{
	if (node == this)
	{
		return (nullptr);
	}

	const TreeBase *prev = node->prevNode;
	if (!prev)
	{
		return (node->superNode);
	}

	return (prev->GetRightmostNode());
}

TreeBase *TreeBase::GetNextLevelNode(const TreeBase *node) const
{
	TreeBase *next = nullptr;
	for (;;)
	{
		if (node == this)
		{
			break;
		}

		next = node->Next();
		if (next)
		{
			break;
		}

		node = node->superNode;
	}

	return (next);
}

TreeBase *TreeBase::GetPreviousLevelNode(const TreeBase *node) const
{
	TreeBase *prev = nullptr;
	for (;;)
	{
		if (node == this)
		{
			break;
		}

		prev = node->Previous();
		if (prev)
		{
			break;
		}

		node = node->superNode;
	}

	return (prev);
}

int32 TreeBase::GetSubnodeCount(void) const
{
	machine count = 0;
	const TreeBase *subnode = firstSubnode;
	while (subnode)
	{
		count++;
		subnode = subnode->nextNode;
	}

	return ((int32) count);
}

int32 TreeBase::GetSubtreeNodeCount(void) const
{
	machine count = 0;
	const TreeBase *subnode = firstSubnode;
	while (subnode)
	{
		count++;
		subnode = GetNextNode(subnode);
	}

	return ((int32) count);
}

int32 TreeBase::GetNodeIndex(void) const
{
	machine index = 0;

	const TreeBase *element = this;
	for (;;)
	{
		element = element->Previous();
		if (!element)
		{
			break;
		}

		index++;
	}

	return ((int32) index);
}

int32 TreeBase::GetNodeDepth(void) const
{
	machine depth = 0;

	const TreeBase *element = this;
	for (;;)
	{
		element = element->GetSuperNode();
		if (!element)
		{
			break;
		}

		depth++;
	}

	return ((int32) depth);
}

void TreeBase::RemoveSubtree(void)
{
	TreeBase *subnode = firstSubnode;
	while (subnode)
	{
		TreeBase *next = subnode->nextNode;
		subnode->prevNode = nullptr;
		subnode->nextNode = nullptr;
		subnode->superNode = nullptr;
		subnode = next;
	}

	firstSubnode = nullptr;
	lastSubnode = nullptr;
}

void TreeBase::PurgeSubtree(void)
{
	while (firstSubnode)
	{
		delete firstSubnode;
	}
}

void TreeBase::AppendSubnode(TreeBase *node)
{
	TreeBase *tree = node->superNode;
	if (tree)
	{
		TreeBase *prev = node->prevNode;
		TreeBase *next = node->nextNode;

		if (prev)
		{
			prev->nextNode = next;
			node->prevNode = nullptr;
		}

		if (next)
		{
			next->prevNode = prev;
			node->nextNode = nullptr;
		}

		if (tree->firstSubnode == node)
		{
			tree->firstSubnode = next;
		}

		if (tree->lastSubnode == node)
		{
			tree->lastSubnode = prev;
		}
	}

	if (lastSubnode)
	{
		lastSubnode->nextNode = node;
		node->prevNode = lastSubnode;
		lastSubnode = node;
	}
	else
	{
		firstSubnode = node;
		lastSubnode = node;
	}

	node->superNode = this;
}

void TreeBase::PrependSubnode(TreeBase *node)
{
	TreeBase *tree = node->superNode;
	if (tree)
	{
		TreeBase *prev = node->prevNode;
		TreeBase *next = node->nextNode;

		if (prev)
		{
			prev->nextNode = next;
			node->prevNode = nullptr;
		}

		if (next)
		{
			next->prevNode = prev;
			node->nextNode = nullptr;
		}

		if (tree->firstSubnode == node)
		{
			tree->firstSubnode = next;
		}

		if (tree->lastSubnode == node)
		{
			tree->lastSubnode = prev;
		}
	}

	if (firstSubnode)
	{
		firstSubnode->prevNode = node;
		node->nextNode = firstSubnode;
		firstSubnode = node;
	}
	else
	{
		firstSubnode = node;
		lastSubnode = node;
	}

	node->superNode = this;
}

void TreeBase::InsertSubnodeBefore(TreeBase *node, TreeBase *before)
{
	TreeBase *tree = node->superNode;
	if (tree)
	{
		TreeBase *prev = node->prevNode;
		TreeBase *next = node->nextNode;

		if (prev)
		{
			prev->nextNode = next;
		}

		if (next)
		{
			next->prevNode = prev;
		}

		if (tree->firstSubnode == node)
		{
			tree->firstSubnode = next;
		}

		if (tree->lastSubnode == node)
		{
			tree->lastSubnode = prev;
		}
	}

	node->superNode = this;
	node->nextNode = before;

	if (before)
	{
		TreeBase *after = before->prevNode;
		node->prevNode = after;
		before->prevNode = node;

		if (after)
		{
			after->nextNode = node;
		}
		else
		{
			firstSubnode = node;
		}
	}
	else
	{
		TreeBase *after = lastSubnode;
		node->prevNode = after;

		if (after)
		{
			after->nextNode = node;
			lastSubnode = node;
		}
		else
		{
			firstSubnode = node;
			lastSubnode = node;
		}
	}
}

void TreeBase::InsertSubnodeAfter(TreeBase *node, TreeBase *after)
{
	TreeBase *tree = node->superNode;
	if (tree)
	{
		TreeBase *prev = node->prevNode;
		TreeBase *next = node->nextNode;

		if (prev)
		{
			prev->nextNode = next;
		}

		if (next)
		{
			next->prevNode = prev;
		}

		if (tree->firstSubnode == node)
		{
			tree->firstSubnode = next;
		}

		if (tree->lastSubnode == node)
		{
			tree->lastSubnode = prev;
		}
	}

	node->superNode = this;
	node->prevNode = after;

	if (after)
	{
		TreeBase *before = after->nextNode;
		node->nextNode = before;
		after->nextNode = node;

		if (before)
		{
			before->prevNode = node;
		}
		else
		{
			lastSubnode = node;
		}
	}
	else
	{
		TreeBase *before = firstSubnode;
		node->nextNode = before;

		if (before)
		{
			before->prevNode = node;
			firstSubnode = node;
		}
		else
		{
			firstSubnode = node;
			lastSubnode = node;
		}
	}
}

void TreeBase::RemoveSubnode(TreeBase *subnode)
{
	TreeBase *prev = subnode->prevNode;
	TreeBase *next = subnode->nextNode;

	if (prev)
	{
		prev->nextNode = next;
	}

	if (next)
	{
		next->prevNode = prev;
	}

	if (firstSubnode == subnode)
	{
		firstSubnode = next;
	}

	if (lastSubnode == subnode)
	{
		lastSubnode = prev;
	}

	subnode->prevNode = nullptr;
	subnode->nextNode = nullptr;
	subnode->superNode = nullptr;
}

void TreeBase::Detach(void)
{
	if (superNode)
	{
		superNode->RemoveSubnode(this);
	}
}
