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


#include "ODDLMap.h"


using namespace ODDL;


MapElementBase::~MapElementBase()
{
	if (owningMap)
	{
		owningMap->RemoveNode(this);
	}
}

void MapElementBase::Detach(void)
{
	if (owningMap)
	{
		owningMap->RemoveNode(this);
	}
}

MapElementBase *MapElementBase::First(void)
{
	MapElementBase *element = this;
	for (;;)
	{
		MapElementBase *left = element->leftSubnode;
		if (!left)
		{
			break;
		}

		element = left;
	}

	return (element);
}

MapElementBase *MapElementBase::Last(void)
{
	MapElementBase *element = this;
	for (;;)
	{
		MapElementBase *right = element->rightSubnode;
		if (!right)
		{
			break;
		}

		element = right;
	}

	return (element);
}

MapElementBase *MapElementBase::Previous(void) const
{
	if (leftSubnode)
	{
		return (leftSubnode->Last());
	}

	const MapElementBase *element = this;
	for (;;)
	{
		MapElementBase *super = element->superNode;
		if (!super)
		{
			break;
		}

		if (super->rightSubnode == element)
		{
			return (super);
		}

		element = super;
	}

	return (nullptr);
}

MapElementBase *MapElementBase::Next(void) const
{
	if (rightSubnode)
	{
		return (rightSubnode->First());
	}

	const MapElementBase *element = this;
	for (;;)
	{
		MapElementBase *super = element->superNode;
		if (!super)
		{
			break;
		}

		if (super->leftSubnode == element)
		{
			return (super);
		}

		element = super;
	}

	return (nullptr);
}

void MapElementBase::RemoveSubtree(void)
{
	if (leftSubnode)
	{
		leftSubnode->RemoveSubtree();
	}

	if (rightSubnode)
	{
		rightSubnode->RemoveSubtree();
	}

	superNode = nullptr;
	leftSubnode = nullptr;
	rightSubnode = nullptr;
	owningMap = nullptr;
}

void MapElementBase::PurgeSubtree(void)
{
	if (leftSubnode)
	{
		leftSubnode->PurgeSubtree();
	}

	if (rightSubnode)
	{
		rightSubnode->PurgeSubtree();
	}

	owningMap = nullptr;
	delete this;
}


MapBase::~MapBase()
{
	Purge();
}

MapElementBase *MapBase::operator [](machine index) const
{
	machine i = 0;
	MapElementBase *element = First();
	while (element)
	{
		if (i == index)
		{
			return (element);
		}

		element = element->Next();
		i++;
	}

	return (nullptr);
}

int32 MapBase::GetElementCount(void) const
{
	machine count = 0;
	const MapElementBase *element = First();
	while (element)
	{
		count++;
		element = element->Next();
	}

	return ((int32) count);
}

MapElementBase *MapBase::RotateLeft(MapElementBase *node)
{
	MapElementBase *right = node->rightSubnode;

	if (node != rootNode)
	{
		MapElementBase *super = node->superNode;

		if (super->leftSubnode == node)
		{
			super->leftSubnode = right;
		}
		else
		{
			super->rightSubnode = right;
		}

		right->superNode = super;
	}
	else
	{
		rootNode = right;
		right->superNode = nullptr;
	}

	MapElementBase *subnode = right->leftSubnode;
	if (subnode)
	{
		subnode->superNode = node;
	}

	node->rightSubnode = subnode;

	right->leftSubnode = node;
	node->superNode = right;
	node->balance = -(--right->balance);

	return (right);
}

MapElementBase *MapBase::RotateRight(MapElementBase *node)
{
	MapElementBase *left = node->leftSubnode;

	if (node != rootNode)
	{
		MapElementBase *super = node->superNode;

		if (super->leftSubnode == node)
		{
			super->leftSubnode = left;
		}
		else
		{
			super->rightSubnode = left;
		}

		left->superNode = super;
	}
	else
	{
		rootNode = left;
		left->superNode = nullptr;
	}

	MapElementBase *subnode = left->rightSubnode;
	if (subnode)
	{
		subnode->superNode = node;
	}

	node->leftSubnode = subnode;

	left->rightSubnode = node;
	node->superNode = left;
	node->balance = -(++left->balance);

	return (left);
}

MapElementBase *MapBase::ZigZagLeft(MapElementBase *node)
{
	MapElementBase *right = node->rightSubnode;
	MapElementBase *top = right->leftSubnode;

	if (node != rootNode)
	{
		MapElementBase *super = node->superNode;

		if (super->leftSubnode == node)
		{
			super->leftSubnode = top;
		}
		else
		{
			super->rightSubnode = top;
		}

		top->superNode = super;
	}
	else
	{
		rootNode = top;
		top->superNode = nullptr;
	}

	MapElementBase *subLeft = top->leftSubnode;
	if (subLeft)
	{
		subLeft->superNode = node;
	}

	node->rightSubnode = subLeft;

	MapElementBase *subRight = top->rightSubnode;
	if (subRight)
	{
		subRight->superNode = right;
	}

	right->leftSubnode = subRight;

	top->leftSubnode = node;
	top->rightSubnode = right;
	node->superNode = top;
	right->superNode = top;

	int32 b = top->balance;
	node->balance = -MaxZero(b);
	right->balance = -MinZero(b);
	top->balance = 0;

	return (top);
}

MapElementBase *MapBase::ZigZagRight(MapElementBase *node)
{
	MapElementBase *left = node->leftSubnode;
	MapElementBase *top = left->rightSubnode;

	if (node != rootNode)
	{
		MapElementBase *super = node->superNode;

		if (super->leftSubnode == node)
		{
			super->leftSubnode = top;
		}
		else
		{
			super->rightSubnode = top;
		}

		top->superNode = super;
	}
	else
	{
		rootNode = top;
		top->superNode = nullptr;
	}

	MapElementBase *subLeft = top->leftSubnode;
	if (subLeft)
	{
		subLeft->superNode = left;
	}

	left->rightSubnode = subLeft;

	MapElementBase *subRight = top->rightSubnode;
	if (subRight)
	{
		subRight->superNode = node;
	}

	node->leftSubnode = subRight;

	top->leftSubnode = left;
	top->rightSubnode = node;
	node->superNode = top;
	left->superNode = top;

	int32 b = top->balance;
	node->balance = -MinZero(b);
	left->balance = -MaxZero(b);
	top->balance = 0;

	return (top);
}

void MapBase::SetRootNode(MapElementBase *node)
{
	MapBase *map = node->owningMap;
	if (map)
	{
		map->RemoveNode(node);
	}

	node->owningMap = this;
	node->balance = 0;

	rootNode = node;
}

void MapBase::InsertLeftSubnode(MapElementBase *node, MapElementBase *subnode)
{
	MapBase *map = subnode->owningMap;
	if (map)
	{
		map->RemoveNode(subnode);
	}

	node->leftSubnode = subnode;
	subnode->superNode = node;
	subnode->owningMap = this;
	subnode->balance = 0;

	int32 b = node->balance - 1;
	node->balance = b;
	if (b != 0)
	{
		int32 dir1 = -1;
		for (;;)
		{
			int32	dir2;

			MapElementBase *super = node->superNode;
			if (!super)
			{
				break;
			}

			b = super->balance;
			if (super->leftSubnode == node)
			{
				super->balance = --b;
				dir2 = -1;
			}
			else
			{
				super->balance = ++b;
				dir2 = 1;
			}

			if (b == 0)
			{
				break;
			}

			if (Abs(b) == 2)
			{
				if (dir2 == -1)
				{
					if (dir1 == -1)
					{
						RotateRight(super);
					}
					else
					{
						ZigZagRight(super);
					}
				}
				else
				{
					if (dir1 == 1)
					{
						RotateLeft(super);
					}
					else
					{
						ZigZagLeft(super);
					}
				}

				break;
			}

			dir1 = dir2;
			node = super;
		}
	}
}

void MapBase::InsertRightSubnode(MapElementBase *node, MapElementBase *subnode)
{
	MapBase *map = subnode->owningMap;
	if (map)
	{
		map->RemoveNode(subnode);
	}

	node->rightSubnode = subnode;
	subnode->superNode = node;
	subnode->owningMap = this;
	subnode->balance = 0;

	int32 b = node->balance + 1;
	node->balance = b;
	if (b != 0)
	{
		int32 dir1 = 1;
		for (;;)
		{
			int32	dir2;

			MapElementBase *super = node->superNode;
			if (!super)
			{
				break;
			}

			b = super->balance;
			if (super->leftSubnode == node)
			{
				super->balance = --b;
				dir2 = -1;
			}
			else
			{
				super->balance = ++b;
				dir2 = 1;
			}

			if (b == 0)
			{
				break;
			}

			if (Abs(b) == 2)
			{
				if (dir2 == -1)
				{
					if (dir1 == -1)
					{
						RotateRight(super);
					}
					else
					{
						ZigZagRight(super);
					}
				}
				else
				{
					if (dir1 == 1)
					{
						RotateLeft(super);
					}
					else
					{
						ZigZagLeft(super);
					}
				}

				break;
			}

			dir1 = dir2;
			node = super;
		}
	}
}

void MapBase::ReplaceNode(MapElementBase *node, MapElementBase *element)
{
	MapBase *map = element->owningMap;
	if (map)
	{
		map->RemoveNode(element);
	}

	MapElementBase *super = node->superNode;
	if (super)
	{
		if (super->leftSubnode == node)
		{
			super->leftSubnode = element;
		}
		else
		{
			super->rightSubnode = element;
		}
	}

	element->superNode = super;
	element->balance = node->balance;
	element->owningMap = this;

	MapElementBase *subnode = node->leftSubnode;
	element->leftSubnode = subnode;
	if (subnode)
	{
		subnode->superNode = element;
	}

	subnode = node->rightSubnode;
	element->rightSubnode = subnode;
	if (subnode)
	{
		subnode->superNode = element;
	}

	node->superNode = nullptr;
	node->leftSubnode = nullptr;
	node->rightSubnode = nullptr;
	node->owningMap = nullptr;
}

void MapBase::RemoveBranchNode(MapElementBase *node, MapElementBase *subnode)
{
	MapElementBase *super = node->superNode;
	if (subnode)
	{
		subnode->superNode = super;
	}

	if (super)
	{
		int32	db;

		if (super->leftSubnode == node)
		{
			super->leftSubnode = subnode;
			db = 1;
		}
		else
		{
			super->rightSubnode = subnode;
			db = -1;
		}

		for (;;)
		{
			int32 b = (super->balance += db);
			if (Abs(b) == 1)
			{
				break;
			}

			node = super;
			super = super->superNode;

			if (b != 0)
			{
				if (b > 0)
				{
					int32 rb = node->rightSubnode->balance;
					if (rb >= 0)
					{
						node = RotateLeft(node);
						if (rb == 0)
						{
							break;
						}
					}
					else
					{
						node = ZigZagLeft(node);
					}
				}
				else
				{
					int32 lb = node->leftSubnode->balance;
					if (lb <= 0)
					{
						node = RotateRight(node);
						if (lb == 0)
						{
							break;
						}
					}
					else
					{
						node = ZigZagRight(node);
					}
				}
			}

			if (!super)
			{
				break;
			}

			db = (super->leftSubnode == node) ? 1 : -1;
		}
	}
	else
	{
		rootNode = subnode;
	}
}

void MapBase::RemoveNode(MapElementBase *node)
{
	MapElementBase *left = node->leftSubnode;
	MapElementBase *right = node->rightSubnode;

	if ((left) && (right))
	{
		MapElementBase *top = right->First();
		RemoveBranchNode(top, top->rightSubnode);

		MapElementBase *super = node->superNode;
		top->superNode = super;
		if (super)
		{
			if (super->leftSubnode == node)
			{
				super->leftSubnode = top;
			}
			else
			{
				super->rightSubnode = top;
			}
		}
		else
		{
			rootNode = top;
		}

		left = node->leftSubnode;
		top->leftSubnode = left;
		if (left)
		{
			left->superNode = top;
		}

		right = node->rightSubnode;
		top->rightSubnode = right;
		if (right)
		{
			right->superNode = top;
		}

		top->balance = node->balance;
	}
	else
	{
		RemoveBranchNode(node, (left) ? left : right);
	}

	node->superNode = nullptr;
	node->leftSubnode = nullptr;
	node->rightSubnode = nullptr;
	node->owningMap = nullptr;
}

void MapBase::RemoveAll(void)
{
	if (rootNode)
	{
		rootNode->RemoveSubtree();
		rootNode = nullptr;
	}
}

void MapBase::Purge(void)
{
	if (rootNode)
	{
		rootNode->PurgeSubtree();
		rootNode = nullptr;
	}
}
