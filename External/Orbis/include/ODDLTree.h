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


#ifndef ODDLTree_h
#define ODDLTree_h


/*
	This file contains simplified versions of the tree container classes used by the Tombstone Engine.
*/


#include "ODDLTypes.h"


namespace ODDL
{
	class TreeBase
	{
		private:

			TreeBase		*prevNode;
			TreeBase		*nextNode;
			TreeBase		*superNode;
			TreeBase		*firstSubnode;
			TreeBase		*lastSubnode;

		protected:

			TreeBase()
			{
				prevNode = nullptr;
				nextNode = nullptr;
				superNode = nullptr;
				firstSubnode = nullptr;
				lastSubnode = nullptr;
			}

			virtual ~TreeBase();

			TreeBase *Previous(void) const
			{
				return (prevNode);
			}

			TreeBase *Next(void) const
			{
				return (nextNode);
			}

			TreeBase *GetSuperNode(void) const
			{
				return (superNode);
			}

			TreeBase *GetFirstSubnode(void) const
			{
				return (firstSubnode);
			}

			TreeBase *GetLastSubnode(void) const
			{
				return (lastSubnode);
			}

			TreeBase *GetRootNode(void);
			const TreeBase *GetRootNode(void) const;

			bool Successor(const TreeBase *node) const;

			TreeBase *GetLeftmostNode(void);
			const TreeBase *GetLeftmostNode(void) const;
			TreeBase *GetRightmostNode(void);
			const TreeBase *GetRightmostNode(void) const;

			TreeBase *GetNextNode(const TreeBase *node) const;
			TreeBase *GetPreviousNode(const TreeBase *node);
			const TreeBase *GetPreviousNode(const TreeBase *node) const;
			TreeBase *GetNextLevelNode(const TreeBase *node) const;
			TreeBase *GetPreviousLevelNode(const TreeBase *node) const;

			void AppendSubnode(TreeBase *node);
			void PrependSubnode(TreeBase *node);
			void InsertSubnodeBefore(TreeBase *node, TreeBase *before);
			void InsertSubnodeAfter(TreeBase *node, TreeBase *after);
			void RemoveSubnode(TreeBase *subnode);

		public:

			int32 GetSubnodeCount(void) const;
			int32 GetSubtreeNodeCount(void) const;

			int32 GetNodeIndex(void) const;
			int32 GetNodeDepth(void) const;

			void RemoveSubtree(void);
			void PurgeSubtree(void);

			virtual void Detach(void);
	};


	template <class type> class Tree : public TreeBase
	{
		protected:

			Tree() = default;

		public:

			~Tree() = default;

			type *Previous(void) const
			{
				return (static_cast<type *>(static_cast<Tree<type> *>(TreeBase::Previous())));
			}

			type *Next(void) const
			{
				return (static_cast<type *>(static_cast<Tree<type> *>(TreeBase::Next())));
			}

			type *GetSuperNode(void) const
			{
				return (static_cast<type *>(static_cast<Tree<type> *>(TreeBase::GetSuperNode())));
			}

			type *GetFirstSubnode(void) const
			{
				return (static_cast<type *>(static_cast<Tree<type> *>(TreeBase::GetFirstSubnode())));
			}

			type *GetLastSubnode(void) const
			{
				return (static_cast<type *>(static_cast<Tree<type> *>(TreeBase::GetLastSubnode())));
			}

			type *GetRootNode(void)
			{
				return (static_cast<type *>(static_cast<Tree<type> *>(TreeBase::GetRootNode())));
			}

			const type *GetRootNode(void) const
			{
				return (static_cast<const type *>(static_cast<const Tree<type> *>(TreeBase::GetRootNode())));
			}

			bool Successor(const Tree<type> *node) const
			{
				return (TreeBase::Successor(node));
			}

			type *GetLeftmostNode(void)
			{
				return (static_cast<type *>(static_cast<Tree<type> *>(TreeBase::GetLeftmostNode())));
			}

			const type *GetLeftmostNode(void) const
			{
				return (static_cast<const type *>(static_cast<const Tree<type> *>(TreeBase::GetLeftmostNode())));
			}

			type *GetRightmostNode(void)
			{
				return (static_cast<type *>(static_cast<Tree<type> *>(TreeBase::GetRightmostNode())));
			}

			const type *GetRightmostNode(void) const
			{
				return (static_cast<const type *>(static_cast<const Tree<type> *>(TreeBase::GetRightmostNode())));
			}

			type *GetNextNode(const Tree<type> *node) const
			{
				return (static_cast<type *>(static_cast<Tree<type> *>(TreeBase::GetNextNode(node))));
			}

			type *GetPreviousNode(const Tree<type> *node)
			{
				return (static_cast<type *>(static_cast<Tree<type> *>(TreeBase::GetPreviousNode(node))));
			}

			const type *GetPreviousNode(const Tree<type> *node) const
			{
				return (static_cast<const type *>(static_cast<const Tree<type> *>(TreeBase::GetPreviousNode(node))));
			}

			type *GetNextLevelNode(const Tree<type> *node) const
			{
				return (static_cast<type *>(static_cast<Tree<type> *>(TreeBase::GetNextLevelNode(node))));
			}

			type *GetPreviousLevelNode(const Tree<type> *node) const
			{
				return (static_cast<type *>(static_cast<Tree<type> *>(TreeBase::GetPreviousLevelNode(node))));
			}

			virtual void AppendSubnode(type *node);
			virtual void PrependSubnode(type *node);
			virtual void InsertSubnodeBefore(type *node, type *before);
			virtual void InsertSubnodeAfter(type *node, type *after);
			virtual void RemoveSubnode(type *node);
	};


	template <class type> void Tree<type>::AppendSubnode(type *node)
	{
		TreeBase::AppendSubnode(static_cast<Tree<type> *>(node));
	}

	template <class type> void Tree<type>::PrependSubnode(type *node)
	{
		TreeBase::PrependSubnode(static_cast<Tree<type> *>(node));
	}

	template <class type> void Tree<type>::InsertSubnodeBefore(type *node, type *before)
	{
		TreeBase::InsertSubnodeBefore(static_cast<Tree<type> *>(node), static_cast<Tree<type> *>(before));
	}

	template <class type> void Tree<type>::InsertSubnodeAfter(type *node, type *after)
	{
		TreeBase::InsertSubnodeAfter(static_cast<Tree<type> *>(node), static_cast<Tree<type> *>(after));
	}

	template <class type> void Tree<type>::RemoveSubnode(type *node)
	{
		TreeBase::RemoveSubnode(static_cast<Tree<type> *>(node));
	}
}


#endif
