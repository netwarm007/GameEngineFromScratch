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


#ifndef ODDLMap_h
#define ODDLMap_h


/*
	This file contains the associative map container classes used by the Tombstone Engine.
	The implementation is essentially an intrusive AVL tree with O(log n) operations.
*/


#include "ODDLTypes.h"


namespace ODDL
{
	class MapBase;
	class MapElementBase;
	template <class> class Map;


	struct MapReservation
	{
		MapElementBase		*superNode;
		int32				direction;
	};


	class MapElementBase
	{
		friend class MapBase;

		private:

			MapElementBase		*superNode;
			MapElementBase		*leftSubnode;
			MapElementBase		*rightSubnode;

			MapBase				*owningMap;
			int32				balance;

			MapElementBase *First(void);
			MapElementBase *Last(void);

			void RemoveSubtree(void);
			void PurgeSubtree(void);

		protected:

			MapElementBase()
			{
				superNode = nullptr;
				leftSubnode = nullptr;
				rightSubnode = nullptr;
				owningMap = nullptr;
			}

			virtual ~MapElementBase();

			MapElementBase *GetLeftSubnode(void) const
			{
				return (leftSubnode);
			}

			MapElementBase *GetRightSubnode(void) const
			{
				return (rightSubnode);
			}

			MapBase *GetOwningMap(void) const
			{
				return (owningMap);
			}

			MapElementBase *Previous(void) const;
			MapElementBase *Next(void) const;

		public:

			virtual void Detach(void);
	};


	class MapBase
	{
		friend class MapElementBase;

		private:

			MapElementBase		*rootNode;

			MapElementBase *RotateLeft(MapElementBase *node);
			MapElementBase *RotateRight(MapElementBase *node);
			MapElementBase *ZigZagLeft(MapElementBase *node);
			MapElementBase *ZigZagRight(MapElementBase *node);

			void RemoveBranchNode(MapElementBase *node, MapElementBase *subnode);

		protected:

			MapBase()
			{
				rootNode = nullptr;
			}

			~MapBase();

			MapElementBase *operator [](machine index) const;

			MapElementBase *GetRootNode(void) const
			{
				return (rootNode);
			}

			MapElementBase *First(void) const
			{
				return ((rootNode) ? rootNode->First() : nullptr);
			}

			MapElementBase *Last(void) const
			{
				return ((rootNode) ? rootNode->Last() : nullptr);
			}

			bool Member(const MapElementBase *element) const
			{
				return (element->owningMap == this);
			}

			void SetRootNode(MapElementBase *node);

			void InsertLeftSubnode(MapElementBase *node, MapElementBase *subnode);
			void InsertRightSubnode(MapElementBase *node, MapElementBase *subnode);

			void ReplaceNode(MapElementBase *node, MapElementBase *element);
			void RemoveNode(MapElementBase *node);

		public:

			bool Empty(void) const
			{
				return (!rootNode);
			}

			int32 GetElementCount(void) const;

			void RemoveAll(void);
			void Purge(void);
	};


	//# \class	MapElement		The base class for objects that can be stored in a map.
	//
	//# Objects inherit from the $MapElement$ class so that they can be stored in a map.
	//
	//# \def	template <class type> class MapElement : public MapElementBase
	//
	//# \tparam		type	The type of the class that can be stored in a map. This parameter should be the
	//#						type of the class that inherits directly from the $MapElement$ class.
	//
	//# \ctor	MapElement();
	//
	//# The constructor has protected access takes no parameters. The $MapElement$ class can only exist
	//# as a base class for another class.
	//
	//# \desc
	//# The $MapElement$ class should be declared as a base class for objects that need to be stored in a map.
	//# The $type$ template parameter should match the class type of such objects, and these objects can be
	//# stored in a $@Map@$ container declared with the same $type$ template parameter.
	//
	//# \privbase	MapElementBase		Used internally to encapsulate common functionality that is independent
	//#									of the template parameters.
	//
	//# \also	$@Map@$


	//# \function	MapElement::Previous		Returns the previous element in a map.
	//
	//# \proto	type *Previous(void) const;
	//
	//# \desc
	//# The $Previous$ function returns a pointer to the element immediately preceding an object in its
	//# owning map. If the object is the first element in a map, or the object does not belong to a
	//# map, then the return value is $nullptr$.
	//
	//# \also	$@MapElement::Next@$


	//# \function	MapElement::Next		Returns the next element in a map.
	//
	//# \proto	type *Next(void) const;
	//
	//# \desc
	//# The $Next$ function returns a pointer to the element immediately succeeding an object in its
	//# owning map. If the object is the last element in a map, or the object does not belong to a
	//# map, then the return value is $nullptr$.
	//
	//# \also	$@MapElement::Previous@$


	//# \function	MapElement::GetOwningMap		Returns the map to which an object belongs.
	//
	//# \proto	Map<type> *GetOwningMap(void) const;
	//
	//# \desc
	//# The $GetOwningMap$ function returns a pointer to the $@Map@$ container to which an object belongs.
	//# If the object is not a member of a map, then the return value is $nullptr$.
	//
	//# \also	$@Map::Member@$


	//# \function	MapElement::Detach		Removes an object from any map to which it belongs.
	//
	//# \proto	virtual void Detach(void);
	//
	//# \desc
	//# The $Detach$ function removes an object from its owning map. If the object is not a member of
	//# a map, then the $Detach$ function has no effect.
	//
	//# \also	$@Map::Remove@$


	template <class type> class MapElement : public MapElementBase
	{
		public:

			MapElement() = default;

			type *GetLeftSubnode(void) const
			{
				return (static_cast<type *>(static_cast<MapElement<type> *>(MapElementBase::GetLeftSubnode())));
			}

			type *GetRightSubnode(void) const
			{
				return (static_cast<type *>(static_cast<MapElement<type> *>(MapElementBase::GetRightSubnode())));
			}

			type *Previous(void) const
			{
				return (static_cast<type *>(static_cast<MapElement<type> *>(MapElementBase::Previous())));
			}

			type *Next(void) const
			{
				return (static_cast<type *>(static_cast<MapElement<type> *>(MapElementBase::Next())));
			}

			Map<type> *GetOwningMap(void) const
			{
				return (static_cast<Map<type> *>(MapElementBase::GetOwningMap()));
			}
	};


	//# \class	Map		An associative container class that holds a set of objects.
	//
	//# The $Map$ class encapsulates an associative key-value map.
	//
	//# \def	template <class type> class Map : public MapBase
	//
	//# \tparam		type	The type of the class that can be stored in the map. The class specified
	//#						by this parameter should inherit directly from the $@MapElement@$ class
	//#						using the same template parameter.
	//
	//# \ctor	Map();
	//
	//# \desc
	//# The $Map$ class template is a container used to store an associative key-value map of objects.
	//# The class type of objects that are to be stored in the map must be a subclass of the $@MapElement@$
	//# class template using the same template parameters as the $Map$ container.
	//#
	//# Upon construction, a $Map$ object is empty. When a $Map$ object is destroyed, all of the members
	//# of the map are also destroyed. To avoid deleting the members of a map when a $Map$ object is
	//# destroyed, first call the $@Map::RemoveAll@$ function to remove all of the map's members.
	//#
	//# The class specified by the $type$ template parameter must define a type named $KeyType$ and a
	//# function named $GetKey$ that has one of the following two prototypes.
	//
	//# \source
	//# KeyType GetKey(void) const;
	//# const KeyType& GetKey(void) const;
	//
	//# \desc
	//# This function should return the key associated with the object for which it is called. The $KeyType$
	//# type must be capable of being compared to other key values using the $<$ and $>$ operators.
	//
	//# \operator	type *operator [](machine index) const;
	//#				Returns the element of a map whose zero-based index is $index$. If $index$ is
	//#				greater than or equal to the number of elements in the list, then the return
	//#				value is $nullptr$.
	//
	//# \privbase	MapBase		Used internally to encapsulate common functionality that is independent
	//#							of the template parameters.
	//
	//# \also	$@MapElement@$


	//# \function	Map::First		Returns the first element in a map.
	//
	//# \proto	type *First(void) const;
	//
	//# \desc
	//# The $First$ function returns a pointer to the first element in a map. If the map is empty,
	//# then this function returns $nullptr$.
	//#
	//# The first element in a map is always the one having the least key value based on the
	//# ordering defined for the $KeyType$ type. The $@MapElement::Next@$ function can
	//# be repeated called to iterate through the members of a map in order.
	//
	//# \also	$@Map::Last@$


	//# \function	Map::Last		Returns the last element in a map.
	//
	//# \proto	type *Last(void) const;
	//
	//# \desc
	//# The $Last$ function returns a pointer to the last element in a map. If the map is empty,
	//# then this function returns $nullptr$.
	//#
	//# The last element in a map is always the one having the greatest key value based on the
	//# ordering defined for the $KeyType$ type. The $@MapElement::Previous@$ function can
	//# be repeated called to iterate through the members of a map in reverse order.
	//
	//# \also	$@Map::First@$


	//# \function	Map::Member		Returns a boolean value indicating whether a particular object is
	//#								a member of a map.
	//
	//# \proto	bool Member(const MapElement<type> *element) const;
	//
	//# \param	element		A pointer to the object to test for membership.
	//
	//# \desc
	//# The $Member$ function returns $true$ if the object specified by the $element$ parameter is
	//# a member of the map, and $false$ otherwise.
	//
	//# \also	$@MapElement::GetOwningMap@$


	//# \function	Map::Empty		Returns a boolean value indicating whether a map is empty.
	//
	//# \proto	bool Empty(void) const;
	//
	//# \desc
	//# The $Empty$ function returns $true$ if the map contains no elements, and $false$ otherwise.
	//
	//# \also	$@Map::First@$
	//# \also	$@Map::Last@$


	//# \function	Map::GetElementCount	Returns the number of elements in a map.
	//
	//# \proto	int32 GetElementCount(void) const;
	//
	//# \desc
	//# The $GetElementCount$ function iterates through the members of a map and returns the count.
	//
	//# \also	$@Map::Empty@$


	//# \function	Map::Insert		Adds an object to a map.
	//
	//# \proto	bool Insert(type *element);
	//
	//# \param	element		A pointer to the object to add to the map.
	//
	//# \desc
	//# The $Insert$ function adds the object specified by the $element$ parameter to a map.
	//# If the object is a member of a different map of the same type, then it is first removed from
	//# that map before being added to the new map.
	//#
	//# Only one object having a particular key value may be stored in a map at one time. If the key
	//# value associated with the $element$ parameter is not found in the map, then the object is
	//# inserted, and $true$ is returned. If a different object with the same key value is already
	//# in the map, then the new object is not inserted in the map and $false$ is returned.
	//
	//# \also	$@Map::InsertReplace@$
	//# \also	$@Map::Find@$


	//# \function	Map::InsertReplace		Adds an object to a map and replaces an existing object having the same key.
	//
	//# \proto	type *InsertReplace(type *element);
	//
	//# \param	element		A pointer to the object to add to the map.
	//
	//# \desc
	//# The $InsertReplace$ function adds the object specified by the $element$ parameter to a map.
	//# If an object having the same key is already a member of the map, then it is replaced by the new
	//# object being inserted. The old object is removed from the map, and a pointer to it is returned.
	//# If no object is replaced, then the return value is $nullptr$.
	//#
	//# If the object being inserted is a member of a different map of the same type, then it is first removed
	//# from that map before being added to the new map. If the object is already a member of the map for which
	//# the $InsertReplace$ function is called, then no action is taken, and the return value is $nullptr$.
	//
	//# \also	$@Map::Insert@$
	//# \also	$@Map::Find@$


	//# \function	Map::Remove		Removes a particular element from a map.
	//
	//# \proto	void Remove(MapElement<type> *element);
	//
	//# \param	element		A pointer to the object to remove from the map.
	//
	//# \desc
	//# The $Remove$ function removes the object specified by the $element$ parameter from a map.
	//# The object must belong to map for which the $Remove$ function is called, or the internal links will become corrupted.
	//
	//# \also	$@Map::RemoveAll@$
	//# \also	$@Map::Purge@$
	//# \also	$@MapElement::Detach@$


	//# \function	Map::RemoveAll		Removes all elements from a map.
	//
	//# \proto	void RemoveAll(void);
	//
	//# \desc
	//# The $RemoveAll$ function removes all objects contained in a map, but does not delete them.
	//# The map is subsequently empty.
	//
	//# \also	$@Map::Remove@$
	//# \also	$@Map::Purge@$
	//# \also	$@MapElement::Detach@$


	//# \function	Map::Purge		Deletes all elements in a map.
	//
	//# \proto	void Purge(void);
	//
	//# \desc
	//# The $Purge$ function deletes all objects contained in a map. The map is subsequently empty.
	//# To remove all elements of a map without destroying them, use the $@Map::RemoveAll@$ function.
	//
	//# \also	$@Map::Remove@$
	//# \also	$@Map::RemoveAll@$
	//# \also	$@MapElement::Detach@$


	//# \function	Map::Find		Finds an object in a map.
	//
	//# \proto	type *Find(const KeyType& key) const;
	//
	//# \desc
	//# The $Find$ function searches a map for an object having the key value given by the $key$ parameter.
	//# If a matching object is found, then a pointer to it is returned. If no matching object is found, then
	//# the return value is $nullptr$. This function is guaranteed to run in <i>O</i>(log&nbsp;<i>n</i>) time,
	//# where <i>n</i> is the number of objects stored in the map.
	//
	//# \also	$@Map::Insert@$
	//# \also	$@Map::InsertReplace@$
	//# \also	$@Map::Remove@$


	template <class type> class Map : public MapBase
	{
		public:

			typedef typename type::KeyType		KeyType;

			Map() = default;

			type *operator [](machine index) const
			{
				return (static_cast<type *>(static_cast<MapElement<type> *>(MapBase::operator [](index))));
			}

			type *First(void) const
			{
				return (static_cast<type *>(static_cast<MapElement<type> *>(MapBase::First())));
			}

			type *Last(void) const
			{
				return (static_cast<type *>(static_cast<MapElement<type> *>(MapBase::Last())));
			}

			type *GetRootNode(void) const
			{
				return (static_cast<type *>(static_cast<MapElement<type> *>(MapBase::GetRootNode())));
			}

			bool Member(const MapElement<type> *element) const
			{
				return (MapBase::Member(element));
			}

			void Remove(MapElement<type> *element)
			{
				RemoveNode(element);
			}

			bool Insert(MapElement<type> *element);
			type *InsertReplace(MapElement<type> *element);

			void Insert(MapElement<type> *element, const MapReservation *reservation);
			bool Reserve(const KeyType& key, MapReservation *reservation);

			type *Find(const KeyType& key) const;
	};


	template <class type> bool Map<type>::Insert(MapElement<type> *element)
	{
		MapElement<type> *node = GetRootNode();
		if (node)
		{
			const KeyType& key = static_cast<type *>(element)->GetKey();
			for (;;)
			{
				const KeyType& nodeKey = static_cast<type *>(node)->GetKey();
				if (key < nodeKey)
				{
					MapElement<type> *subnode = node->GetLeftSubnode();
					if (!subnode)
					{
						InsertLeftSubnode(node, element);
						break;
					}

					node = subnode;
				}
				else if (nodeKey < key)
				{
					MapElement<type> *subnode = node->GetRightSubnode();
					if (!subnode)
					{
						InsertRightSubnode(node, element);
						break;
					}

					node = subnode;
				}
				else
				{
					return (false);
				}
			}
		}
		else
		{
			SetRootNode(element);
		}

		return (true);
	}

	template <class type> type *Map<type>::InsertReplace(MapElement<type> *element)
	{
		MapElement<type> *node = GetRootNode();
		if (node)
		{
			const KeyType& key = static_cast<type *>(element)->GetKey();
			for (;;)
			{
				const KeyType& nodeKey = static_cast<type *>(node)->GetKey();
				if (key < nodeKey)
				{
					MapElement<type> *subnode = node->GetLeftSubnode();
					if (!subnode)
					{
						InsertLeftSubnode(node, element);
						break;
					}

					node = subnode;
				}
				else if (nodeKey < key)
				{
					MapElement<type> *subnode = node->GetRightSubnode();
					if (!subnode)
					{
						InsertRightSubnode(node, element);
						break;
					}

					node = subnode;
				}
				else
				{
					if (element != node)
					{
						ReplaceNode(node, element);
						return (static_cast<type *>(node));
					}

					break;
				}
			}
		}
		else
		{
			SetRootNode(element);
		}

		return (nullptr);
	}

	template <class type> void Map<type>::Insert(MapElement<type> *element, const MapReservation *reservation)
	{
		MapElementBase *node = reservation->superNode;
		if (node)
		{
			if (reservation->direction < 0)
			{
				InsertLeftSubnode(node, element);
			}
			else
			{
				InsertRightSubnode(node, element);
			}
		}
		else
		{
			SetRootNode(element);
		}
	}

	template <class type> bool Map<type>::Reserve(const KeyType& key, MapReservation *reservation)
	{
		MapElement<type> *node = GetRootNode();
		if (node)
		{
			for (;;)
			{
				const KeyType& nodeKey = static_cast<type *>(node)->GetKey();
				if (key < nodeKey)
				{
					MapElement<type> *subnode = node->GetLeftSubnode();
					if (!subnode)
					{
						reservation->superNode = node;
						reservation->direction = -1;
						break;
					}

					node = subnode;
				}
				else if (nodeKey < key)
				{
					MapElement<type> *subnode = node->GetRightSubnode();
					if (!subnode)
					{
						reservation->superNode = node;
						reservation->direction = 1;
						break;
					}

					node = subnode;
				}
				else
				{
					return (false);
				}
			}
		}
		else
		{
			reservation->superNode = nullptr;
		}

		return (true);
	}

	template <class type> type *Map<type>::Find(const KeyType& key) const
	{
		MapElement<type> *node = GetRootNode();
		while (node)
		{
			const KeyType& nodeKey = static_cast<type *>(node)->GetKey();
			if (key < nodeKey)
			{
				node = node->GetLeftSubnode();
			}
			else if (nodeKey < key)
			{
				node = node->GetRightSubnode();
			}
			else
			{
				break;
			}
		}

		return (static_cast<type *>(node));
	}
}


#endif
