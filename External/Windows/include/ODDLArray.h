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


#ifndef ODDLArray_h
#define ODDLArray_h


/*
	This file contains the array container classes used by the Tombstone Engine.
*/


#include "ODDLTypes.h"


namespace ODDL
{
	//# \class	Array	A container class that holds an array of objects.
	//
	//# The $Array$ class represents a dynamically resizable array of objects
	//# for which any entry can be accessed in constant time.
	//
	//# \def	template <typename type, int32 baseCount = 0> class Array final : public ImmutableArray<type>
	//
	//# \tparam		type			The type of the class that can be stored in the array.
	//# \tparam		baseCount		The minimum number of array elements for which storage is available inside the $Array$ object itself.
	//
	//# \ctor	explicit Array(int32 count = 0);
	//
	//# \param	count	The number of array elements for which space is initially reserved in the array's storage.
	//
	//# \desc
	//# The $Array$ class represents a homogeneous array of objects whose type is given by the
	//# $type$ template parameter. Upon construction, the initial size of the array is zero, but
	//# space is reserved for the number of objects given by the $count$ parameter. The array is
	//# stored contiguously in memory, allowing constant-time random access to its elements.
	//#
	//# As elements are added to the array (using the $@Array::AddElement@$ function), the storage
	//# size is automatically increased to a size somewhat larger than that needed to store the new
	//# element. The cost of adding an element is thus amortized linear time.
	//#
	//# If the $baseCount$ template parameter is zero (the default), then storage space for the array
	//# elements is always allocated on the heap separately from the $Array$ object. If the value of
	//# $baseCount$ is greater than zero, then space for that number of array elements is built into the
	//# structure of the $Array$ object so that no separate allocations need to be made until the size
	//# of the array exceeds the value of $baseCount$.
	//#
	//# The $count$ parameter can only be specified if the $baseCount$ template parameter is zero.
	//#
	//# An $Array$ object can be implicitly converted to a pointer to its first element. This allows the
	//# use of the $[]$ operator to access individual elements of the array.
	//
	//# \privbase	ImmutableArray<type>		Used internally.


	//# \function	Array::GetElementCount		Returns the current size of an array.
	//
	//# \proto	int32 GetElementCount(void) const;
	//
	//# \desc
	//# The $GetElementCount$ function returns the number of objects currently stored in an array.
	//# When an array is constructed, its initial element count is zero.
	//
	//# \also	$@Array::SetElementCount@$
	//# \also	$@Array::AddElement@$
	//# \also	$@Array::InsertElement@$
	//# \also	$@Array::RemoveElement@$


	//# \function	Array::SetElementCount		Sets the current size of an array.
	//
	//# \proto	void SetElementCount(int32 count, const type *init = nullptr);
	//
	//# \param	count	The new size of the array.
	//# \param	init	A pointer to an object that is used to copy-construct new objects in the array.
	//
	//# \desc
	//# The $SetElementCount$ function sets the number of objects currently stored in an array.
	//# If $count$ is greater than the current size of the array, then space is allocated for
	//# $count$ objects and each new object is copy-constructed from the value of the $init$ parameter.
	//# If $count$ is less than the current size of the array, then the logical size of the array
	//# is reduced, and each object beyond the new size of the array is destroyed in reverse order.
	//#
	//# If the $init$ parameter is $nullptr$, then any new objects created are default-constructed if the
	//# type of object stored in the array is a non-POD type. If the type of object stored in the array is
	//# a POD type, then any new objects created are left uninitialized.
	//
	//# \also	$@Array::GetElementCount@$
	//# \also	$@Array::AddElement@$
	//# \also	$@Array::InsertElement@$
	//# \also	$@Array::RemoveElement@$


	//# \function	Array::AddElement		Adds an object to the end of an array.
	//
	//# \proto	template <typename T> void AddElement(T&& element);
	//
	//# \param	element		The new element to add to the array.
	//
	//# \desc
	//# The $AddElement$ function increases the size of an array by one and either copy-constructs
	//# or move-constructs the new element using the object referenced by the $element$ parameter,
	//# depending on whether an lvalue reference or rvalue reference is passed to the function.
	//
	//# \also	$@Array::InsertElement@$
	//# \also	$@Array::RemoveElement@$
	//# \also	$@Array::GetElementCount@$
	//# \also	$@Array::SetElementCount@$


	//# \function	Array::InsertElement	Inserts an object into an array.
	//
	//# \proto	template <typename T> void InsertElement(int32 index, T&& element);
	//
	//# \param	index		The location at which the object is to be inserted.
	//# \param	element		The new element to insert into the array.
	//
	//# \desc
	//# The $InsertElement$ function increases the size of an array by one, moves all of the existing
	//# elements at location $index$ or greater up by one, and either copy-constructs or move-constructs
	//# the new element into the array using the object referenced by the $element$ parameter, depending on
	//# whether an lvalue reference or rvalue reference is passed to the function. When the existing elements
	//# are moved, they are move-constructed in their new locations, and the old objects are destroyed.
	//#
	//# If the $index$ parameter is greater than or equal to the current size of the array, then the
	//# array is enlarged to the size $index&nbsp;+&nbsp;1$. In this case, elements between the old size and
	//# new size are default-constructed if the type of object stored in the array is a non-POD type, and the
	//# elements are left uninitialized if the type of object stored in the array is a POD type.
	//
	//# \also	$@Array::RemoveElement@$
	//# \also	$@Array::AddElement@$
	//# \also	$@Array::GetElementCount@$
	//# \also	$@Array::SetElementCount@$


	//# \function	Array::RemoveElement	Removes an object from an array.
	//
	//# \proto	void RemoveElement(int32 index);
	//
	//# \param	index	The location at which to remove an object.
	//
	//# \desc
	//# The $RemoveElement$ function decreases the size of an array by one, destroys the object at location
	//# $index$, and moves all of the existing elements at location $index&nbsp;+&nbsp;1$ or greater down by one.
	//# When the existing elements are moved, they are move-constructed to their new locations, and the old
	//# objects are destroyed.
	//#
	//# If the $index$ parameter is greater than or equal to the current size of the array, then
	//# calling the $RemoveElement$ function has no effect.
	//
	//# \also	$@Array::InsertElement@$
	//# \also	$@Array::AddElement@$
	//# \also	$@Array::GetElementCount@$
	//# \also	$@Array::SetElementCount@$


	//# \function	Array::Clear		Removes all objects from an array.
	//
	//# \proto	void Clear(void);
	//
	//# \desc
	//# The $Clear$ function destroys all objects in an array (in reverse order) and sets the size of
	//# the array to zero. The storage for the array is not deallocated, so this function is best used
	//# when the array is likely to be filled with a similar amount of data again. To both destroy all
	//# objects in an array and deallocate the storage, call the $@Array::Purge@$ function.
	//
	//# \also	$@Array::Purge@$
	//# \also	$@Array::RemoveElement@$
	//# \also	$@Array::SetElementCount@$


	//# \function	Array::Purge		Removes all objects from an array and deallocates storage.
	//
	//# \proto	void Purge(void);
	//
	//# \desc
	//# The $Purge$ function destroys all objects in an array (in reverse order) and sets the size of
	//# the array to zero. The storage for the array is also deallocated, returning the array to its
	//# initial state. To destory all objects in an array without deallocating the storage, call the
	//# $@Array::Clear@$ function.
	//
	//# \also	$@Array::Clear@$
	//# \also	$@Array::RemoveElement@$
	//# \also	$@Array::SetElementCount@$


	//# \function	Array::FindElement		Finds a specific element in an array.
	//
	//# \proto	int32 FindElement(const type& element) const;
	//
	//# \param	element		The value of the element to find.
	//
	//# \desc
	//# The $FindElement$ function searches an array for the first element matching the value passed into the
	//# $element$ parameter based on the $==$ operator. If a match is found, its index is returned.
	//# If no match is found, then the return value is &minus;1. The running time of this function is
	//# <i>O</i>(<i>n</i>), where <i>n</i> is the number of elements in the array.


	template <typename type> class ImmutableArray
	{
		protected:

			int32		elementCount;
			int32		reservedCount;

			type		*arrayPointer;

			ImmutableArray() = default;
			ImmutableArray(const ImmutableArray& array) {}
			~ImmutableArray() = default;

		public:

			operator type *(void)
			{
				return (arrayPointer);
			}

			operator type *(void) const
			{
				return (arrayPointer);
			}

			type *begin(void) const
			{
				return (arrayPointer);
			}

			type *end(void) const
			{
				return (arrayPointer + elementCount);
			}

			int32 GetElementCount(void) const
			{
				return (elementCount);
			}

			bool Empty(void) const
			{
				return (elementCount == 0);
			}

			int32 FindElement(const type& element) const;
	};

	template <typename type> int32 ImmutableArray<type>::FindElement(const type& element) const
	{
		for (int32 a = 0; a < elementCount; a++)
		{
			if (arrayPointer[a] == element)
			{
				return (a);
			}
		}

		return (-1);
	}


	template <typename type, int32 baseCount = 0> class Array final : public ImmutableArray<type>
	{
		private:

			using ImmutableArray<type>::elementCount;
			using ImmutableArray<type>::reservedCount;
			using ImmutableArray<type>::arrayPointer;

			char		arrayStorage[baseCount * sizeof(type)];

			void SetReservedCount(int32 count);

		public:

			explicit Array();
			Array(const Array& array);
			Array(Array&& array);
			~Array();

			void Clear(void);
			void Purge(void);
			void Reserve(int32 count);

			void SetElementCount(int32 count, const type *init = nullptr);
			type *AddElement(void);

			template <typename T> void AddElement(T&& element);
			template <typename T> void InsertElement(int32 index, T&& element);

			void RemoveElement(int32 index);
	};


	template <typename type, int32 baseCount> Array<type, baseCount>::Array()
	{
		elementCount = 0;
		reservedCount = baseCount;
		arrayPointer = reinterpret_cast<type *>(arrayStorage);
	}

	template <typename type, int32 baseCount> Array<type, baseCount>::Array(const Array& array)
	{
		elementCount = array.elementCount;
		reservedCount = array.reservedCount;

		if (elementCount > baseCount)
		{
			arrayPointer = reinterpret_cast<type *>(new char[sizeof(type) * reservedCount]);
		}
		else
		{
			arrayPointer = reinterpret_cast<type *>(arrayStorage);
		}

		for (machine a = 0; a < elementCount; a++)
		{
			new(&arrayPointer[a]) type(array.arrayPointer[a]);
		}
	}

	template <typename type, int32 baseCount> Array<type, baseCount>::Array(Array&& array)
	{
		elementCount = array.elementCount;
		reservedCount = array.reservedCount;

		if (elementCount > baseCount)
		{
			arrayPointer = array.arrayPointer;
		}
		else
		{
			arrayPointer = reinterpret_cast<type *>(arrayStorage);

			type *pointer = array.arrayPointer;
			for (machine a = 0; a < elementCount; a++)
			{
				new(&arrayPointer[a]) type(static_cast<type&&>(pointer[a]));
				pointer[a].~type();
			}
		}

		array.elementCount = 0;
		array.reservedCount = baseCount;
		array.arrayPointer = reinterpret_cast<type *>(array.arrayStorage);
	}

	template <typename type, int32 baseCount> Array<type, baseCount>::~Array()
	{
		type *pointer = arrayPointer + elementCount;
		for (machine a = elementCount - 1; a >= 0; a--)
		{
			(--pointer)->~type();
		}

		char *ptr = reinterpret_cast<char *>(arrayPointer);
		if (ptr != arrayStorage)
		{
			delete[] ptr;
		}
	}

	template <typename type, int32 baseCount> void Array<type, baseCount>::Clear(void)
	{
		type *pointer = arrayPointer + elementCount;
		for (machine a = elementCount - 1; a >= 0; a--)
		{
			(--pointer)->~type();
		}

		elementCount = 0;
	}

	template <typename type, int32 baseCount> void Array<type, baseCount>::Purge(void)
	{
		type *pointer = arrayPointer + elementCount;
		for (machine a = elementCount - 1; a >= 0; a--)
		{
			(--pointer)->~type();
		}

		char *ptr = reinterpret_cast<char *>(arrayPointer);
		if (ptr != arrayStorage)
		{
			delete[] ptr;
		}

		elementCount = 0;
		reservedCount = baseCount;
		arrayPointer = reinterpret_cast<type *>(arrayStorage);
	}

	template <typename type, int32 baseCount> void Array<type, baseCount>::SetReservedCount(int32 count)
	{
		reservedCount = Max(Max(count, 4), reservedCount + Max((reservedCount / 2 + 3) & ~3, baseCount));
		type *newPointer = reinterpret_cast<type *>(new char[sizeof(type) * reservedCount]);

		type *pointer = arrayPointer;
		for (machine a = 0; a < elementCount; a++)
		{
			new(&newPointer[a]) type(static_cast<type&&>(*pointer));
			pointer->~type();
			pointer++;
		}

		char *ptr = reinterpret_cast<char *>(arrayPointer);
		if (ptr != arrayStorage)
		{
			delete[] ptr;
		}

		arrayPointer = newPointer;
	}

	template <typename type, int32 baseCount> void Array<type, baseCount>::Reserve(int32 count)
	{
		if (count > reservedCount)
		{
			SetReservedCount(count);
		}
	}

	template <typename type, int32 baseCount> void Array<type, baseCount>::SetElementCount(int32 count, const type *init)
	{
		if (count > reservedCount)
		{
			SetReservedCount(count);
		}

		if (count > elementCount)
		{
			type *pointer = arrayPointer + (elementCount - 1);
			if (init)
			{
				for (machine a = elementCount; a < count; a++)
				{
					new(++pointer) type(*init);
				}
			}
			else
			{
				for (machine a = elementCount; a < count; a++)
				{
					new(++pointer) type;
				}
			}
		}
		else if (count < elementCount)
		{
			type *pointer = arrayPointer + elementCount;
			for (machine a = elementCount - 1; a >= count; a--)
			{
				(--pointer)->~type();
			}
		}

		elementCount = count;
	}

	template <typename type, int32 baseCount> type *Array<type, baseCount>::AddElement(void)
	{
		if (elementCount >= reservedCount)
		{
			SetReservedCount(elementCount + 1);
		}

		type *pointer = arrayPointer + elementCount;
		new(pointer) type;

		elementCount++;
		return (pointer);
	}

	template <typename type, int32 baseCount> template <typename T> void Array<type, baseCount>::AddElement(T&& element)
	{
		if (elementCount >= reservedCount)
		{
			SetReservedCount(elementCount + 1);
		}

		type *pointer = arrayPointer + elementCount;
		new(pointer) type(static_cast<T&&>(element));

		elementCount++;
	}

	template <typename type, int32 baseCount> template <typename T> void Array<type, baseCount>::InsertElement(int32 index, T&& element)
	{
		if (index >= elementCount)
		{
			int32 count = index + 1;
			if (count > reservedCount)
			{
				SetReservedCount(count);
			}

			type *pointer = &arrayPointer[elementCount - 1];
			for (machine a = elementCount; a < index; a++)
			{
				new(++pointer) type;
			}

			new (++pointer) type(static_cast<T&&>(element));
			elementCount = count;
		}
		else
		{
			int32 count = elementCount + 1;
			if (count > reservedCount)
			{
				SetReservedCount(count);
			}

			type *pointer = &arrayPointer[elementCount];
			for (machine a = elementCount; a > index; a--)
			{
				new(pointer) type(static_cast<type&&>(pointer[-1]));
				(--pointer)->~type();
			}

			new (&arrayPointer[index]) type(static_cast<T&&>(element));
			elementCount = count;
		}
	}

	template <typename type, int32 baseCount> void Array<type, baseCount>::RemoveElement(int32 index)
	{
		if (index < elementCount)
		{
			type *pointer = &arrayPointer[index];
			pointer->~type();

			for (machine a = index + 1; a < elementCount; a++)
			{
				new(pointer) type(static_cast<type&&>(pointer[1]));
				(++pointer)->~type();
			}

			elementCount--;
		}
	}


	template <typename type> class Array<type, 0> final : public ImmutableArray<type>
	{
		private:

			using ImmutableArray<type>::elementCount;
			using ImmutableArray<type>::reservedCount;
			using ImmutableArray<type>::arrayPointer;

			void SetReservedCount(int32 count);

		public:

			explicit Array(int32 count = 0);
			Array(const Array& array);
			Array(Array&& array);
			~Array();

			void Clear(void);
			void Purge(void);
			void Reserve(int32 count);

			void SetElementCount(int32 count, const type *init = nullptr);
			type *AddElement(void);

			template <typename T> void AddElement(T&& element);
			template <typename T> void InsertElement(int32 index, T&& element);

			void RemoveElement(int32 index);
	};


	template <typename type> Array<type, 0>::Array(int32 count)
	{
		elementCount = 0;
		reservedCount = count;

		arrayPointer = (count > 0) ? reinterpret_cast<type *>(new char[sizeof(type) * count]) : nullptr;
	}

	template <typename type> Array<type, 0>::Array(const Array& array)
	{
		elementCount = array.elementCount;
		reservedCount = array.reservedCount;

		if (reservedCount > 0)
		{
			arrayPointer = reinterpret_cast<type *>(new char[sizeof(type) * reservedCount]);
			for (machine a = 0; a < elementCount; a++)
			{
				new(&arrayPointer[a]) type(array.arrayPointer[a]);
			}
		}
		else
		{
			arrayPointer = nullptr;
		}
	}

	template <typename type> Array<type, 0>::Array(Array&& array)
	{
		elementCount = array.elementCount;
		reservedCount = array.reservedCount;
		arrayPointer = array.arrayPointer;

		array.elementCount = 0;
		array.reservedCount = 0;
		array.arrayPointer = nullptr;
	}

	template <typename type> Array<type, 0>::~Array()
	{
		type *pointer = arrayPointer + elementCount;
		for (machine a = elementCount - 1; a >= 0; a--)
		{
			(--pointer)->~type();
		}

		delete[] reinterpret_cast<char *>(arrayPointer);
	}

	template <typename type> void Array<type, 0>::Clear(void)
	{
		type *pointer = arrayPointer + elementCount;
		for (machine a = elementCount - 1; a >= 0; a--)
		{
			(--pointer)->~type();
		}

		elementCount = 0;
	}

	template <typename type> void Array<type, 0>::Purge(void)
	{
		type *pointer = arrayPointer + elementCount;
		for (machine a = elementCount - 1; a >= 0; a--)
		{
			(--pointer)->~type();
		}

		delete[] reinterpret_cast<char *>(arrayPointer);

		elementCount = 0;
		reservedCount = 0;
		arrayPointer = nullptr;
	}

	template <typename type> void Array<type, 0>::SetReservedCount(int32 count)
	{
		reservedCount = Max(Max(count, 4), reservedCount + Max((reservedCount / 2 + 3) & ~3, 4));
		type *newPointer = reinterpret_cast<type *>(new char[sizeof(type) * reservedCount]);

		type *pointer = arrayPointer;
		if (pointer)
		{
			for (machine a = 0; a < elementCount; a++)
			{
				new(&newPointer[a]) type(static_cast<type&&>(*pointer));
				pointer->~type();
				pointer++;
			}

			delete[] reinterpret_cast<char *>(arrayPointer);
		}

		arrayPointer = newPointer;
	}

	template <typename type> void Array<type, 0>::Reserve(int32 count)
	{
		if (count > reservedCount)
		{
			SetReservedCount(count);
		}
	}

	template <typename type> void Array<type, 0>::SetElementCount(int32 count, const type *init)
	{
		if (count > reservedCount)
		{
			SetReservedCount(count);
		}

		if (count > elementCount)
		{
			type *pointer = arrayPointer + (elementCount - 1);
			if (init)
			{
				for (machine a = elementCount; a < count; a++)
				{
					new(++pointer) type(*init);
				}
			}
			else
			{
				for (machine a = elementCount; a < count; a++)
				{
					new(++pointer) type;
				}
			}
		}
		else if (count < elementCount)
		{
			type *pointer = arrayPointer + elementCount;
			for (machine a = elementCount - 1; a >= count; a--)
			{
				(--pointer)->~type();
			}
		}

		elementCount = count;
	}

	template <typename type> type *Array<type, 0>::AddElement(void)
	{
		if (elementCount >= reservedCount)
		{
			SetReservedCount(elementCount + 1);
		}

		type *pointer = arrayPointer + elementCount;
		new(pointer) type;

		elementCount++;
		return (pointer);
	}

	template <typename type> template <typename T> void Array<type, 0>::AddElement(T&& element)
	{
		if (elementCount >= reservedCount)
		{
			SetReservedCount(elementCount + 1);
		}

		type *pointer = arrayPointer + elementCount;
		new(pointer) type(static_cast<T&&>(element));

		elementCount++;
	}

	template <typename type> template <typename T> void Array<type, 0>::InsertElement(int32 index, T&& element)
	{
		if (index >= elementCount)
		{
			int32 count = index + 1;
			if (count > reservedCount)
			{
				SetReservedCount(count);
			}

			type *pointer = &arrayPointer[elementCount - 1];
			for (machine a = elementCount; a < index; a++)
			{
				new(++pointer) type;
			}

			new (++pointer) type(static_cast<T&&>(element));
			elementCount = count;
		}
		else
		{
			int32 count = elementCount + 1;
			if (count > reservedCount)
			{
				SetReservedCount(count);
			}

			type *pointer = &arrayPointer[elementCount];
			for (machine a = elementCount; a > index; a--)
			{
				new(pointer) type(static_cast<type&&>(pointer[-1]));
				(--pointer)->~type();
			}

			new (&arrayPointer[index]) type(static_cast<T&&>(element));
			elementCount = count;
		}
	}

	template <typename type> void Array<type, 0>::RemoveElement(int32 index)
	{
		if (index < elementCount)
		{
			type *pointer = &arrayPointer[index];
			pointer->~type();

			for (machine a = index + 1; a < elementCount; a++)
			{
				new(pointer) type(static_cast<type&&>(pointer[1]));
				(++pointer)->~type();
			}

			elementCount--;
		}
	}
}


#endif
