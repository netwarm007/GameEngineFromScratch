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


#ifndef ODDLString_h
#define ODDLString_h


/*
	This file contains a simplified version of the string class used by the Tombstone Engine.
*/


#include "ODDLTypes.h"


namespace ODDL
{
	namespace Text
	{
		int32 WriteGlyphCodeUTF8(char *text, unsigned_int32 code);
		int32 ValidateGlyphCodeUTF8(const char *text);

		int32 GetTextLength(const char *text);
		int32 CopyText(const char *source, char *dest);
		int32 CopyText(const char *source, char *dest, int32 max);

		bool CompareText(const char *s1, const char *s2);
		bool CompareText(const char *s1, const char *s2, int32 max);
		bool CompareTextCaseless(const char *s1, const char *s2);
		bool CompareTextLessThan(const char *s1, const char *s2);
		bool CompareTextLessThanCaseless(const char *s1, const char *s2);
		bool CompareTextLessEqual(const char *s1, const char *s2);
		bool CompareTextLessEqualCaseless(const char *s1, const char *s2);
	}


	class String
	{
		private:

			enum
			{
				kStringAllocSize = 63
			};

			int32		logicalSize;
			int32		physicalSize;
			char		*stringPointer;

			static char		emptyString[1];

			String(const char *s1, const char *s2);

			static unsigned_int32 GetPhysicalSize(unsigned_int32 size)
			{
				return ((size + (kStringAllocSize + 4)) & ~kStringAllocSize);
			}

			void Resize(int32 size);

		public:

			String();
			~String();

			String(String&& s)
			{
				logicalSize = s.logicalSize;
				physicalSize = s.physicalSize;
				stringPointer = s.stringPointer;

				s.stringPointer = emptyString;
			}

			String(const String& s);
			String(const char *s);
			String(const char *s, int32 length);

			operator char *(void)
			{
				return (stringPointer);
			}

			operator const char *(void) const
			{
				return (stringPointer);
			}

			bool operator ==(const char *s) const
			{
				return (Text::CompareTextCaseless(stringPointer, s));
			}

			bool operator !=(const char *s) const
			{
				return (!Text::CompareTextCaseless(stringPointer, s));
			}

			bool operator <(const char *s) const
			{
				return (Text::CompareTextLessThanCaseless(stringPointer, s));
			}

			bool operator >=(const char *s) const
			{
				return (!Text::CompareTextLessThanCaseless(stringPointer, s));
			}

			bool operator <=(const char *s) const
			{
				return (Text::CompareTextLessEqualCaseless(stringPointer, s));
			}

			bool operator >(const char *s) const
			{
				return (!Text::CompareTextLessEqualCaseless(stringPointer, s));
			}

			String operator +(const char *s) const
			{
				return (String(stringPointer, s));
			}

			int32 Length(void) const
			{
				return (logicalSize - 1);
			}

			void Purge(void);
			String& Set(const char *s, int32 length);

			String& operator =(String&& s);
			String& operator =(const String& s);
			String& operator =(const char *s);
			String& operator +=(const String& s);
			String& operator +=(const char *s);
			String& operator +=(char k);

			String& SetLength(int32 length);
	};


	class ConstCharKey
	{
		private:

			const char		*ptr;

		public:

			ConstCharKey() = default;

			ConstCharKey(const char *c)
			{
				ptr = c;
			}

			ConstCharKey(const String& s)
			{
				ptr = s;
			}

			operator const char *(void) const
			{
				return (ptr);
			}

			ConstCharKey& operator =(const char *c)
			{
				ptr = c;
				return (*this);
			}

			bool operator ==(const char *c) const
			{
				return (Text::CompareText(ptr, c));
			}

			bool operator !=(const char *c) const
			{
				return (!Text::CompareText(ptr, c));
			}

			bool operator <(const char *c) const
			{
				return (Text::CompareTextLessThan(ptr, c));
			}
	};
}


#endif
