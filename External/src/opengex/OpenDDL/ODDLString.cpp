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


#include "ODDLString.h"


using namespace ODDL;


char String::emptyString[1] = "";


int32 Text::WriteGlyphCodeUTF8(char *text, unsigned_int32 code)
{
	if (code <= 0x00007F)
	{
		text[0] = (char) code;
		return (1);
	}

	if (code <= 0x0007FF)
	{
		text[0] = (char) (((code >> 6) & 0x1F) | 0xC0);
		text[1] = (char) ((code & 0x3F) | 0x80);
		return (2);
	}

	if (code <= 0x00FFFF)
	{
		text[0] = (char) (((code >> 12) & 0x0F) | 0xE0);
		text[1] = (char) (((code >> 6) & 0x3F) | 0x80);
		text[2] = (char) ((code & 0x3F) | 0x80);
		return (3);
	}

	if (code <= 0x10FFFF)
	{
		text[0] = (char) (((code >> 18) & 0x07) | 0xF0);
		text[1] = (char) (((code >> 12) & 0x3F) | 0x80);
		text[2] = (char) (((code >> 6) & 0x3F) | 0x80);
		text[3] = (char) ((code & 0x3F) | 0x80);
		return (4);
	}

	return (0);
}

int32 Text::ValidateGlyphCodeUTF8(const char *text)
{
	int32 c = text[0];
	if (c >= 0)
	{
		return (1);
	}

	unsigned_int32 byte1 = c & 0xFF;
	if (byte1 - 0xC0 < 0x38)
	{
		unsigned_int32 byte2 = reinterpret_cast<const unsigned_int8 *>(text)[1];
		if ((byte2 & 0xC0) != 0x80)
		{
			return (0);
		}

		if (byte1 < 0xE0)
		{
			return (2);
		}

		unsigned_int32 byte3 = reinterpret_cast<const unsigned_int8 *>(text)[2];
		if ((byte3 & 0xC0) != 0x80)
		{
			return (0);
		}

		if (byte1 < 0xF0)
		{
			return (3);
		}

		unsigned_int32 byte4 = reinterpret_cast<const unsigned_int8 *>(text)[2];
		if ((byte4 & 0xC0) != 0x80)
		{
			return (0);
		}

		return (4);
	}

	return (0);
}

int32 Text::GetTextLength(const char *text)
{
	const char *start = text;
	while (*text != 0)
	{
		text++;
	}

	return ((int32) (text - start));
}

int32 Text::CopyText(const char *source, char *dest)
{
	const char *c = source;
	for (;;)
	{
		unsigned_int32 k = *reinterpret_cast<const unsigned_int8 *>(c);
		*dest++ = (char) k;
		if (k == 0)
		{
			break;
		}

		c++;
	}

	return ((int32) (c - source));
}

int32 Text::CopyText(const char *source, char *dest, int32 max)
{
	const char *c = source;
	while (--max >= 0)
	{
		unsigned_int32 k = *reinterpret_cast<const unsigned_int8 *>(c);
		if (k == 0)
		{
			break;
		}

		*dest++ = (char) k;
		c++;
	}

	dest[0] = 0;
	return ((int32) (c - source));
}

bool Text::CompareText(const char *s1, const char *s2)
{
	for (machine a = 0;; a++)
	{
		unsigned_int32 x = *reinterpret_cast<const unsigned_int8 *>(s1 + a);
		unsigned_int32 y = *reinterpret_cast<const unsigned_int8 *>(s2 + a);

		if (x != y)
		{
			return (false);
		}

		if (x == 0)
		{
			break;
		}
	}

	return (true);
}

bool Text::CompareText(const char *s1, const char *s2, int32 max)
{
	for (machine a = 0;; a++)
	{
		if (--max < 0)
		{
			break;
		}

		unsigned_int32 x = *reinterpret_cast<const unsigned_int8 *>(s1 + a);
		unsigned_int32 y = *reinterpret_cast<const unsigned_int8 *>(s2 + a);

		if (x != y)
		{
			return (false);
		}

		if (x == 0)
		{
			break;
		}
	}

	return (true);
}

bool Text::CompareTextCaseless(const char *s1, const char *s2)
{
	for (machine a = 0;; a++)
	{
		unsigned_int32 x = *reinterpret_cast<const unsigned_int8 *>(s1 + a);
		unsigned_int32 y = *reinterpret_cast<const unsigned_int8 *>(s2 + a);

		if (x - 'A' < 26U)
		{
			x += 32;
		}

		if (y - 'A' < 26U)
		{
			y += 32;
		}

		if (x != y)
		{
			return (false);
		}

		if (x == 0)
		{
			break;
		}
	}

	return (true);
}

bool Text::CompareTextLessThan(const char *s1, const char *s2)
{
	for (machine a = 0;; a++)
	{
		unsigned_int32 x = *reinterpret_cast<const unsigned_int8 *>(s1 + a);
		unsigned_int32 y = *reinterpret_cast<const unsigned_int8 *>(s2 + a);

		if ((x != y) || (x == 0))
		{
			return (x < y);
		}
	}
}

bool Text::CompareTextLessThanCaseless(const char *s1, const char *s2)
{
	for (machine a = 0;; a++)
	{
		unsigned_int32 x = *reinterpret_cast<const unsigned_int8 *>(s1 + a);
		unsigned_int32 y = *reinterpret_cast<const unsigned_int8 *>(s2 + a);

		if (x - 'a' < 26U)
		{
			x -= 32;
		}

		if (y - 'a' < 26U)
		{
			y -= 32;
		}

		if ((x != y) || (x == 0))
		{
			return (x < y);
		}
	}
}

bool Text::CompareTextLessEqual(const char *s1, const char *s2)
{
	for (machine a = 0;; a++)
	{
		unsigned_int32 x = *reinterpret_cast<const unsigned_int8 *>(s1 + a);
		unsigned_int32 y = *reinterpret_cast<const unsigned_int8 *>(s2 + a);

		if ((x != y) || (x == 0))
		{
			return (x <= y);
		}
	}
}

bool Text::CompareTextLessEqualCaseless(const char *s1, const char *s2)
{
	for (machine a = 0;; a++)
	{
		unsigned_int32 x = *reinterpret_cast<const unsigned_int8 *>(s1 + a);
		unsigned_int32 y = *reinterpret_cast<const unsigned_int8 *>(s2 + a);

		if (x - 'a' < 26U)
		{
			x -= 32;
		}

		if (y - 'a' < 26U)
		{
			y -= 32;
		}

		if ((x != y) || (x == 0))
		{
			return (x <= y);
		}
	}
}


String::String()
{
	logicalSize = 1;
	physicalSize = 0;
	stringPointer = emptyString;
}

String::~String()
{
	if (stringPointer != emptyString)
	{
		delete[] stringPointer;
	}
}

String::String(const String& s)
{
	int32 size = s.logicalSize;
	logicalSize = size;
	if (size > 1)
	{
		physicalSize = GetPhysicalSize(size);
		stringPointer = new char[physicalSize];
		Text::CopyText(s, stringPointer);
	}
	else
	{
		physicalSize = 0;
		stringPointer = emptyString;
	}
}

String::String(const char *s)
{
	int32 size = Text::GetTextLength(s) + 1;
	logicalSize = size;
	if (size > 1)
	{
		physicalSize = GetPhysicalSize(size);
		stringPointer = new char[physicalSize];
		Text::CopyText(s, stringPointer);
	}
	else
	{
		physicalSize = 0;
		stringPointer = emptyString;
	}
}

String::String(const char *s, int32 length)
{
	length = Min(length, Text::GetTextLength(s));

	int32 size = length + 1;
	logicalSize = size;
	if (size > 1)
	{
		physicalSize = GetPhysicalSize(size);
		stringPointer = new char[physicalSize];
		Text::CopyText(s, stringPointer, length);
	}
	else
	{
		physicalSize = 0;
		stringPointer = emptyString;
	}
}

String::String(const char *s1, const char *s2)
{
	int32 len1 = Text::GetTextLength(s1);
	int32 len2 = Text::GetTextLength(s2);

	int32 size = len1 + len2 + 1;
	logicalSize = size;
	if (size > 1)
	{
		physicalSize = GetPhysicalSize(size);
		stringPointer = new char[physicalSize];
		Text::CopyText(s1, stringPointer);
		Text::CopyText(s2, stringPointer + len1);
	}
	else
	{
		physicalSize = 0;
		stringPointer = emptyString;
	}
}

void String::Purge(void)
{
	if (stringPointer != emptyString)
	{
		delete[] stringPointer;
		stringPointer = emptyString;

		logicalSize = 1;
		physicalSize = 0;
	}
}

void String::Resize(int32 size)
{
	logicalSize = size;
	if ((size > physicalSize) || (size < physicalSize / 2))
	{
		if (stringPointer != emptyString)
		{
			delete[] stringPointer;
		}

		physicalSize = GetPhysicalSize(size);
		stringPointer = new char[physicalSize];
	}
}

String& String::Set(const char *s, int32 length)
{
	length = Min(length, Text::GetTextLength(s));

	int32 size = length + 1;
	if (size > 1)
	{
		Resize(size);
		Text::CopyText(s, stringPointer, length);
	}
	else
	{
		Purge();
	}

	return (*this);
}

String& String::operator =(String&& s)
{
	if (stringPointer != emptyString)
	{
		delete[] stringPointer;
	}

	logicalSize = s.logicalSize;
	physicalSize = s.physicalSize;
	stringPointer = s.stringPointer;

	s.stringPointer = emptyString;
	return (*this);
}

String& String::operator =(const String& s)
{
	int32 size = s.logicalSize;
	if (size > 1)
	{
		Resize(size);
		Text::CopyText(s, stringPointer);
	}
	else
	{
		Purge();
	}

	return (*this);
}

String& String::operator =(const char *s)
{
	int32 size = Text::GetTextLength(s) + 1;
	if (size > 1)
	{
		Resize(size);
		Text::CopyText(s, stringPointer);
	}
	else
	{
		Purge();
	}

	return (*this);
}

String& String::operator +=(const String& s)
{
	int32 length = s.Length();
	if (length > 0)
	{
		int32 size = logicalSize + length;
		if (size > 1)
		{
			if (size > physicalSize)
			{
				physicalSize = Max(GetPhysicalSize(size), physicalSize + physicalSize / 2);
				char *newPointer = new char[physicalSize];

				if (stringPointer != emptyString)
				{
					Text::CopyText(stringPointer, newPointer);
					delete[] stringPointer;
				}

				stringPointer = newPointer;
			}

			Text::CopyText(s, stringPointer + logicalSize - 1);
			logicalSize = size;
		}
	}

	return (*this);
}

String& String::operator +=(const char *s)
{
	int32 length = Text::GetTextLength(s);
	if (length > 0)
	{
		int32 size = logicalSize + length;
		if (size > 1)
		{
			if (size > physicalSize)
			{
				physicalSize = Max(GetPhysicalSize(size), physicalSize + physicalSize / 2);
				char *newPointer = new char[physicalSize];

				if (stringPointer != emptyString)
				{
					Text::CopyText(stringPointer, newPointer);
					delete[] stringPointer;
				}

				stringPointer = newPointer;
			}

			Text::CopyText(s, stringPointer + logicalSize - 1);
			logicalSize = size;
		}
	}

	return (*this);
}

String& String::operator +=(char k)
{
	int32 size = logicalSize + 1;
	if (size > physicalSize)
	{
		physicalSize = Max(GetPhysicalSize(size), physicalSize + physicalSize / 2);
		char *newPointer = new char[physicalSize];

		if (stringPointer != emptyString)
		{
			Text::CopyText(stringPointer, newPointer);
			delete[] stringPointer;
		}

		stringPointer = newPointer;
	}

	stringPointer[logicalSize - 1] = k;
	stringPointer[logicalSize] = 0;
	logicalSize = size;
	return (*this);
}

String& String::SetLength(int32 length)
{
	int32 size = length + 1;
	if (size > 1)
	{
		if (size != logicalSize)
		{
			logicalSize = size;
			if ((size > physicalSize) || (size < physicalSize / 2))
			{
				physicalSize = GetPhysicalSize(size);
				char *newPointer = new char[physicalSize];

				if (stringPointer != emptyString)
				{
					Text::CopyText(stringPointer, newPointer, length);
					delete[] stringPointer;
				}

				stringPointer = newPointer;
			}

			stringPointer[length] = 0;
		}
	}
	else
	{
		Purge();
	}

	return (*this);
}
