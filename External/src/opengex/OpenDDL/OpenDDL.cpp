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


#include "OpenDDL.h"


using namespace ODDL;


namespace ODDL
{
	namespace Data
	{
		const int8 hexadecimalCharValue[55] =
		{
			0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1, -1, -1, -1, -1, -1,
			-1, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			-1, 10, 11, 12, 13, 14, 15
		};

		const int8 identifierCharState[256] =
		{
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
			0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1,
			0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2,
			2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
			2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
			2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
			2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
		};


		int32 ReadEscapeChar(const char *text, unsigned_int32 *value);
		int32 ReadStringEscapeChar(const char *text, int32 *stringLength, char *restrict string);
		DataResult ReadCharLiteral(const char *text, int32 *textLength, unsigned_int64 *value);
		DataResult ReadDecimalLiteral(const char *text, int32 *textLength, unsigned_int64 *value);
		DataResult ReadHexadecimalLiteral(const char *text, int32 *textLength, unsigned_int64 *value);
		DataResult ReadOctalLiteral(const char *text, int32 *textLength, unsigned_int64 *value);
		DataResult ReadBinaryLiteral(const char *text, int32 *textLength, unsigned_int64 *value);
		bool ParseSign(const char *& text);
	}
}


int32 Data::GetWhitespaceLength(const char *text)
{
	const unsigned_int8 *byte = reinterpret_cast<const unsigned_int8 *>(text);
	for (;;)
	{
		unsigned_int32 c = byte[0];
		if (c == 0)
		{
			break;
		}

		if (c >= 33U)
		{
			if (c != '/')
			{
				break;
			}

			c = byte[1];
			if (c == '/')
			{
				byte += 2;
				for (;;)
				{
					c = byte[0];
					if (c == 0)
					{
						goto end;
					}

					byte++;

					if (c == 10)
					{
						break;
					}
				}

				continue;
			}
			else if (c == '*')
			{
				byte += 2;
				for (;;)
				{
					c = byte[0];
					if (c == 0)
					{
						goto end;
					}

					byte++;

					if ((c == '*') && (byte[0] == '/'))
					{
						byte++;
						break;
					}
				}

				continue;
			}

			break;
		}

		byte++;
	}

	end:
	return ((int32) (reinterpret_cast<const char *>(byte) - text));
}

DataResult Data::ReadDataType(const char *text, int32 *textLength, DataType *value)
{
	const unsigned_int8 *byte = reinterpret_cast<const unsigned_int8 *>(text);

	unsigned_int32 c = byte[0];
	if (c == 'i')
	{
		int32 length = ((byte[1] == 'n') && (byte[2] == 't')) ? 3 : 1;

		if ((byte[length] == '8') && (identifierCharState[byte[length + 1]] == 0))
		{
			*value = kDataInt8;
			*textLength = length + 1;
			return (kDataOkay);
		}

		if ((byte[length] == '1') && (byte[length + 1] == '6') && (identifierCharState[byte[length + 2]] == 0))
		{
			*value = kDataInt16;
			*textLength = length + 2;
			return (kDataOkay);
		}

		if ((byte[length] == '3') && (byte[length + 1] == '2') && (identifierCharState[byte[length + 2]] == 0))
		{
			*value = kDataInt32;
			*textLength = length + 2;
			return (kDataOkay);
		}

		if ((byte[length] == '6') && (byte[length + 1] == '4') && (identifierCharState[byte[length + 2]] == 0))
		{
			*value = kDataInt64;
			*textLength = length + 2;
			return (kDataOkay);
		}
	}
	else if (c == 'u')
	{
		int32 length = (Text::CompareText(&text[1], "nsigned_int", 11)) ? 12 : 1;

		if ((byte[length] == '8') && (identifierCharState[byte[length + 1]] == 0))
		{
			*value = kDataUnsignedInt8;
			*textLength = length + 1;
			return (kDataOkay);
		}

		if ((byte[length] == '1') && (byte[length + 1] == '6') && (identifierCharState[byte[length + 2]] == 0))
		{
			*value = kDataUnsignedInt16;
			*textLength = length + 2;
			return (kDataOkay);
		}

		if ((byte[length] == '3') && (byte[length + 1] == '2') && (identifierCharState[byte[length + 2]] == 0))
		{
			*value = kDataUnsignedInt32;
			*textLength = length + 2;
			return (kDataOkay);
		}

		if ((byte[length] == '6') && (byte[length + 1] == '4') && (identifierCharState[byte[length + 2]] == 0))
		{
			*value = kDataUnsignedInt64;
			*textLength = length + 2;
			return (kDataOkay);
		}
	}
	else if (c == 'f')
	{
		int32 length = (Text::CompareText(&text[1], "loat", 4)) ? 5 : 1;
		
		if (identifierCharState[byte[length]] == 0)
		{
			*value = kDataFloat;
			*textLength = length;
			return (kDataOkay);
		}

		if ((byte[length] == '1') && (byte[length + 1] == '6') && (identifierCharState[byte[length + 2]] == 0))
		{
			*value = kDataHalf;
			*textLength = length + 2;
			return (kDataOkay);
		}

		if ((byte[length] == '3') && (byte[length + 1] == '2') && (identifierCharState[byte[length + 2]] == 0))
		{
			*value = kDataFloat;
			*textLength = length + 2;
			return (kDataOkay);
		}

		if ((byte[length] == '6') && (byte[length + 1] == '4') && (identifierCharState[byte[length + 2]] == 0))
		{
			*value = kDataDouble;
			*textLength = length + 2;
			return (kDataOkay);
		}
	}
	else if (c == 'b')
	{
		int32 length = (Text::CompareText(&text[1], "ool", 3)) ? 4 : 1;
		
		if (identifierCharState[byte[length]] == 0)
		{
			*value = kDataBool;
			*textLength = length;
			return (kDataOkay);
		}
	}
	else if (c == 'h')
	{
		int32 length = (Text::CompareText(&text[1], "alf", 3)) ? 4 : 1;
		
		if (identifierCharState[byte[length]] == 0)
		{
			*value = kDataHalf;
			*textLength = length;
			return (kDataOkay);
		}
	}
	else if (c == 'd')
	{
		int32 length = (Text::CompareText(&text[1], "ouble", 5)) ? 6 : 1;
		
		if (identifierCharState[byte[length]] == 0)
		{
			*value = kDataDouble;
			*textLength = length;
			return (kDataOkay);
		}
	}
	else if (c == 's')
	{
		int32 length = (Text::CompareText(&text[1], "tring", 5)) ? 6 : 1;
		
		if (identifierCharState[byte[length]] == 0)
		{
			*value = kDataString;
			*textLength = length;
			return (kDataOkay);
		}
	}
	else if (c == 'r')
	{
		int32 length = ((byte[1] == 'e') && (byte[2] == 'f')) ? 3 : 1;

		if (identifierCharState[byte[length]] == 0)
		{
			*value = kDataRef;
			*textLength = length;
			return (kDataOkay);
		}
	}
	else if (c == 't')
	{
		int32 length = (Text::CompareText(&text[1], "ype", 3)) ? 4 : 1;
		
		if (identifierCharState[byte[length]] == 0)
		{
			*value = kDataType;
			*textLength = length;
			return (kDataOkay);
		}
	}

	return (kDataTypeInvalid);
}

DataResult Data::ReadIdentifier(const char *text, int32 *textLength, char *restrict identifier)
{
	const unsigned_int8 *byte = reinterpret_cast<const unsigned_int8 *>(text);
	int32 count = 0;

	unsigned_int32 c = byte[0];
	int32 state = identifierCharState[c];

	if (state == 1)
	{
		if (c < 'A')
		{
			return (kDataIdentifierIllegalChar);
		}

		if (identifier)
		{
			identifier[count] = (char) c;
		}

		count++;
		for (;;)
		{
			c = byte[count];
			state = identifierCharState[c];

			if (state == 1)
			{
				if (identifier)
				{
					identifier[count] = (char) c;
				}

				count++;
				continue;
			}
			else if (state == 2)
			{
				return (kDataIdentifierIllegalChar);
			}

			break;
		}

		if (identifier)
		{
			identifier[count] = 0;
		}

		*textLength = count;
		return (kDataOkay);
	}
	else if (state == 2)
	{
		return (kDataIdentifierIllegalChar);
	}

	return (kDataIdentifierEmpty);
}

int32 Data::ReadEscapeChar(const char *text, unsigned_int32 *value)
{
	const unsigned_int8 *byte = reinterpret_cast<const unsigned_int8 *>(text);
	unsigned_int32 c = byte[0];

	if ((c == '\"') || (c == '\'') || (c == '?') || (c == '\\'))
	{
		*value = c;
		return (1);
	}
	else if (c == 'a')
	{
		*value = '\a';
		return (1);
	}
	else if (c == 'b')
	{
		*value = '\b';
		return (1);
	}
	else if (c == 'f')
	{
		*value = '\f';
		return (1);
	}
	else if (c == 'n')
	{
		*value = '\n';
		return (1);
	}
	else if (c == 'r')
	{
		*value = '\r';
		return (1);
	}
	else if (c == 't')
	{
		*value = '\t';
		return (1);
	}
	else if (c == 'v')
	{
		*value = '\v';
		return (1);
	}
	else if (c == 'x')
	{
		c = byte[1] - '0';
		if (c < 55U)
		{
			int32 x = hexadecimalCharValue[c];
			if (x >= 0)
			{
				c = byte[2] - '0';
				if (c < 55U)
				{
					int32 y = hexadecimalCharValue[c];
					if (y >= 0)
					{
						*value = (char) ((x << 4) | y);
						return (3);
					}
				}
			}
		}
	}

	return (0);
}

int32 Data::ReadStringEscapeChar(const char *text, int32 *stringLength, char *restrict string)
{
	const unsigned_int8 *byte = reinterpret_cast<const unsigned_int8 *>(text);
	unsigned_int32 c = byte[0];

	if (c == 'u')
	{
		unsigned_int32 code = 0;

		for (machine a = 1; a <= 4; a++)
		{
			c = byte[a] - '0';
			if (c >= 55U)
			{
				return (0);
			}

			int32 x = hexadecimalCharValue[c];
			if (x < 0)
			{
				return (0);
			}

			code = (code << 4) | x;
		}

		if (code != 0)
		{
			if (string)
			{
				*stringLength = Text::WriteGlyphCodeUTF8(string, code);
			}
			else
			{
				*stringLength = 1 + (code >= 0x000080) + (code >= 0x000800);
			}

			return (5);
		}
	}
	if (c == 'U')
	{
		unsigned_int32 code = 0;

		for (machine a = 1; a <= 6; a++)
		{
			c = byte[a] - '0';
			if (c >= 55U)
			{
				return (0);
			}

			int32 x = hexadecimalCharValue[c];
			if (x < 0)
			{
				return (0);
			}

			code = (code << 4) | x;
		}

		if ((code != 0) && (code <= 0x10FFFF))
		{
			if (string)
			{
				*stringLength = Text::WriteGlyphCodeUTF8(string, code);
			}
			else
			{
				*stringLength = 1 + (code >= 0x000080) + (code >= 0x000800) + (code >= 0x010000);
			}

			return (7);
		}
	}
	else
	{
		unsigned_int32		value;

		int32 textLength = ReadEscapeChar(text, &value);
		if (textLength != 0)
		{
			if (string)
			{
				*string = (char) value;
			}

			*stringLength = 1;
			return (textLength);
		}
	}

	return (0);
}

DataResult Data::ReadStringLiteral(const char *text, int32 *textLength, int32 *stringLength, char *restrict string)
{
	const unsigned_int8 *byte = reinterpret_cast<const unsigned_int8 *>(text);
	int32 count = 0;

	for (;;)
	{
		unsigned_int32 c = byte[0];
		if ((c == 0) || (c == '\"'))
		{
			break;
		}

		if ((c < 32U) || (c == 127U))
		{
			return (kDataStringIllegalChar);
		}

		if (c != '\\')
		{
			int32 len = Text::ValidateGlyphCodeUTF8(reinterpret_cast<const char *>(byte));
			if (len == 0)
			{
				return (kDataStringIllegalChar);
			}

			if (string)
			{
				for (machine a = 0; a < len; a++)
				{
					string[a] = (char) byte[a];
				}

				string += len;
			}

			byte += len;
			count += len;
		}
		else
		{
			int32	stringLen;

			int32 textLen = ReadStringEscapeChar(reinterpret_cast<const char *>(++byte), &stringLen, string);
			if (textLen == 0)
			{
				return (kDataStringIllegalEscape);
			}

			if (string)
			{
				string += stringLen;
			}

			byte += textLen;
			count += stringLen;
		}
	}

	*textLength = (int32) (reinterpret_cast<const char *>(byte) - text);
	*stringLength = count;
	return (kDataOkay);
}

DataResult Data::ReadBoolLiteral(const char *text, int32 *textLength, bool *value)
{
	const unsigned_int8 *byte = reinterpret_cast<const unsigned_int8 *>(text);

	unsigned_int32 c = byte[0];
	if (c == 'f')
	{
		if ((byte[1] == 'a') && (byte[2] == 'l') && (byte[3] == 's') && (byte[4] == 'e') && (identifierCharState[byte[5]] == 0))
		{
			*value = false;
			*textLength = 5;
			return (kDataOkay);
		}
	}
	else if (c == 't')
	{
		if ((byte[1] == 'r') && (byte[2] == 'u') && (byte[3] == 'e') && (identifierCharState[byte[4]] == 0))
		{
			*value = true;
			*textLength = 4;
			return (kDataOkay);
		}
	}

	return (kDataBoolInvalid);
}

DataResult Data::ReadDecimalLiteral(const char *text, int32 *textLength, unsigned_int64 *value)
{
	const unsigned_int8 *byte = reinterpret_cast<const unsigned_int8 *>(text);

	unsigned_int64 v = 0;
	bool separator = false;
	for (;;)
	{
		unsigned_int32 x = byte[0] - '0';
		if (x < 10U)
		{
			if (v >= 0x199999999999999AULL)
			{
				return (kDataIntegerOverflow);
			}

			unsigned_int64 w = v;
			v = v * 10 + x;

			if ((w >= 9U) && (v < 9U))
			{
				return (kDataIntegerOverflow);
			}

			separator = true;
		}
		else
		{
			if ((x != 47) || (!separator))
			{
				break;
			}

			separator = false;
		}

		byte++;
	}

	if (!separator)
	{
		return (kDataSyntaxError);
	}

	*value = v;
	*textLength = (int32) (reinterpret_cast<const char *>(byte) - text);
	return (kDataOkay);
}

DataResult Data::ReadHexadecimalLiteral(const char *text, int32 *textLength, unsigned_int64 *value)
{
	const unsigned_int8 *byte = reinterpret_cast<const unsigned_int8 *>(text + 2);

	unsigned_int64 v = 0;
	bool separator = false;
	for (;;)
	{
		unsigned_int32 c = byte[0] - '0';
		if (c >= 55U)
		{
			break;
		}

		int32 x = hexadecimalCharValue[c];
		if (x >= 0)
		{
			if ((v >> 60) != 0)
			{
				return (kDataIntegerOverflow);
			}

			v = (v << 4) | x;
			separator = true;
		}
		else
		{
			if ((c != 47) || (!separator))
			{
				break;
			}

			separator = false;
		}

		byte++;
	}

	if (!separator)
	{
		return (kDataSyntaxError);
	}

	*value = v;
	*textLength = (int32) (reinterpret_cast<const char *>(byte) - text);
	return (kDataOkay);
}

DataResult Data::ReadOctalLiteral(const char *text, int32 *textLength, unsigned_int64 *value)
{
	const unsigned_int8 *byte = reinterpret_cast<const unsigned_int8 *>(text + 2);

	unsigned_int64 v = 0;
	bool separator = false;
	for (;;)
	{
		unsigned_int32 x = byte[0] - '0';
		if (x < 8U)
		{
			if (v >= 0x2000000000000000ULL)
			{
				return (kDataIntegerOverflow);
			}

			unsigned_int64 w = v;
			v = v * 8 + x;

			if ((w >= 7U) && (v < 7U))
			{
				return (kDataIntegerOverflow);
			}

			separator = true;
		}
		else
		{
			if ((x != 47) || (!separator))
			{
				break;
			}

			separator = false;
		}

		byte++;
	}

	if (!separator)
	{
		return (kDataSyntaxError);
	}

	*value = v;
	*textLength = (int32) (reinterpret_cast<const char *>(byte) - text);
	return (kDataOkay);
}

DataResult Data::ReadBinaryLiteral(const char *text, int32 *textLength, unsigned_int64 *value)
{
	const unsigned_int8 *byte = reinterpret_cast<const unsigned_int8 *>(text + 2);

	unsigned_int64 v = 0;
	bool separator = false;
	for (;;)
	{
		unsigned_int32 x = byte[0] - '0';
		if (x < 2U)
		{
			if ((v >> 63) != 0)
			{
				return (kDataIntegerOverflow);
			}

			v = (v << 1) | x;
			separator = true;
		}
		else
		{
			if ((x != 47) || (!separator))
			{
				break;
			}

			separator = false;
		}

		byte++;
	}

	if (!separator)
	{
		return (kDataSyntaxError);
	}

	*value = v;
	*textLength = (int32) (reinterpret_cast<const char *>(byte) - text);
	return (kDataOkay);
}

DataResult Data::ReadCharLiteral(const char *text, int32 *textLength, unsigned_int64 *value)
{
	const unsigned_int8 *byte = reinterpret_cast<const unsigned_int8 *>(text);

	unsigned_int64 v = 0;
	for (;;)
	{
		unsigned_int32 c = byte[0];
		if ((c == 0) || (c == '\''))
		{
			break;
		}

		if ((c < 32U) || (c >= 127U))
		{
			return (kDataCharIllegalChar);
		}

		if (c != '\\')
		{
			if ((v >> 56) != 0)
			{
				return (kDataIntegerOverflow);
			}

			v = (v << 8) | c;
			byte++;
		}
		else
		{
			unsigned_int32		x;

			int32 length = ReadEscapeChar(reinterpret_cast<const char *>(++byte), &x);
			if (length == 0)
			{
				return (kDataCharIllegalEscape);
			}

			if ((v >> 56) != 0)
			{
				return (kDataIntegerOverflow);
			}

			v = (v << 8) | x;
			byte += length;
		}
	}

	*value = v;
	*textLength = (int32) (reinterpret_cast<const char *>(byte) - text);
	return (kDataOkay);
}

DataResult Data::ReadUnsignedLiteral(const char *text, int32 *textLength, unsigned_int64 *value)
{
	const unsigned_int8 *byte = reinterpret_cast<const unsigned_int8 *>(text);

	unsigned_int32 c = byte[0];
	if (c == '0')
	{
		c = byte[1];

		if ((c == 'x') || (c == 'X'))
		{
			return (ReadHexadecimalLiteral(text, textLength, value));
		}

		if ((c == 'o') || (c == 'O'))
		{
			return (ReadOctalLiteral(text, textLength, value));
		}

		if ((c == 'b') || (c == 'B'))
		{
			return (ReadBinaryLiteral(text, textLength, value));
		}
	}
	else if (c == '\'')
	{
		int32	len;

		DataResult result = ReadCharLiteral(reinterpret_cast<const char *>(byte + 1), &len, value);
		if (result == kDataOkay)
		{
			if (byte[len + 1] != '\'')
			{
				return (kDataCharEndOfFile);
			}

			*textLength = len + 2;
		}

		return (result);
	}

	return (ReadDecimalLiteral(text, textLength, value));
}

DataResult Data::ReadFloatMagnitude(const char *text, int32 *textLength, unsigned_int16 *value)
{
	const unsigned_int8 *byte = reinterpret_cast<const unsigned_int8 *>(text);

	unsigned_int32 c = byte[0];
	if (c == '0')
	{
		c = byte[1];

		if ((c == 'x') || (c == 'X'))
		{
			unsigned_int64		v;

			DataResult result = ReadHexadecimalLiteral(text, textLength, &v);
			if (result == kDataOkay)
			{
				if (v > 0x000000000000FFFF)
				{
					return (kDataFloatOverflow);
				}

				*reinterpret_cast<unsigned_int32 *>(value) = (unsigned_int32) v;
			}

			return (result);
		}

		if ((c == 'o') || (c == 'O'))
		{
			unsigned_int64		v;

			DataResult result = ReadOctalLiteral(text, textLength, &v);
			if (result == kDataOkay)
			{
				if (v > 0x000000000000FFFF)
				{
					return (kDataFloatOverflow);
				}

				*reinterpret_cast<unsigned_int32 *>(value) = (unsigned_int32) v;
			}

			return (result);
		}

		if ((c == 'b') || (c == 'B'))
		{
			unsigned_int64		v;

			DataResult result = ReadBinaryLiteral(text, textLength, &v);
			if (result == kDataOkay)
			{
				if (v > 0x000000000000FFFF)
				{
					return (kDataFloatOverflow);
				}

				*reinterpret_cast<unsigned_int32 *>(value) = (unsigned_int32) v;
			}

			return (result);
		}
	}

	float v = 0.0F;
	bool separator = false;
	for (;;)
	{
		unsigned_int32 x = byte[0] - '0';
		if (x < 10U)
		{
			v = v * 10.0F + (float) x;
			separator = true;
		}
		else
		{
			if ((x != 47) || (!separator))
			{
				break;
			}

			separator = false;
		}

		byte++;
	}

	if (!separator)
	{
		return (kDataSyntaxError);
	}

	c = byte[0];
	if (c == '.')
	{
		byte++;

		float decimal = 10.0F;
		separator = false;
		for (;;)
		{
			unsigned_int32 x = byte[0] - '0';
			if (x < 10U)
			{
				v += (float) x / decimal;
				decimal *= 10.0F;
				separator = true;
			}
			else
			{
				if ((x != 47) || (!separator))
				{
					break;
				}

				separator = false;
			}

			byte++;
		}

		if (!separator)
		{
			return (kDataSyntaxError);
		}

		c = byte[0];
	}

	if ((c == 'e') || (c == 'E'))
	{
		bool negative = false;

		c = (++byte)[0];
		if (c == '-')
		{
			negative = true;
			byte++;
		}
		else if (c == '+')
		{
			byte++;
		}
		else if (c - '0' >= 10U)
		{
			return (kDataFloatInvalid);
		}

		int32 exponent = 0;
		bool digit = false;
		separator = false;
		for (;;)
		{
			unsigned_int32 x = byte[0] - '0';
			if (x < 10U)
			{
				exponent = Min(exponent * 10 + x, 65535);
				digit = true;
				separator = true;
			}
			else
			{
				if ((x != 47) || (!separator))
				{
					break;
				}

				separator = false;
			}

			byte++;
		}

		if ((!digit) || (!separator))
		{
			return (kDataSyntaxError);
		}

		if (exponent != 0)
		{
			if (negative)
			{
				exponent = -exponent;
			}

			v *= (float) exp((float) exponent * 2.3025850929940456840179914546844F);
		}
	}

	unsigned_int32 f = *reinterpret_cast<unsigned_int32 *>(&v);
	unsigned_int32 s = (f >> 16) & 0x8000;
	unsigned_int32 m = (f >> 13) & 0x03FF;
	int32 e = ((f >> 23) & 0xFF) - 127;

	if (e >= -14)
	{
		if (e <= 15)
		{
			*value = (unsigned_int16) (s | ((e + 15) << 10) | m);
		}
		else
		{
			*value = (unsigned_int16) (s | 0x7C00);
		}
	}
	else
	{
		*value = (unsigned_int16) s;
	}

	*textLength = (int32) (reinterpret_cast<const char *>(byte) - text);
	return (kDataOkay);
}

DataResult Data::ReadFloatMagnitude(const char *text, int32 *textLength, float *value)
{
	const unsigned_int8 *byte = reinterpret_cast<const unsigned_int8 *>(text);

	unsigned_int32 c = byte[0];
	if (c == '0')
	{
		c = byte[1];

		if ((c == 'x') || (c == 'X'))
		{
			unsigned_int64		v;

			DataResult result = ReadHexadecimalLiteral(text, textLength, &v);
			if (result == kDataOkay)
			{
				if (v > 0x00000000FFFFFFFF)
				{
					return (kDataFloatOverflow);
				}

				*reinterpret_cast<unsigned_int32 *>(value) = (unsigned_int32) v;
			}

			return (result);
		}

		if ((c == 'o') || (c == 'O'))
		{
			unsigned_int64		v;

			DataResult result = ReadOctalLiteral(text, textLength, &v);
			if (result == kDataOkay)
			{
				if (v > 0x00000000FFFFFFFF)
				{
					return (kDataFloatOverflow);
				}

				*reinterpret_cast<unsigned_int32 *>(value) = (unsigned_int32) v;
			}

			return (result);
		}

		if ((c == 'b') || (c == 'B'))
		{
			unsigned_int64		v;

			DataResult result = ReadBinaryLiteral(text, textLength, &v);
			if (result == kDataOkay)
			{
				if (v > 0x00000000FFFFFFFF)
				{
					return (kDataFloatOverflow);
				}

				*reinterpret_cast<unsigned_int32 *>(value) = (unsigned_int32) v;
			}

			return (result);
		}
	}

	float v = 0.0F;
	bool separator = false;
	for (;;)
	{
		unsigned_int32 x = byte[0] - '0';
		if (x < 10U)
		{
			v = v * 10.0F + (float) x;
			separator = true;
		}
		else
		{
			if ((x != 47) || (!separator))
			{
				break;
			}

			separator = false;
		}

		byte++;
	}

	if (!separator)
	{
		return (kDataSyntaxError);
	}

	c = byte[0];
	if (c == '.')
	{
		byte++;

		float decimal = 10.0F;
		separator = false;
		for (;;)
		{
			unsigned_int32 x = byte[0] - '0';
			if (x < 10U)
			{
				v += (float) x / decimal;
				decimal *= 10.0F;
				separator = true;
			}
			else
			{
				if ((x != 47) || (!separator))
				{
					break;
				}

				separator = false;
			}

			byte++;
		}

		if (!separator)
		{
			return (kDataSyntaxError);
		}

		c = byte[0];
	}

	if ((c == 'e') || (c == 'E'))
	{
		bool negative = false;

		c = (++byte)[0];
		if (c == '-')
		{
			negative = true;
			byte++;
		}
		else if (c == '+')
		{
			byte++;
		}
		else if (c - '0' >= 10U)
		{
			return (kDataFloatInvalid);
		}

		int32 exponent = 0;
		bool digit = false;
		separator = false;
		for (;;)
		{
			unsigned_int32 x = byte[0] - '0';
			if (x < 10U)
			{
				exponent = Min(exponent * 10 + x, 65535);
				digit = true;
				separator = true;
			}
			else
			{
				if ((x != 47) || (!separator))
				{
					break;
				}

				separator = false;
			}

			byte++;
		}

		if ((!digit) || (!separator))
		{
			return (kDataSyntaxError);
		}

		if (exponent != 0)
		{
			if (negative)
			{
				exponent = -exponent;
			}

			v *= (float) exp((float) exponent * 2.3025850929940456840179914546844F);
		}
	}

	*value = v;
	*textLength = (int32) (reinterpret_cast<const char *>(byte) - text);
	return (kDataOkay);
}

DataResult Data::ReadFloatMagnitude(const char *text, int32 *textLength, double *value)
{
	const unsigned_int8 *byte = reinterpret_cast<const unsigned_int8 *>(text);

	unsigned_int32 c = byte[0];
	if (c == '0')
	{
		c = byte[1];

		if ((c == 'x') || (c == 'X'))
		{
			unsigned_int64		v;

			DataResult result = ReadHexadecimalLiteral(text, textLength, &v);
			if (result == kDataIntegerOverflow)
			{
				return (kDataFloatOverflow);
			}

			*reinterpret_cast<unsigned_int64 *>(value) = v;
			return (result);
		}

		if ((c == 'o') || (c == 'O'))
		{
			unsigned_int64		v;

			DataResult result = ReadOctalLiteral(text, textLength, &v);
			if (result == kDataIntegerOverflow)
			{
				return (kDataFloatOverflow);
			}

			*reinterpret_cast<unsigned_int64 *>(value) = v;
			return (result);
		}

		if ((c == 'b') || (c == 'B'))
		{
			unsigned_int64		v;

			DataResult result = ReadBinaryLiteral(text, textLength, &v);
			if (result == kDataIntegerOverflow)
			{
				return (kDataFloatOverflow);
			}

			*reinterpret_cast<unsigned_int64 *>(value) = v;
			return (result);
		}
	}

	double v = 0.0;
	bool separator = false;
	for (;;)
	{
		unsigned_int32 x = byte[0] - '0';
		if (x < 10U)
		{
			v = v * 10.0 + (double) x;
			separator = true;
		}
		else
		{
			if ((x != 47) || (!separator))
			{
				break;
			}

			separator = false;
		}

		byte++;
	}

	if (!separator)
	{
		return (kDataSyntaxError);
	}

	c = byte[0];
	if (c == '.')
	{
		double decimal = 10.0;
		separator = false;
		for (;;)
		{
			unsigned_int32 x = byte[0] - '0';
			if (x < 10U)
			{
				v += (double) x / decimal;
				decimal *= 10.0;
				separator = true;
			}
			else
			{
				if ((x != 47) || (!separator))
				{
					break;
				}

				separator = false;
			}

			byte++;
		}

		if (!separator)
		{
			return (kDataSyntaxError);
		}

		c = byte[0];
	}

	if ((c == 'e') || (c == 'E'))
	{
		bool negative = false;

		c = (++byte)[0];
		if (c == '-')
		{
			negative = true;
			byte++;
		}
		else if (c == '+')
		{
			byte++;
		}
		else if (c - '0' >= 10U)
		{
			return (kDataFloatInvalid);
		}

		int32 exponent = 0;
		bool digit = false;
		separator = false;
		for (;;)
		{
			unsigned_int32 x = byte[0] - '0';
			if (x < 10U)
			{
				exponent = Min(exponent * 10 + x, 65535);
				digit = true;
				separator = true;
			}
			else
			{
				if ((x != 47) || (!separator))
				{
					break;
				}

				separator = false;
			}

			byte++;
		}

		if ((!digit) || (!separator))
		{
			return (kDataSyntaxError);
		}

		if (exponent != 0)
		{
			if (negative)
			{
				exponent = -exponent;
			}

			v *= exp((double) exponent * 2.3025850929940456840179914546844);
		}
	}

	*value = v;
	*textLength = (int32) (reinterpret_cast<const char *>(byte) - text);
	return (kDataOkay);
}

bool Data::ParseSign(const char *& text)
{
	char c = text[0];

	if (c == '-')
	{
		text++;
		text += GetWhitespaceLength(text);
		return (true);
	}

	if (c == '+')
	{
		text++;
		text += GetWhitespaceLength(text);
	}

	return (false);
}


StructureRef::StructureRef(bool global)
{
	globalRefFlag = global;
}

StructureRef::~StructureRef()
{
}

void StructureRef::Reset(bool global)
{
	nameArray.Purge();
	globalRefFlag = global;
}


DataResult BoolDataType::ParseValue(const char *& text, PrimType *value)
{
	int32	length;

	DataResult result = Data::ReadBoolLiteral(text, &length, value);
	if (result != kDataOkay)
	{
		return (result);
	}

	text += length;
	text += Data::GetWhitespaceLength(text);

	return (kDataOkay);
}


DataResult Int8DataType::ParseValue(const char *& text, PrimType *value)
{
	int32			length;
	unsigned_int64	unsignedValue;

	bool negative = Data::ParseSign(text);

	DataResult result = Data::ReadUnsignedLiteral(text, &length, &unsignedValue);
	if (result != kDataOkay)
	{
		return (result);
	}

	if (!negative)
	{
		if (unsignedValue > 0x7F)
		{
			return (kDataIntegerOverflow);
		}

		*value = (int8) unsignedValue;
	}
	else
	{
		if (unsignedValue > 0x80)
		{
			return (kDataIntegerOverflow);
		}

		*value = (int8) -(int64) unsignedValue;
	}

	text += length;
	text += Data::GetWhitespaceLength(text);

	return (kDataOkay);
}


DataResult Int16DataType::ParseValue(const char *& text, PrimType *value)
{
	int32			length;
	unsigned_int64	unsignedValue;

	bool negative = Data::ParseSign(text);

	DataResult result = Data::ReadUnsignedLiteral(text, &length, &unsignedValue);
	if (result != kDataOkay)
	{
		return (result);
	}

	if (!negative)
	{
		if (unsignedValue > 0x7FFF)
		{
			return (kDataIntegerOverflow);
		}

		*value = (int16) unsignedValue;
	}
	else
	{
		if (unsignedValue > 0x8000)
		{
			return (kDataIntegerOverflow);
		}

		*value = (int16) -(int64) unsignedValue;
	}

	text += length;
	text += Data::GetWhitespaceLength(text);

	return (kDataOkay);
}


DataResult Int32DataType::ParseValue(const char *& text, PrimType *value)
{
	int32			length;
	unsigned_int64	unsignedValue;

	bool negative = Data::ParseSign(text);

	DataResult result = Data::ReadUnsignedLiteral(text, &length, &unsignedValue);
	if (result != kDataOkay)
	{
		return (result);
	}

	if (!negative)
	{
		if (unsignedValue > 0x7FFFFFFF)
		{
			return (kDataIntegerOverflow);
		}

		*value = (int32) unsignedValue;
	}
	else
	{
		if (unsignedValue > 0x80000000)
		{
			return (kDataIntegerOverflow);
		}

		*value = (int32) -(int64) unsignedValue;
	}

	text += length;
	text += Data::GetWhitespaceLength(text);

	return (kDataOkay);
}


DataResult Int64DataType::ParseValue(const char *& text, PrimType *value)
{
	int32			length;
	unsigned_int64	unsignedValue;

	bool negative = Data::ParseSign(text);

	DataResult result = Data::ReadUnsignedLiteral(text, &length, &unsignedValue);
	if (result != kDataOkay)
	{
		return (result);
	}

	if (!negative)
	{
		if (unsignedValue > 0x7FFFFFFFFFFFFFFF)
		{
			return (kDataIntegerOverflow);
		}

		*value = unsignedValue;
	}
	else
	{
		if (unsignedValue > 0x8000000000000000)
		{
			return (kDataIntegerOverflow);
		}

		*value = -(int64) unsignedValue;
	}

	text += length;
	text += Data::GetWhitespaceLength(text);

	return (kDataOkay);
}


DataResult UnsignedInt8DataType::ParseValue(const char *& text, PrimType *value)
{
	int32			length;
	unsigned_int64	unsignedValue;

	bool negative = Data::ParseSign(text);

	DataResult result = Data::ReadUnsignedLiteral(text, &length, &unsignedValue);
	if (result != kDataOkay)
	{
		return (result);
	}

	if (negative)
	{
		unsignedValue = (unsigned_int64) -(int64) unsignedValue;
	}

	*value = (unsigned_int8) unsignedValue;

	text += length;
	text += Data::GetWhitespaceLength(text);

	return (kDataOkay);
}


DataResult UnsignedInt16DataType::ParseValue(const char *& text, PrimType *value)
{
	int32			length;
	unsigned_int64	unsignedValue;

	bool negative = Data::ParseSign(text);

	DataResult result = Data::ReadUnsignedLiteral(text, &length, &unsignedValue);
	if (result != kDataOkay)
	{
		return (result);
	}

	if (negative)
	{
		unsignedValue = (unsigned_int64) -(int64) unsignedValue;
	}

	*value = (unsigned_int16) unsignedValue;

	text += length;
	text += Data::GetWhitespaceLength(text);

	return (kDataOkay);
}


DataResult UnsignedInt32DataType::ParseValue(const char *& text, PrimType *value)
{
	int32			length;
	unsigned_int64	unsignedValue;

	bool negative = Data::ParseSign(text);

	DataResult result = Data::ReadUnsignedLiteral(text, &length, &unsignedValue);
	if (result != kDataOkay)
	{
		return (result);
	}

	if (negative)
	{
		unsignedValue = (unsigned_int64) -(int64) unsignedValue;
	}

	*value = (unsigned_int32) unsignedValue;

	text += length;
	text += Data::GetWhitespaceLength(text);

	return (kDataOkay);
}


DataResult UnsignedInt64DataType::ParseValue(const char *& text, PrimType *value)
{
	int32			length;
	unsigned_int64	unsignedValue;

	bool negative = Data::ParseSign(text);

	DataResult result = Data::ReadUnsignedLiteral(text, &length, &unsignedValue);
	if (result != kDataOkay)
	{
		return (result);
	}

	if (negative)
	{
		unsignedValue = (unsigned_int64) -(int64) unsignedValue;
	}

	*value = unsignedValue;

	text += length;
	text += Data::GetWhitespaceLength(text);

	return (kDataOkay);
}


DataResult HalfDataType::ParseValue(const char *& text, PrimType *value)
{
	int32			length;
	unsigned_int16	floatValue;

	bool negative = Data::ParseSign(text);

	DataResult result = Data::ReadFloatMagnitude(text, &length, &floatValue);
	if (result != kDataOkay)
	{
		return (result);
	}

	if (negative)
	{
		floatValue ^= 0x8000;
	}

	*value = floatValue;

	text += length;
	text += Data::GetWhitespaceLength(text);

	return (kDataOkay);
}


DataResult FloatDataType::ParseValue(const char *& text, PrimType *value)
{
	int32	length;
	float	floatValue;

	bool negative = Data::ParseSign(text);

	DataResult result = Data::ReadFloatMagnitude(text, &length, &floatValue);
	if (result != kDataOkay)
	{
		return (result);
	}

	if (negative)
	{
		floatValue = -floatValue;
	}

	*value = floatValue;

	text += length;
	text += Data::GetWhitespaceLength(text);

	return (kDataOkay);
}


DataResult DoubleDataType::ParseValue(const char *& text, PrimType *value)
{
	int32		length;
	double		floatValue;

	bool negative = Data::ParseSign(text);

	DataResult result = Data::ReadFloatMagnitude(text, &length, &floatValue);
	if (result != kDataOkay)
	{
		return (result);
	}

	if (negative)
	{
		floatValue = -floatValue;
	}

	*value = floatValue;

	text += length;
	text += Data::GetWhitespaceLength(text);

	return (kDataOkay);
}


DataResult StringDataType::ParseValue(const char *& text, PrimType *value)
{
	int32	textLength;
	int32	stringLength;

	if (text[0] != '"')
	{
		return (kDataStringInvalid);
	}

	int32 accumLength = 0;
	for (;;)
	{
		text++;

		DataResult result = Data::ReadStringLiteral(text, &textLength, &stringLength);
		if (result != kDataOkay)
		{
			return (result);
		}

		value->SetLength(accumLength + stringLength);
		Data::ReadStringLiteral(text, &textLength, &stringLength, &(*value)[accumLength]);
		accumLength += stringLength;

		text += textLength;
		if (text[0] != '"')
		{
			return (kDataStringInvalid);
		}

		text++;
		text += Data::GetWhitespaceLength(text);

		if (text[0] != '"')
		{
			break;
		}
	}

	return (kDataOkay);
}


DataResult RefDataType::ParseValue(const char *& text, PrimType *value)
{
	int32	textLength;

	char c = text[0];
	value->Reset(c != '%');

	if ((unsigned_int32) (c - '$') > 2U)
	{
		const unsigned_int8 *byte = reinterpret_cast<const unsigned_int8 *>(text);
		if ((byte[0] == 'n') && (byte[1] == 'u') && (byte[2] == 'l') && (byte[3] == 'l') && (Data::identifierCharState[byte[4]] == 0))
		{
			text += 4;
			text += Data::GetWhitespaceLength(text);

			return (kDataOkay);
		}

		return (kDataReferenceInvalid);
	}

	do
	{
		text++;

		DataResult result = Data::ReadIdentifier(text, &textLength);
		if (result != kDataOkay)
		{
			return (result);
		}

		String		string;

		string.SetLength(textLength);
		Data::ReadIdentifier(text, &textLength, string);
		value->AddName(static_cast<String&&>(string));

		text += textLength;
		text += Data::GetWhitespaceLength(text);
	} while (text[0] == '%');

	return (kDataOkay);
}


DataResult TypeDataType::ParseValue(const char *& text, PrimType *value)
{
	int32	length;

	DataResult result = Data::ReadDataType(text, &length, value);
	if (result != kDataOkay)
	{
		return (result);
	}

	text += length;
	text += Data::GetWhitespaceLength(text);

	return (kDataOkay);
}


Structure::Structure(StructureType type)
{
	structureType = type;
	baseStructureType = 0;
	globalNameFlag = true;
}

Structure::~Structure()
{
}

Structure *Structure::GetFirstSubstructure(StructureType type) const
{
	Structure *structure = GetFirstSubnode();
	while (structure)
	{
		if (structure->GetStructureType() == type)
		{
			return (structure);
		}

		structure = structure->Next();
	}

	return (nullptr);
}

Structure *Structure::GetLastSubstructure(StructureType type) const
{
	Structure *structure = GetLastSubnode();
	while (structure)
	{
		if (structure->GetStructureType() == type)
		{
			return (structure);
		}

		structure = structure->Previous();
	}

	return (nullptr);
}

Structure *Structure::FindStructure(const StructureRef& reference, int32 index) const
{
	if ((index != 0) || (!reference.GetGlobalRefFlag()))
	{
		const ImmutableArray<String>& nameArray = reference.GetNameArray();

		int32 count = nameArray.GetElementCount();
		if (count != 0)
		{
			Structure *structure = structureMap.Find(nameArray[index]);
			if (structure)
			{
				if (++index < count)
				{
					structure = structure->FindStructure(reference, index);
				}

				return (structure);
			}
		}
	}

	return (nullptr);
}

bool Structure::ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value)
{
	return (false);
}

bool Structure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	return (true);
}

DataResult Structure::ProcessData(DataDescription *dataDescription)
{
	Structure *structure = GetFirstSubnode();
	while (structure)
	{
		DataResult result = structure->ProcessData(dataDescription);
		if (result != kDataOkay)
		{
			if (!dataDescription->errorStructure)
			{
				dataDescription->errorStructure = structure;
			}

			return (result);
		}

		structure = structure->Next();
	}

	return (kDataOkay);
}


PrimitiveStructure::PrimitiveStructure(StructureType type) : Structure(type)
{
	SetBaseStructureType(kStructurePrimitive);

	arraySize = 0;
}

PrimitiveStructure::~PrimitiveStructure()
{
}


template <class type> DataStructure<type>::DataStructure() : PrimitiveStructure(type::kStructureType)
{
}

template <class type> DataStructure<type>::~DataStructure()
{
}

template <class type> DataResult DataStructure<type>::ParseData(const char *& text)
{
	int32 count = 0;

	unsigned_int32 arraySize = GetArraySize();
	if (arraySize == 0)
	{
		for (;;)
		{
			dataArray.SetElementCount(count + 1);

			DataResult result = type::ParseValue(text, &dataArray[count]);
			if (result != kDataOkay)
			{
				return (result);
			}

			text += Data::GetWhitespaceLength(text);

			if (text[0] == ',')
			{
				text++;
				text += Data::GetWhitespaceLength(text);

				count++;
				continue;
			}

			break;
		}
	}
	else
	{
		for (;;)
		{
			if (text[0] != '{')
			{
				return (kDataPrimitiveInvalidFormat);
			}

			text++;
			text += Data::GetWhitespaceLength(text);

			dataArray.SetElementCount(count + arraySize);

			for (unsigned_machine index = 0; index < arraySize; index++)
			{
				if (index != 0)
				{
					if (text[0] != ',')
					{
						return (kDataPrimitiveArrayUnderSize);
					}

					text++;
					text += Data::GetWhitespaceLength(text);
				}

				DataResult result = type::ParseValue(text, &dataArray[count + index]);
				if (result != kDataOkay)
				{
					return (result);
				}

				text += Data::GetWhitespaceLength(text);
			}

			char c = text[0];
			if (c != '}')
			{
				return ((c == ',') ? kDataPrimitiveArrayOverSize : kDataPrimitiveInvalidFormat);
			}

			text++;
			text += Data::GetWhitespaceLength(text);

			if (text[0] == ',')
			{
				text++;
				text += Data::GetWhitespaceLength(text);

				count += arraySize;
				continue;
			}

			break;
		}
	}

	return (kDataOkay);
}


RootStructure::RootStructure() : Structure(kStructureRoot)
{
}

RootStructure::~RootStructure()
{
}

bool RootStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	return (dataDescription->ValidateTopLevelStructure(structure));
}


DataDescription::DataDescription()
{
}

DataDescription::~DataDescription()
{
}

Structure *DataDescription::FindStructure(const StructureRef& reference) const
{
	if (reference.GetGlobalRefFlag())
	{
		const ImmutableArray<String>& nameArray = reference.GetNameArray();

		int32 count = nameArray.GetElementCount();
		if (count != 0)
		{
			Structure *structure = structureMap.Find(nameArray[0]);
			if ((structure) && (count > 1))
			{
				structure = structure->FindStructure(reference, 1);
			}

			return (structure);
		}
	}

	return (nullptr);
}

Structure *DataDescription::CreatePrimitive(const String& identifier)
{
	int32		length;
	DataType	value;

	if (Data::ReadDataType(identifier, &length, &value) == kDataOkay)
	{
		switch (value)
		{
			case kDataBool:
				return (new DataStructure<BoolDataType>);
			case kDataInt8:
				return (new DataStructure<Int8DataType>);
			case kDataInt16:
				return (new DataStructure<Int16DataType>);
			case kDataInt32:
				return (new DataStructure<Int32DataType>);
			case kDataInt64:
				return (new DataStructure<Int64DataType>);
			case kDataUnsignedInt8:
				return (new DataStructure<UnsignedInt8DataType>);
			case kDataUnsignedInt16:
				return (new DataStructure<UnsignedInt16DataType>);
			case kDataUnsignedInt32:
				return (new DataStructure<UnsignedInt32DataType>);
			case kDataUnsignedInt64:
				return (new DataStructure<UnsignedInt64DataType>);
			case kDataHalf:
				return (new DataStructure<HalfDataType>);
			case kDataFloat:
				return (new DataStructure<FloatDataType>);
			case kDataDouble:
				return (new DataStructure<DoubleDataType>);
			case kDataString:
				return (new DataStructure<StringDataType>);
			case kDataRef:
				return (new DataStructure<RefDataType>);
			case kDataType:
				return (new DataStructure<TypeDataType>);
		}
	}

	return (nullptr);
}

Structure *DataDescription::CreateStructure(const String& identifier) const
{
	return (nullptr);
}

bool DataDescription::ValidateTopLevelStructure(const Structure *structure) const
{
	return (true);
}

DataResult DataDescription::ProcessData(void)
{
	return (rootStructure.ProcessData(this));
}

DataResult DataDescription::ParseProperties(const char *& text, Structure *structure)
{
	for (;;)
	{
		int32		length;
		DataType	type;
		void		*value;

		DataResult result = Data::ReadIdentifier(text, &length);
		if (result != kDataOkay)
		{
			return (result);
		}

		String		identifier;

		identifier.SetLength(length);
		Data::ReadIdentifier(text, &length, identifier);

		if (!structure->ValidateProperty(this, identifier, &type, &value))
		{
			return (kDataPropertyUndefined);
		}

		identifier.Purge();

		text += length;
		text += Data::GetWhitespaceLength(text);

		if (text[0] != '=')
		{
			return (kDataPropertySyntaxError);
		}

		text++;
		text += Data::GetWhitespaceLength(text);

		switch (type)
		{
			case kDataBool:
				result = BoolDataType::ParseValue(text, static_cast<BoolDataType::PrimType *>(value));
				break;
			case kDataInt8:
				result = Int8DataType::ParseValue(text, static_cast<Int8DataType::PrimType *>(value));
				break;
			case kDataInt16:
				result = Int16DataType::ParseValue(text, static_cast<Int16DataType::PrimType *>(value));
				break;
			case kDataInt32:
				result = Int32DataType::ParseValue(text, static_cast<Int32DataType::PrimType *>(value));
				break;
			case kDataInt64:
				result = Int64DataType::ParseValue(text, static_cast<Int64DataType::PrimType *>(value));
				break;
			case kDataUnsignedInt8:
				result = UnsignedInt8DataType::ParseValue(text, static_cast<UnsignedInt8DataType::PrimType *>(value));
				break;
			case kDataUnsignedInt16:
				result = UnsignedInt16DataType::ParseValue(text, static_cast<UnsignedInt16DataType::PrimType *>(value));
				break;
			case kDataUnsignedInt32:
				result = UnsignedInt32DataType::ParseValue(text, static_cast<UnsignedInt32DataType::PrimType *>(value));
				break;
			case kDataUnsignedInt64:
				result = UnsignedInt64DataType::ParseValue(text, static_cast<UnsignedInt64DataType::PrimType *>(value));
				break;
			case kDataHalf:
				result = HalfDataType::ParseValue(text, static_cast<HalfDataType::PrimType *>(value));
				break;
			case kDataFloat:
				result = FloatDataType::ParseValue(text, static_cast<FloatDataType::PrimType *>(value));
				break;
			case kDataDouble:
				result = DoubleDataType::ParseValue(text, static_cast<DoubleDataType::PrimType *>(value));
				break;
			case kDataString:
				result = StringDataType::ParseValue(text, static_cast<StringDataType::PrimType *>(value));
				break;
			case kDataRef:
				result = RefDataType::ParseValue(text, static_cast<RefDataType::PrimType *>(value));
				break;
			case kDataType:
				result = TypeDataType::ParseValue(text, static_cast<TypeDataType::PrimType *>(value));
				break;
			default:
				return (kDataPropertyInvalidType);
		}

		if (result != kDataOkay)
		{
			return (result);
		}

		if (text[0] == ',')
		{
			text++;
			text += Data::GetWhitespaceLength(text);

			continue;
		}

		break;
	}

	return (kDataOkay);
}

DataResult DataDescription::ParseStructures(const char *& text, Structure *root)
{
	for (;;)
	{
		int32	length;

		DataResult result = Data::ReadIdentifier(text, &length);
		if (result != kDataOkay)
		{
			return (result);
		}

		String		identifier;

		identifier.SetLength(length);
		Data::ReadIdentifier(text, &length, identifier);

		bool primitive = false;

		Structure *structure = CreatePrimitive(identifier);
		if (structure)
		{
			primitive = true;
		}
		else
		{
			structure = CreateStructure(identifier);
			if (!structure)
			{
				return (kDataStructUndefined);
			}
		}

		identifier.Purge();

		AutoDelete<Structure> structurePtr(structure);
		structure->textLocation = text;

		text += length;
		text += Data::GetWhitespaceLength(text);

		if ((primitive) && (text[0] == '['))
		{
			unsigned_int64		value;

			text++;
			text += Data::GetWhitespaceLength(text);

			if (Data::ParseSign(text))
			{
				return (kDataPrimitiveIllegalArraySize);
			}

			result = Data::ReadUnsignedLiteral(text, &length, &value);
			if (result != kDataOkay)
			{
				return (result);
			}

			if ((value == 0) || (value > kDataMaxPrimitiveArraySize))
			{
				return (kDataPrimitiveIllegalArraySize);
			}

			text += length;
			text += Data::GetWhitespaceLength(text);

			if (text[0] != ']')
			{
				return (kDataPrimitiveSyntaxError);
			}

			text++;
			text += Data::GetWhitespaceLength(text);

			static_cast<PrimitiveStructure *>(structure)->arraySize = (unsigned_int32) value;
		}

		if (!root->ValidateSubstructure(this, structure))
		{
			return (kDataInvalidStructure);
		}

		char c = text[0];
		if ((unsigned_int32) (c - '$') < 2U)
		{
			text++;

			result = Data::ReadIdentifier(text, &length);
			if (result != kDataOkay)
			{
				return (result);
			}

			Data::ReadIdentifier(text, &length, structure->structureName.SetLength(length));

			bool global = (c == '$');
			structure->globalNameFlag = global;

			Map<Structure> *map = (global) ? &structureMap : &root->structureMap;
			if (!map->Insert(structure))
			{
				return (kDataStructNameExists);
			}

			text += length;
			text += Data::GetWhitespaceLength(text);
		}

		if ((!primitive) && (text[0] == '('))
		{
			text++;
			text += Data::GetWhitespaceLength(text);

			if (text[0] != ')')
			{
				result = ParseProperties(text, structure);
				if (result != kDataOkay)
				{
					return (result);
				}

				if (text[0] != ')')
				{
					return (kDataPropertySyntaxError);
				}
			}

			text++;
			text += Data::GetWhitespaceLength(text);
		}

		if (text[0] != '{')
		{
			return (kDataSyntaxError);
		}

		text++;
		text += Data::GetWhitespaceLength(text);

		if (text[0] != '}')
		{
			if (primitive)
			{
				result = static_cast<PrimitiveStructure *>(structure)->ParseData(text);
				if (result != kDataOkay)
				{
					return (result);
				}
			}
			else
			{
				result = ParseStructures(text, structure);
				if (result != kDataOkay)
				{
					return (result);
				}
			}
		}

		if (text[0] != '}')
		{
			return (kDataSyntaxError);
		}

		text++;
		text += Data::GetWhitespaceLength(text);

		root->AppendSubnode(structure);
		structurePtr = nullptr;

		c = text[0];
		if ((c == 0) || (c == '}'))
		{
			break;
		}
	}

	return (kDataOkay);
}

DataResult DataDescription::ProcessText(const char *text)
{
	rootStructure.PurgeSubtree();

	errorStructure = nullptr;
	errorLine = 0;

	const char *start = text;
	text += Data::GetWhitespaceLength(text);

	DataResult result = ParseStructures(text, &rootStructure);
	if ((result == kDataOkay) && (text[0] != 0))
	{
		result = kDataSyntaxError;
	}

	if (result == kDataOkay)
	{
		result = ProcessData();
		if ((result != kDataOkay) && (errorStructure))
		{
			text = errorStructure->textLocation;
		}
	}

	if (result != kDataOkay)
	{
		rootStructure.PurgeSubtree();

		int32 line = 1;
		while (text != start)
		{
			if ((--text)[0] == '\n')
			{
				line++;
			}
		}

		errorLine = line;
	}

	return (result);
}
