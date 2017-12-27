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


/* Modified by
 * Tim Chen <chenwenli@chenwenl.com>
 * ON
 * Oct. 13, 2017
 * For
 * his tutorial Game Engine from Scratch
 * At
 * https://zhuanlan.zhihu.com/c_119702958
 */



#ifndef OpenDDL_h
#define OpenDDL_h


#include "ODDLArray.h"
#include "ODDLString.h"
#include "ODDLTree.h"
#include "ODDLMap.h"


namespace ODDL
{
    typedef unsigned_int32      DataResult;
    typedef unsigned_int32      DataType;
    typedef unsigned_int32      StructureType;


    enum
    {
        kDataMaxPrimitiveArraySize          = 256
    };


    enum : StructureType
    {
        kStructureRoot                      = 0,
        kStructurePrimitive                 = "PRIM"_i32
    };


    enum : DataType
    {
        kDataBool                           = "BOOL"_i32,       //## Boolean.
        kDataInt8                           = "INT8"_i32,       //## 8-bit signed integer.
        kDataInt16                          = "IN16"_i32,       //## 16-bit signed integer.
        kDataInt32                          = "IN32"_i32,       //## 32-bit signed integer.
        kDataInt64                          = "IN64"_i32,       //## 64-bit signed integer.
        kDataUnsignedInt8                   = "UIN8"_i32,       //## 8-bit unsigned integer.
        kDataUnsignedInt16                  = "UI16"_i32,       //## 16-bit unsigned integer.
        kDataUnsignedInt32                  = "UI32"_i32,       //## 32-bit unsigned integer.
        kDataUnsignedInt64                  = "UI64"_i32,       //## 64-bit unsigned integer.
        kDataHalf                           = "HALF"_i32,       //## 16-bit floating-point.
        kDataFloat                          = "FLOT"_i32,       //## 32-bit floating-point.
        kDataDouble                         = "DOUB"_i32,       //## 64-bit floating-point.
        kDataString                         = "STRG"_i32,       //## String.
        kDataRef                            = "RFNC"_i32,       //## Reference.
        kDataType                           = "TYPE"_i32        //## Type.
    };


    enum : DataResult
    {
        kDataOkay                           = 0,
        kDataSyntaxError                    = "SYNT"_i32,       //## The syntax is invalid.
        kDataIdentifierEmpty                = "IDEM"_i32,       //## No identifier was found where one was expected.
        kDataIdentifierIllegalChar          = "IDIC"_i32,       //## An identifier contains an illegal character.
        kDataStringInvalid                  = "STIV"_i32,       //## A string literal is invalid.
        kDataStringIllegalChar              = "STIC"_i32,       //## A string literal contains an illegal character.
        kDataStringIllegalEscape            = "STIE"_i32,       //## A string literal contains an illegal escape sequence.
        kDataStringEndOfFile                = "STEF"_i32,       //## The end of file was reached inside a string literal.
        kDataCharIllegalChar                = "CHIC"_i32,       //## A character literal contains an illegal character.
        kDataCharIllegalEscape              = "CHIE"_i32,       //## A character literal contains an illegal escape sequence.
        kDataCharEndOfFile                  = "CHEF"_i32,       //## The end of file was reached inside a character literal.
        kDataBoolInvalid                    = "BLIV"_i32,       //## A boolean value is not "true" or "false".
        kDataTypeInvalid                    = "TYIV"_i32,       //## A data type value does not name a primitive type.
        kDataIntegerOverflow                = "INOV"_i32,       //## An integer value lies outside the range of representable values for the number of bits in its underlying type.
        kDataFloatOverflow                  = "FLOV"_i32,       //## A hexadecimal or binary literal used to represent a floating-point value contains more bits than the underlying type.
        kDataFloatInvalid                   = "FLIV"_i32,       //## A floating-point literal has an invalid format.
        kDataReferenceInvalid               = "RFIV"_i32,       //## A reference uses an invalid syntax.
        kDataStructUndefined                = "STUD"_i32,       //## An undefined structure type was encountered.
        kDataStructNameExists               = "STNE"_i32,       //## A structure name is equal to a previously used structure name.
        kDataPropertySyntaxError            = "PPSE"_i32,       //## A property list contains a syntax error.
        kDataPropertyUndefined              = "PPUD"_i32,       //## An undefined property was encountered. This error is generated when the $@Structure::ValidateProperty@$ function returns $false$.
        kDataPropertyInvalidType            = "PPIT"_i32,       //## A property has specified an invalid type. This error is generated if the $@Structure::ValidateProperty@$ function does not specify a recognized data type.
        kDataPrimitiveSyntaxError           = "PMSE"_i32,       //## A primitive data structure contains a syntax error.
        kDataPrimitiveIllegalArraySize      = "PMAS"_i32,       //## A primitive data array size is too large.
        kDataPrimitiveInvalidFormat         = "PMIF"_i32,       //## A primitive data structure contains data in an invalid format.
        kDataPrimitiveArrayUnderSize        = "PMUS"_i32,       //## A primitive array contains too few elements.
        kDataPrimitiveArrayOverSize         = "PMOS"_i32,       //## A primitive array contains too many elements.
        kDataInvalidStructure               = "IVST"_i32        //## A structure contains a substructure of an invalid type, or a structure of an invalid type appears at the top level of the file. This error is generated when either the $@Structure::ValidateSubstructure@$ function or $@DataDescription::ValidateTopLevelStructure@$ function returns $false$.
    };


    enum : DataResult
    {
        kDataMissingSubstructure            = "MSSB"_i32,       //## A structure is missing a substructure of a required type.
        kDataExtraneousSubstructure         = "EXSB"_i32,       //## A structure contains too many substructures of a legal type.
        kDataInvalidDataFormat              = "IVDF"_i32,       //## The primitive data contained in a structure uses an invalid format (type or subarray size).
        kDataBrokenRef                      = "BREF"_i32        //## The target of a reference does not exist.
    };


    class DataDescription;


    namespace Data
    {
        extern const int8 identifierCharState[256];

        int32 GetWhitespaceLength(const char *text);
        DataResult ReadDataType(const char *text, int32 *textLength, DataType *value);
        DataResult ReadIdentifier(const char *text, int32 *textLength, char *restrict identifier = nullptr);
        DataResult ReadStringLiteral(const char *text, int32 *textLength, int32 *stringLength, char *restrict string = nullptr);
        DataResult ReadBoolLiteral(const char *text, int32 *textLength, bool *value);
        DataResult ReadUnsignedLiteral(const char *text, int32 *textLength, unsigned_int64 *value);
        DataResult ReadFloatMagnitude(const char *text, int32 *textLength, unsigned_int16 *value);
        DataResult ReadFloatMagnitude(const char *text, int32 *textLength, float *value);
        DataResult ReadFloatMagnitude(const char *text, int32 *textLength, double *value);
    }


    //# \class  StructureRef        Represents a structure reference in an OpenDDL file.
    //
    //# The $StructureRef$ class represents a structure reference in an OpenDDL file.
    //
    //# \def    class StructureRef
    //
    //# \ctor   StructureRef(bool global = true);
    //
    //# \param  global      A boolean value that indicates whether the reference is global.
    //
    //# \desc
    //# The $StructureRef$ class holds an array of structure names that compose an OpenDDL reference.
    //# A reference can be global or local, depending on whether the first name in the sequence is a
    //# global name or local name. Only the first name can be a global name, and the rest, if any,
    //# are always local names.
    //#
    //# The $@StructureRef::GetNameArray@$ function can be used to retrieve the array of
    //# $@Utilities/String@$ objects containing the sequence of names stored in the reference. For a
    //# null reference, this array is empty. For non-null references, the $@StructureRef::GetGlobalRefFlag@$
    //# function can be called to determine whether a reference is global or local.
    //#
    //# The $@StructureRef::AddName@$ function is used to add names to a reference. Initially,
    //# a $StructureRef$ object is a null reference, and thus its name array is empty.
    //
    //# \also   $@DataDescription::FindStructure@$
    //# \also   $@Structure::FindStructure@$


    //# \function   StructureRef::GetNameArray      Returns the array of names stored in a reference.
    //
    //# \proto  const ImmutableArray<String>& GetNameArray(void) const;
    //
    //# \desc
    //# The $GetNameArray$ function returns the array of names stored in a structure reference.
    //# The $@Utilities/Array::GetElementCount@$ function can be used to retrieve the number of names in
    //# the array, and the $[]$ operator can be used to retrieve each individual name in the array.
    //# For a null reference, the name array is empty (i.e., has zero elements).
    //
    //# \also   $@StructureRef::GetGlobalRefFlag@$
    //# \also   $@StructureRef::AddName@$
    //# \also   $@StructureRef::Reset@$
    //# \also   $@Utilities/Array@$


    //# \function   StructureRef::GetGlobalRefFlag      Returns a boolean value indicating whether a reference is global.
    //
    //# \proto  bool GetGlobalRefFlag(void) const;
    //
    //# \desc
    //# The $GetGlobalRefFlag$ function returns $true$ if the structure reference is global, and it returns
    //# $false$ if the structure reference is local. A structure reference is global if the first name in its
    //# name array is a global name, as specified in the OpenDDL file.
    //
    //# \also   $@StructureRef::GetNameArray@$
    //# \also   $@StructureRef::AddName@$
    //# \also   $@StructureRef::Reset@$


    //# \function   StructureRef::AddName       Adds a name to a reference.
    //
    //# \proto  void AddName(String&& name);
    //
    //# \param  name    The name to add to the sequence stored in the reference. The dollar sign or percent sign should be omitted.
    //
    //# \desc
    //# The $AddName$ function adds the name specified by the $name$ parameter to the array of names stored
    //# in a structure reference. A move constructor is used to add the name to the array, so the string object
    //# passed in to this function becomes the empty string upon return.
    //
    //# \also   $@StructureRef::GetNameArray@$
    //# \also   $@StructureRef::GetGlobalRefFlag@$
    //# \also   $@StructureRef::Reset@$


    //# \function   StructureRef::Reset     Resets a reference to an empty sequence of names.
    //
    //# \proto  void Reset(bool global = true);
    //
    //# \param  global      A boolean value that indicates whether the reference is global.
    //
    //# \desc
    //# The $Reset$ function removes all of the names stored in a structure reference, making the name
    //# array empty. Upon return, the structure reference is a null reference. New names can be added
    //# to the reference by calling the $@StructureRef::AddName@$ function. The $global$ parameter
    //# specifies whether the first name added is a global name or local name.
    //
    //# \also   $@StructureRef::AddName@$
    //# \also   $@StructureRef::GetNameArray@$
    //# \also   $@StructureRef::GetGlobalRefFlag@$


    class StructureRef
    {
        private:

            Array<String, 1>    nameArray;
            bool                globalRefFlag;

        public:

            StructureRef(bool global = true);
            ~StructureRef();

            const ImmutableArray<String>& GetNameArray(void) const
            {
                return (nameArray);
            }

            bool GetGlobalRefFlag(void) const
            {
                return (globalRefFlag);
            }

            void AddName(String&& name)
            {
                nameArray.AddElement(static_cast<String&&>(name));
            }

            void Reset(bool global = true);
    };


    struct BoolDataType
    {
        typedef bool PrimType;

        enum
        {
            kStructureType = kDataBool
        };

        static DataResult ParseValue(const char *& text, PrimType *value);
    };


    struct Int8DataType
    {
        typedef int8 PrimType;

        enum
        {
            kStructureType = kDataInt8
        };

        static DataResult ParseValue(const char *& text, PrimType *value);
    };


    struct Int16DataType
    {
        typedef int16 PrimType;

        enum
        {
            kStructureType = kDataInt16
        };

        static DataResult ParseValue(const char *& text, PrimType *value);
    };


    struct Int32DataType
    {
        typedef int32 PrimType;

        enum
        {
            kStructureType = kDataInt32
        };

        static DataResult ParseValue(const char *& text, PrimType *value);
    };


    struct Int64DataType
    {
        typedef int64 PrimType;

        enum
        {
            kStructureType = kDataInt64
        };

        static DataResult ParseValue(const char *& text, PrimType *value);
    };


    struct UnsignedInt8DataType
    {
        typedef unsigned_int8 PrimType;

        enum
        {
            kStructureType = kDataUnsignedInt8
        };

        static DataResult ParseValue(const char *& text, PrimType *value);
    };


    struct UnsignedInt16DataType
    {
        typedef unsigned_int16 PrimType;

        enum
        {
            kStructureType = kDataUnsignedInt16
        };

        static DataResult ParseValue(const char *& text, PrimType *value);
    };


    struct UnsignedInt32DataType
    {
        typedef unsigned_int32 PrimType;

        enum
        {
            kStructureType = kDataUnsignedInt32
        };

        static DataResult ParseValue(const char *& text, PrimType *value);
    };


    struct UnsignedInt64DataType
    {
        typedef unsigned_int64 PrimType;

        enum
        {
            kStructureType = kDataUnsignedInt64
        };

        static DataResult ParseValue(const char *& text, PrimType *value);
    };


    struct HalfDataType
    {
        typedef unsigned_int16 PrimType;

        enum
        {
            kStructureType = kDataHalf
        };

        static DataResult ParseValue(const char *& text, PrimType *value);
    };


    struct FloatDataType
    {
        typedef float PrimType;

        enum
        {
            kStructureType = kDataFloat
        };

        static DataResult ParseValue(const char *& text, PrimType *value);
    };


    struct DoubleDataType
    {
        typedef double PrimType;

        enum
        {
            kStructureType = kDataDouble
        };

        static DataResult ParseValue(const char *& text, PrimType *value);
    };


    struct StringDataType
    {
        typedef String PrimType;

        enum
        {
            kStructureType = kDataString
        };

        static DataResult ParseValue(const char *& text, PrimType *value);
    };


    struct RefDataType
    {
        typedef StructureRef PrimType;

        enum
        {
            kStructureType = kDataRef
        };

        static DataResult ParseValue(const char *& text, PrimType *value);
    };


    struct TypeDataType
    {
        typedef DataType PrimType;

        enum
        {
            kStructureType = kDataType
        };

        static DataResult ParseValue(const char *& text, PrimType *value);
    };


    //# \class  Structure       Represents a data structure in an OpenDDL file.
    //
    //# The $Structure$ class represents a data structure in an OpenDDL file.
    //
    //# \def    class Structure : public Tree<Structure>, public MapElement<Structure>
    //
    //# \ctor   Structure(StructureType type);
    //
    //# The constructor has protected access. The $Structure$ class can only exist as the base class for another class.
    //
    //# \param  type    The type of the structure.
    //
    //# \desc
    //# The $Structure$ class is the base class for objects that represent data structures in an Open Data
    //# Description Language (OpenDDL) file. Structures of a specific data type are represented by objects whose
    //# types are subclasses of the $Structure$ class.
    //#
    //# Structures corresponding to built-in primitive data types are represented by objects whose type is
    //# a specialization of the $@DataStructure@$ class template, and these all have a common base class of
    //# type $@PrimitiveStructure@$ that is a direct subclass of the $Structure$ class.
    //#
    //# Custom data structures defined by specific OpenDDL-based file formats are represented by application-defined
    //# subclasses of the $Structure$ class. When an OpenDDL file is parsed, the $@DataDescription::CreateStructure@$
    //# function is called to construct the proper subclass for a given type identifier.
    //#
    //# $Structure$ objects are organized into a tree hierarchy that can be traversed using the functions of the
    //# $@Utilities/Tree<Structure>@$ base class. The tree hierarchy corresponds to the data layout in the OpenDDL file.
    //#
    //# Subclasses for custom data structures should specify a unique 32-bit integer for the $type$ parameter, normally
    //# represented by a four-character code. All four-character codes consisting only of uppercase letters and decimal
    //# digits are reserved for use by the engine.
    //
    //# \base   Utilities/Tree<Structure>           $Structure$ objects are organized into a tree hierarchy.
    //# \base   Utilities/MapElement<Structure>     Used internally by the $DataDescription$ class.
    //
    //# \also   $@PrimitiveStructure@$
    //# \also   $@DataStructure@$
    //# \also   $@DataDescription@$
    //
    //# \wiki   Open_Data_Description_Language      Open Data Description Language


    //# \function   Structure::GetStructureType     Returns the structure type.
    //
    //# \proto  StructureType GetStructureType(void) const;
    //
    //# \desc
    //# The $GetStructureType$ function returns the structure type. This may be a custom data type or one of
    //# the following built-in primitive data types.
    //
    //# \table  DataType
    //
    //# \also   $@Structure::GetBaseStructureType@$
    //# \also   $@PrimitiveStructure@$
    //# \also   $@DataStructure@$


    //# \function   Structure::GetBaseStructureType     Returns the base structure type.
    //
    //# \proto  StructureType GetBaseStructureType(void) const;
    //
    //# \desc
    //# The $GetBaseStructureType$ function returns the base structure type representing a more general classification
    //# than the type returned by the $@Structure::GetStructureType@$ function. By default, the base structure type
    //# is simply the value zero, but a subclass of the $Structure$ class may set the base structure type to any value
    //# it wants by calling the $@Structure::SetBaseStructureType@$ function.
    //#
    //# The base structure type for all built-in primitive data structures derived from the $@PrimitiveStructure@$
    //# class is $kStructurePrimitive$.
    //
    //# \also   $@Structure::SetBaseStructureType@$
    //# \also   $@Structure::GetStructureType@$
    //# \also   $@PrimitiveStructure@$


    //# \function   Structure::SetBaseStructureType     Sets the base structure type.
    //
    //# \proto  void SetBaseStructureType(StructureType type);
    //
    //# \param  type    The base structure type.
    //
    //# \desc
    //# The $GetBaseStructureType$ function sets the base structure type to that specified by the $type$ parameter.
    //# The base structure type represents a more general classification than the type returned by the
    //# $@Structure::GetStructureType@$ function. A subclass of the $Structure$ class may set the base structure
    //# type to a custom value in order to indicate that the data structure belongs to a particular category.
    //#
    //# Subclasses for custom data structures should specify a unique 32-bit integer for the $type$ parameter, normally
    //# represented by a four-character code. All four-character codes consisting only of uppercase letters and decimal
    //# digits are reserved for use by the engine.
    //#
    //# By default, the base structure type for custom subclasses of the $Structure$ class is the value zero.
    //
    //# \also   $@Structure::GetBaseStructureType@$
    //# \also   $@Structure::GetStructureType@$


    //# \function   Structure::GetStructureName     Returns the structure name.
    //
    //# \proto  const char *GetStructureName(void) const;
    //
    //# \desc
    //# The $GetStructureName$ function returns the name of a structure. The dollar sign or percent sign at the
    //# beginning is omitted. If a structure has no name, then the return value points to an empty string&mdash;it is
    //# never $nullptr$.
    //#
    //# Whether the structure's name is global or local can be determined by calling the $@Structure::GetGlobalNameFlag@$ function.
    //
    //# \also   $@Structure::GetGlobalNameFlag@$
    //# \also   $@Structure::GetStructureType@$
    //# \also   $@Structure::GetBaseStructureType@$


    //# \function   Structure::GetGlobalNameFlag    Returns a boolean value indicating whether a structure's name is global.
    //
    //# \proto  bool GetGlobalNameFlag(void) const;
    //
    //# \desc
    //# The $GetGlobalNameFlag$ function returns $true$ if the structure's name is global, and it returns
    //# $false$ if the structure's name is local. A structure's name is global if it begins with a dollar sign in the
    //# OpenDDL file, and the name is local if it begins with a percent sign.
    //
    //# \also   $@Structure::GetStructureName@$


    //# \function   Structure::FindStructure        Finds a named structure using a local reference.
    //
    //# \proto  Structure *FindStructure(const StructureRef& reference, int32 index = 0) const;
    //
    //# \param  reference   The reference to the structure to find.
    //# \param  index       The index of the name to search for within the reference's name array. This is used internally and should be set to its default value of 0.
    //
    //# \desc
    //# The $FindStructure$ function finds the structure referenced by the sequence of names stored in the
    //# $reference$ parameter and returns a pointer to it. If no such structure exists, then the return value is $nullptr$.
    //# Only structures belonging to the subtree of the structure for which this function is called can be
    //# returned by this function. The sequence of names in the reference identify a branch along the subtree
    //# leading to the referenced structure.
    //#
    //# The reference must be a local reference, meaning that the first name stored in the reference is a
    //# local name as indicated by a value of $false$ being returned by the $@StructureRef::GetGlobalRefFlag@$ function.
    //# If the reference is not a local reference, then this function always returns $nullptr$. The $@DataDescription::FindStructure@$
    //# function should be used to find a structure through a global reference.
    //#
    //# If the specified reference has an empty name array, then the return value is always $nullptr$. The empty name
    //# array is assigned to a reference data value when $null$ appears in the OpenDDL file.
    //
    //# \also   $@StructureRef@$
    //# \also   $@Structure::GetStructureName@$
    //# \also   $@Structure::GetGlobalNameFlag@$
    //# \also   $@DataDescription::FindStructure@$
    //# \also   $@DataDescription::GetRootStructure@$


    //# \function   Structure::ValidateProperty     Determines the validity of a property and returns its type and location.
    //
    //# \proto  virtual bool ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value);
    //
    //# \param  dataDescription     The data description object to which the structure belongs.
    //# \param  identifier          The property identifier, as read from an OpenDDL file.
    //# \param  type                A pointer to the location that receives the data type for the property.
    //# \param  value               A pointer to the location that receives a pointer to the property's value.
    //
    //# \desc
    //# The $ValidateProperty$ function is called for each property specified in an OpenDDL file for a particular
    //# data structure to determine whether the property is valid, and if so, what type it expects and where
    //# to store its value. This function should be overridden by any subclass of the $Structure$ class that
    //# defines properties, and it should return $true$ when the $identifier$ parameter identifies one of the
    //# supported properties. If the string specified by the $identifier$ parameter is not recognized, then
    //# the function should return $false$. The default implementation of the $ValidateProperty$ function
    //# always returns $false$.
    //#
    //# When the property identifier is valid, an implementation of the $ValidateProperty$ function must write
    //# the type of data expected by the property to the location specified by the $type$ parameter, and it must
    //# write a pointer to the location holding the property value to the location specified by the $value$
    //# parameter. The data type must be one of the following values.
    //
    //# \table  DataType
    //
    //# For the string and reference data types, the property value must be represented by a $@Utilities/String@$
    //# object with the default template parameter of 0.
    //#
    //# An implementation of the $ValidateProperty$ function must always return the same results for any given
    //# property identifier. If the same property appears multiple times in the property list for a structure,
    //# then values appearing later must overwrite earlier values, and the earlier values must be ignored.
    //
    //# \also   $@Structure::ValidateSubstructure@$
    //# \also   $@DataDescription@$


    //# \function   Structure::ValidateSubstructure     Determines the validity of a substructure.
    //
    //# \proto  virtual bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
    //
    //# \param  dataDescription     The data description object to which the structure belongs.
    //# \param  structure           The substructure to validate.
    //
    //# \desc
    //# The $ValidateSubstructure$ function is called for the $Structure$ object representing the enclosing data
    //# structure each time a new substructure is created to determine whether the new substructure can legally
    //# be contained in the data of the $Structure$ object. An overriding implementation should examine the
    //# structure specified by the $structure$ parameter and return $true$ if it can legally appear as a direct
    //# subnode of the $Structure$ object for which the $ValidateSubstructure$ function is called. Otherwise, the
    //# function should return $false$.
    //#
    //# An implementation would typically call the $@Structure::GetStructureType@$ function to make its decision,
    //# but other information such as the base structure type or the primitive subarray size may also be taken into
    //# account. At the time that the $ValidateSubstructure$ function is called, no data belonging to the structure
    //# is available, so the data itself cannot be used to validate any substructures.
    //#
    //# The default implementation of the $ValidateSubstructure$ function always returns $true$.
    //
    //# \also   $@Structure::ValidateProperty@$
    //# \also   $@Structure::GetStructureType@$
    //# \also   $@Structure::GetBaseStructureType@$
    //# \also   $@PrimitiveStructure::GetArraySize@$
    //# \also   $@DataDescription::ValidateTopLevelStructure@$


    //# \function   Structure::ProcessData      Performs custom processing of the structure data.
    //
    //# \proto  virtual DataResult ProcessData(DataDescription *dataDescription);
    //
    //# \param  dataDescription     The $@DataDescription@$ object to which the structure belongs.
    //
    //# \desc
    //# The $ProcessData$ function can be overridden by a subclass to perform any custom processing of the data
    //# contained in a structure. This function is called for all direct subnodes of the root structure of
    //# a data file when the $@DataDescription::ProcessText@$ function is called. (These correspond to the
    //# top-level data structures in the file itself.) The implementation may examine the structure's subtree
    //# and take whatever action is appropriate to process the data.
    //#
    //# The $dataDescription$ parameter points to the $@DataDescription@$ object to which the structure belongs.
    //# An implementation of the $ProcessData$ function may call the $@DataDescription::FindStructure@$ function
    //# to find referenced structures.
    //#
    //# If no error occurs, then the $ProcessData$ function should return $kDataOkay$. Otherwise, an error
    //# code should be returned. An implementation may return a custom error code or one of the following
    //# standard error codes.
    //
    //# \table  DataProcessResult
    //
    //# The default implementation calls the $ProcessData$ function for each of the direct subnodes of a data
    //# structure. If an error is returned by any of these calls, then this function stops iterating through
    //# its subnodes and returns the error code immediately.
    //#
    //# An overriding implementation of the $ProcessData$ function is not required to call the base class
    //# implementation, but if it does, it must pass the value of the $dataDescription$ parameter through
    //# to the base class.
    //
    //# \also   $@DataDescription::ProcessText@$


    class Structure : public Tree<Structure>, public MapElement<Structure>
    {
        friend class DataDescription;

        public:

            typedef ConstCharKey KeyType;

        private:

            StructureType       structureType;
            StructureType       baseStructureType;

            String              structureName;
            bool                globalNameFlag;

            Map<Structure>      structureMap;

            const char          *textLocation;

        protected:

            Structure(StructureType type);

            void SetBaseStructureType(StructureType type)
            {
                baseStructureType = type;
            }

        public:

            virtual ~Structure();

            using Tree<Structure>::Previous;
            using Tree<Structure>::Next;
            using Tree<Structure>::PurgeSubtree;

            KeyType GetKey(void) const
            {
                return (structureName);
            }

            StructureType GetStructureType(void) const
            {
                return (structureType);
            }

            StructureType GetBaseStructureType(void) const
            {
                return (baseStructureType);
            }

            const char *GetStructureName(void) const
            {
                return (structureName);
            }

            bool GetGlobalNameFlag(void) const
            {
                return (globalNameFlag);
            }

            Structure *GetFirstSubstructure(StructureType type) const;
            Structure *GetLastSubstructure(StructureType type) const;

            Structure *FindStructure(const StructureRef& reference, int32 index = 0) const;

            virtual bool ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value);
            virtual bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;

            virtual DataResult ProcessData(DataDescription *dataDescription);
    };


    //# \class  PrimitiveStructure      Base class for built-in primitive data structures in an OpenDDL file.
    //
    //# The $PrimitiveStructure$ class is the base class for built-in primitive data structures in an OpenDDL file.
    //
    //# \def    class PrimitiveStructure : public Structure
    //
    //# \ctor   PrimitiveStructure(StructureType type);
    //
    //# The constructor has protected access. The $PrimitiveStructure$ class can only exist as the base class for a built-in specialization of the $@DataStructure@$ class template.
    //
    //# \param  type    The type of the structure.
    //
    //# \desc
    //# The $PrimitiveStructure$ class is the base class for all objects that represent built-in primitive
    //# data structures in an Open Data Description Language (OpenDDL) file. Specific types of primitive data
    //# structures are represented by specializations of the $@DataStructure@$ class template.
    //#
    //# The base structure type returned by the $@Structure::GetBaseStructureType@$ function for all built-in
    //# primitive data structures is $kStructurePrimitive$.
    //
    //# \base   Structure       A primitive structure is a special type of $Structure$ object.
    //
    //# \also   $@DataStructure@$


    //# \function   PrimitiveStructure::GetArraySize        Returns the array size for a primitive data type.
    //
    //# \proto  unsigned_int32 GetArraySize(void) const;
    //
    //# \desc
    //# The $GetArraySize$ function returns the number of elements in each subarray contained in the data
    //# belonging to a primitive data structure. This corresponds to the number specified in brackets after
    //# the type identifier. If no array size is specified for a structure, then the $GetArraySize$ function
    //# returns zero.
    //
    //# \also   $@DataStructure::GetArrayDataElement@$


    class PrimitiveStructure : public Structure
    {
        friend class DataDescription;

        private:

            unsigned_int32      arraySize;

        protected:

            PrimitiveStructure(StructureType type);

        public:

            ~PrimitiveStructure();

            unsigned_int32 GetArraySize(void) const
            {
                return (arraySize);
            }

            virtual DataResult ParseData(const char *& text) = 0;
    };


    //# \class  DataStructure       Represents a specific built-in primitive data structure in an OpenDDL file.
    //
    //# The $DataStructure$ class template represents each of the specific built-in primitive data structure in an OpenDDL file.
    //
    //# \def    template <class type> class DataStructure final : public PrimitiveStructure
    //
    //# \tparam     type    An object type representing the specific type of data contained in the structure.
    //
    //# \ctor   DataStructure();
    //
    //# \desc
    //# A specialization of the $DataStructure$ class template represents each of the built-in primitive data structures in
    //# an Open Data Description Language (OpenDDL) file. The $type$ template parameter can only be one of the following types.
    //
    //# \value  BoolDataType            A boolean type that can have the value $true$ or $false$.
    //# \value  Int8DataType            An 8-bit signed integer that can have values in the range [&minus;128,&nbsp;127].
    //# \value  Int16DataType           A 16-bit signed integer that can have values in the range [&minus;32768,&nbsp;32767].
    //# \value  Int32DataType           A 32-bit signed integer that can have values in the range [&minus;2147483648,&nbsp;2147483647].
    //# \value  Int64DataType           A 64-bit signed integer that can have values in the range [&minus;9223372036854775808,&nbsp;9223372036854775807].
    //# \value  UnsignedInt8DataType    An 8-bit unsigned integer that can have values in the range [0,&nbsp;255].
    //# \value  UnsignedInt16DataType   A 16-bit unsigned integer that can have values in the range [0,&nbsp;65535].
    //# \value  UnsignedInt32DataType   A 32-bit unsigned integer that can have values in the range [0,&nbsp;4294967295].
    //# \value  UnsignedInt64DataType   A 64-bit unsigned integer that can have values in the range [0,&nbsp;18446744073709551615].
    //# \value  HalfDataType            A 16-bit floating-point type conforming to the standard S1E5M10 format.
    //# \value  FloatDataType           A 32-bit floating-point type conforming to the standard S1E8M23 format.
    //# \value  DoubleDataType          A 64-bit floating-point type conforming to the standard S1E11M52 format.
    //# \value  StringDataType          A double-quoted character string with contents encoded in UTF-8.
    //# \value  RefDataType             A sequence of structure names, or the keyword $null$.
    //# \value  TypeDataType            A type whose values are identifiers naming types in the first column of this table.
    //
    //# \desc
    //# The raw data belonging to a data structure is stored as a linear array in memory, regardless of whether subarrays
    //# are specified. The total number of data elements present can be retrieved with the $@DataStructure::GetDataElementCount@$
    //# function, and the data for each element can be retrieved with the $@DataStructure::GetDataElement@$ function.
    //# If subarrays are in use, then the elements belonging to each subarray are stored contiguously, and each subarray
    //# is then stored contiguously with the one preceding it. The $@DataStructure::GetArrayDataElement@$ function can be
    //# used to retrieve a pointer to the beginning of a specific subarray.
    //
    //# \base   PrimitiveStructure      Each data structure specialization is a specific type of $PrimitiveStructure$ object.


    //# \function   DataStructure::GetDataElementCount      Returns the total number of data elements stored in a data structure.
    //
    //# \proto  int32 GetDataElementCount(void) const;
    //
    //# \desc
    //# The $GetDataElementCount$ function returns the total number of individual primitive data elements stored in a
    //# data structure. If a data structure contains subarrays, then this function returns the number of elements in
    //# each subarray multiplied by the total number of subarrays.
    //
    //# \also   $@DataStructure::GetDataElement@$
    //# \also   $@DataStructure::GetArrayDataElement@$
    //# \also   $@PrimitiveStructure::GetArraySize@$


    //# \function   DataStructure::GetDataElement       Returns a single data element stored in a data structure.
    //
    //# \proto  const PrimType& GetDataElement(int32 index) const;
    //
    //# \param  index   The zero-based index of the data element to retrieve.
    //
    //# \desc
    //# The $GetDataElement$ function returns a single primitive data element stored in a data structure. The legal values
    //# of the $index$ parameter range from zero to <i>n</i>&nbsp;&minus;1, inclusive, where <i>n</i> is the total count of
    //# data elements returned by the $@DataStructure::GetDataElementCount@$ function.
    //#
    //# The $PrimType$ type is defined by the class corresponding to the $type$ template parameter associated
    //# with the particular specialization of the $DataStructure$ class template.
    //
    //# \also   $@DataStructure::GetArrayDataElement@$
    //# \also   $@DataStructure::GetDataElementCount@$
    //# \also   $@PrimitiveStructure::GetArraySize@$


    //# \function   DataStructure::GetArrayDataElement      Returns a pointer to a subarray stored in a data structure.
    //
    //# \proto  const PrimType *GetArrayDataElement(int32 index) const;
    //
    //# \param  index   The zero-based index of the subarray to retrieve.
    //
    //# \desc
    //# The $GetArrayDataElement$ function returns a pointer to the first element in a subarray stored in a data structure.
    //# The legal values of the $index$ parameter range from zero to <i>n</i>&nbsp;/&nbsp;<i>s</i>&nbsp;&minus;1, inclusive,
    //# where <i>n</i> is the total count of data elements returned by the $@DataStructure::GetDataElementCount@$ function,
    //# and <i>s</i> is the size of each subarray returned by the $@PrimitiveStructure::GetArraySize@$ function. The elements
    //# of each subarray are stored contiguously in memory.
    //#
    //# The $PrimType$ type is defined by the class corresponding to the $type$ template parameter associated
    //# with the particular specialization of the $DataStructure$ class template.
    //
    //# \also   $@DataStructure::GetDataElement@$
    //# \also   $@DataStructure::GetDataElementCount@$
    //# \also   $@PrimitiveStructure::GetArraySize@$


    template <class type> class DataStructure final : public PrimitiveStructure
    {
        private:

            typedef typename type::PrimType PrimType;

            Array<PrimType, 1>      dataArray;

        public:

            DataStructure();
            ~DataStructure();

            int32 GetDataElementCount(void) const
            {
                return (dataArray.GetElementCount());
            }

            const PrimType& GetDataElement(int32 index) const
            {
                return (dataArray[index]);
            }

            const PrimType *GetArrayDataElement(int32 index) const
            {
                return (&dataArray[GetArraySize() * index]);
            }

            DataResult ParseData(const char *& text) override;
    };


    class RootStructure : public Structure
    {
        public:

            RootStructure();
            ~RootStructure();

            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const override;
    };


    //# \class  DataDescription     Represents a derivative file format based on the OpenDDL language.
    //
    //# The $DataDescription$ class represents a derivative file format based on the OpenDDL language.
    //
    //# \def    class DataDescription
    //
    //# \ctor   DataDescription();
    //
    //# The constructor has protected access. The $DataDescription$ class can only exist as the base class for another class.
    //
    //# \desc
    //# The $DataDescription$ class is the base class for objects that represent derivative file format based on
    //# the Open Data Description Language (OpenDDL). It serves as a container for the tree hierarchy of data structures
    //# in an OpenDDL file.
    //#
    //# A subclass of the $DataDescription$ class represents a specific OpenDDL-based file format and provides the means for
    //# constructing custom $@Structure@$ subclasses by overriding the $@DataDescription::CreateStructure@$ function.
    //
    //# \also   $@Structure@$
    //# \also   $@PrimitiveStructure@$
    //# \also   $@DataStructure@$
    //
    //# \wiki   Open_Data_Description_Language      Open Data Description Language


    //# \function   DataDescription::GetRootStructure       Returns root structure for an OpenDDL file.
    //
    //# \proto  const Structure *GetRootStructure(void) const;
    //
    //# \desc
    //# The $GetRootStructure$ function returns the root structure for an OpenDDL file. The direct subnodes
    //# of the root structure correspond to the top-level data structures in the file.
    //
    //# \also   $@DataDescription::FindStructure@$
    //# \also   $@Utilities/Tree<Structure>@$


    //# \function   DataDescription::FindStructure      Finds a named structure.
    //
    //# \proto  Structure *FindStructure(const StructureRef& reference) const;
    //
    //# \param  reference   The reference to the structure to find.
    //
    //# \desc
    //# The $FindStructure$ function finds the structure referenced by the sequence of names stored in the
    //# $reference$ parameter and returns a pointer to it. If no such structure exists, then the return value is $nullptr$.
    //#
    //# The reference must be a global reference, meaning that the first name stored in the reference is a
    //# global name as indicated by a value of $true$ being returned by the $@StructureRef::GetGlobalRefFlag@$ function.
    //# If the reference is not a global reference, then this function always returns $nullptr$. The $@Structure::FindStructure@$
    //# function should be used to find a structure through a local reference.
    //#
    //# If the specified reference has an empty name array, then the return value is always $nullptr$. The empty name
    //# array is assigned to a reference data value when $null$ appears in the OpenDDL file.
    //
    //# \also   $@StructureRef@$
    //# \also   $@Structure::FindStructure@$
    //# \also   $@Structure::GetStructureName@$
    //# \also   $@Structure::GetGlobalNameFlag@$
    //# \also   $@DataDescription::GetRootStructure@$


    //# \function   DataDescription::CreateStructure        Constructs a custom data structure.
    //
    //# \proto  virtual Structure *CreateStructure(const String& identifier) const;
    //
    //# \param  identifier      The identifier of a data structure in an OpenDDL file.
    //
    //# \desc
    //# The $CreateStructure$ function should be overridden by any subclass of the $DataDescription$ class
    //# representing a file format that defines custom data structures. The implementation should use the $new$
    //# operator to create a new object based on the $Structure$ subclass corresponding to the $identifier$
    //# parameter. If the identifier is not recognized, then this function should return $nullptr$. The default
    //# implementation always returns $nullptr$.
    //
    //# \also   $@Structure@$


    //# \function   DataDescription::ValidateTopLevelStructure      Determines the validity of a top-level structure.
    //
    //# \proto  virtual bool ValidateTopLevelStructure(const Structure *structure) const;
    //
    //# \param  structure       The top-level structure to validate.
    //
    //# \desc
    //# The $ValidateTopLevelStructure$ function is called each time a new structure is created at the top level
    //# of an OpenDDL file to determine whether the new structure can legally appear outside all other structures.
    //# An overriding implementation should examine the structure specified by the $structure$ parameter and return
    //# $true$ if it can legally appear at the top level of a file, and it should return $false$ otherwise.
    //#
    //# An implementation would typically call the $@Structure::GetStructureType@$ function to make its decision,
    //# but other information such as the base structure type or the primitive subarray size may also be taken into
    //# account. At the time that the $ValidateTopLevelStructure$ function is called, no data belonging to the structure
    //# is available, so the data itself cannot be used to validate any top-level structures.
    //#
    //# The default implementation of the $ValidateTopLevelStructure$ function always returns $true$.
    //
    //# \also   $@Structure::ValidateSubstructure@$


    //# \function   DataDescription::ProcessText        Parses an OpenDDL file and processes the top-level data structures.
    //
    //# \proto  DataResult ProcessText(const char *text);
    //
    //# \param  text    The full contents of an OpenDDL file with a terminating zero byte.
    //
    //# \desc
    //# The $ProcessText$ function parses the entire OpenDDL file specified by the $text$ parameter. If the file is
    //# successfully parsed, then the data is processed as described below. If an error occurs during the parsing stage,
    //# then the $ProcessText$ function returns one of the following values, and the $DataDescription$ object contains no data.
    //
    //# \table  DataResult
    //
    //# During the parsing stage, the $@DataDescription::CreateStructure@$ is called for each custom data structure
    //# that is encountered in order to create an object whose type is the proper subclass of the $@Structure@$ class.
    //
    //# After a successful parse, the $ProcessText$ function iterates through all of the top-level data structures in
    //# the file (which are the direct subnodes of the root structure returned by the $@DataDescription::GetRootStructure@$
    //# function) and calls the $@Structure::ProcessData@$ function for each one. If an error is returned by any of the
    //# calls to the $@Structure::ProcessData@$ function, then the processing stops, and the same error is returned by the
    //# $DataDescription::ProcessText$ function. If all of the top-level data structures are processed without error, then
    //# the $DataDescription::ProcessText$ function returns $kDataOkay$. The error returned during the processing stage can
    //# be one of the following values or a value defined by a derivative data format.
    //
    //# \table  DataProcessResult
    //
    //# If an error is returned for either the parsing stage or the processing stage, then the line number where the error
    //# occurred can be retrieved by calling the $@DataDescription::GetErrorLine@$ function.
    //#
    //# The default implementation of the $@Structure::ProcessData@$ function iterates over the direct subnodes of a
    //# data structure and calls the $ProcessData$ function for each one. If all overrides call the base class implementation,
    //# then the entire tree of data structures will be visited during the processing stage.
    //#
    //# Any implementation of the $@Structure::ProcessData@$ function may make the following assumptions about the data:
    //#
    //# 1. The input text is syntactically valid.<br/>
    //# 2. Each structure described in the input text was recognized and successfully created.<br/>
    //# 3. Each structure is valid as indicated by the $@Structure::ValidateSubstructure@$ function called for its enclosing structure.<br/>
    //# 4. Each property identifier is valid as indicated by the $@Structure::ValidateProperty@$ function called for the associated structure, and it has a value of the proper type assigned to it.<br/>
    //# 5. Any existing subarrays of primitive data have the correct number of elements, matching the number specified in brackets after the primitive type identifier.
    //
    //# \also   $@Structure::ProcessData@$
    //# \also   $@DataDescription::GetErrorLine@$


    //# \function   DataDescription::GetErrorLine       Returns the line on which an error occurred.
    //
    //# \proto  int32 GetErrorLine(void) const;
    //
    //# \desc
    //# The $GetErrorLine$ function returns the line number on which an error occurred when the $@DataDescription::ParseText@$
    //# function was called. Line numbering begins at one. If the $@DataDescription::ParseText@$ function returned $kDataOkay$,
    //# then the $GetErrorLine$ function will return zero.
    //
    //# \also   $@DataDescription::ProcessText@$


    class DataDescription
    {
        friend Structure;

        private:

            Map<Structure>      structureMap;
            RootStructure       rootStructure;

            const Structure     *errorStructure;
            int32               errorLine;

            static Structure *CreatePrimitive(const String& identifier);

            DataResult ParseProperties(const char *& text, Structure *structure);
            DataResult ParseStructures(const char *& text, Structure *root);

        protected:

            DataDescription();

            virtual DataResult ProcessData(void);

        public:

            virtual ~DataDescription();

            const Structure *GetRootStructure(void) const
            {
                return (&rootStructure);
            }

            int32 GetErrorLine(void) const
            {
                return (errorLine);
            }

            Structure *FindStructure(const StructureRef& reference) const;

            virtual Structure *CreateStructure(const String& identifier) const;
            virtual bool ValidateTopLevelStructure(const Structure *structure) const;

            DataResult ProcessText(const char *text);
    };
}


#endif
