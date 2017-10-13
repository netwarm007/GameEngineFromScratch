/*
	OpenGEX Import Template Software License
	==========================================

	OpenGEX Import Template, version 2.0
	Copyright 2014-2017, Eric Lengyel
	All rights reserved.

	The OpenGEX Import Template is free software published on the following website:

		http://opengex.org/

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
	statement "This software contains an OpenGEX import module based on work by
	Eric Lengyel" (without quotes).

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
 * Oct. 12, 2017
 * For
 * his tutorial Game Engine from Scratch
 * At
 * https://zhuanlan.zhihu.com/c_119702958
 */

#include "OpenGEX.h"


using namespace OGEX;


// Base Structure
OpenGexStructure::OpenGexStructure(StructureType type) : Structure(type)
{
}

OpenGexStructure::~OpenGexStructure()
{
}

Structure *OpenGexStructure::GetFirstCoreSubnode(void) const
{
	Structure *structure = GetFirstSubnode();
	while ((structure) && (structure->GetStructureType() == kStructureExtension))
	{
		structure = structure->Next();
	}

	return (structure);
}

Structure *OpenGexStructure::GetLastCoreSubnode(void) const
{
	Structure *structure = GetLastSubnode();
	while ((structure) && (structure->GetStructureType() == kStructureExtension))
	{
		structure = structure->Previous();
	}

	return (structure);
}

bool OpenGexStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	return (structure->GetStructureType() == kStructureExtension);
}


// Material Structure
MetricStructure::MetricStructure() : OpenGexStructure(kStructureMetric)
{
}

MetricStructure::~MetricStructure()
{
}

bool MetricStructure::ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value)
{
	if (identifier == "key")
	{
		*type = kDataString;
		*value = &metricKey;
		return (true);
	}

	return (false);
}

bool MetricStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	if (structure->GetBaseStructureType() == kStructurePrimitive)
	{
		return (true);
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult MetricStructure::ProcessData(DataDescription *dataDescription)
{
	const Structure *structure = GetFirstCoreSubnode();
	if (!structure)
	{
		return (kDataMissingSubstructure);
	}

	if (GetLastCoreSubnode() != structure)
	{
		return (kDataExtraneousSubstructure);
	}

	if (metricKey == "distance")
	{
		if (structure->GetStructureType() != kDataFloat)
		{
			return (kDataInvalidDataFormat);
		}

		const DataStructure<FloatDataType> *dataStructure = static_cast<const DataStructure<FloatDataType> *>(structure);
		if (dataStructure->GetDataElementCount() != 1)
		{
			return (kDataInvalidDataFormat);
		}

		static_cast<OpenGexDataDescription *>(dataDescription)->SetDistanceScale(dataStructure->GetDataElement(0));
	}
	else if (metricKey == "angle")
	{
		if (structure->GetStructureType() != kDataFloat)
		{
			return (kDataInvalidDataFormat);
		}

		const DataStructure<FloatDataType> *dataStructure = static_cast<const DataStructure<FloatDataType> *>(structure);
		if (dataStructure->GetDataElementCount() != 1)
		{
			return (kDataInvalidDataFormat);
		}

		static_cast<OpenGexDataDescription *>(dataDescription)->SetAngleScale(dataStructure->GetDataElement(0));
	}
	else if (metricKey == "time")
	{
		if (structure->GetStructureType() != kDataFloat)
		{
			return (kDataInvalidDataFormat);
		}

		const DataStructure<FloatDataType> *dataStructure = static_cast<const DataStructure<FloatDataType> *>(structure);
		if (dataStructure->GetDataElementCount() != 1)
		{
			return (kDataInvalidDataFormat);
		}

		static_cast<OpenGexDataDescription *>(dataDescription)->SetTimeScale(dataStructure->GetDataElement(0));
	}
	else if (metricKey == "up")
	{
		if (structure->GetStructureType() != kDataString)
		{
			return (kDataInvalidDataFormat);
		}

		const DataStructure<StringDataType> *dataStructure = static_cast<const DataStructure<StringDataType> *>(structure);
		if (dataStructure->GetDataElementCount() != 1)
		{
			return (kDataInvalidDataFormat);
		}

		const String& string = dataStructure->GetDataElement(0);
		if ((string != "z") && (string != "y"))
		{
			return (kDataOpenGexInvalidUpDirection);
		}

		static_cast<OpenGexDataDescription *>(dataDescription)->SetUpDirection(string);
	}
	else if (metricKey == "forward")
	{
		if (structure->GetStructureType() != kDataString)
		{
			return (kDataInvalidDataFormat);
		}

		const DataStructure<StringDataType> *dataStructure = static_cast<const DataStructure<StringDataType> *>(structure);
		if (dataStructure->GetDataElementCount() != 1)
		{
			return (kDataInvalidDataFormat);
		}

		const String& string = dataStructure->GetDataElement(0);
		if ((string != "x") && (string != "y") && (string != "z") && (string != "-x") && (string != "-y") && (string != "-z"))
		{
			return (kDataOpenGexInvalidForwardDirection);
		}

		static_cast<OpenGexDataDescription *>(dataDescription)->SetForwardDirection(string);
	}

	return (kDataOkay);
}


// Name Structure
NameStructure::NameStructure() : OpenGexStructure(kStructureName)
{
}

NameStructure::~NameStructure()
{
}

bool NameStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	if (structure->GetStructureType() == kDataString)
	{
		return (true);
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult NameStructure::ProcessData(DataDescription *dataDescription)
{
	const Structure *structure = GetFirstCoreSubnode();
	if (!structure)
	{
		return (kDataMissingSubstructure);
	}

	if (GetLastCoreSubnode() != structure)
	{
		return (kDataExtraneousSubstructure);
	}

	const DataStructure<StringDataType> *dataStructure = static_cast<const DataStructure<StringDataType> *>(structure);
	if (dataStructure->GetDataElementCount() != 1)
	{
		return (kDataInvalidDataFormat);
	}

	name = dataStructure->GetDataElement(0);
	return (kDataOkay);
}


// Object Ref Structure
ObjectRefStructure::ObjectRefStructure() : OpenGexStructure(kStructureObjectRef)
{
	targetStructure = nullptr;
}

ObjectRefStructure::~ObjectRefStructure()
{
}

bool ObjectRefStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	if (structure->GetStructureType() == kDataRef)
	{
		return (true);
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult ObjectRefStructure::ProcessData(DataDescription *dataDescription)
{
	const Structure *structure = GetFirstCoreSubnode();
	if (!structure)
	{
		return (kDataMissingSubstructure);
	}

	if (GetLastCoreSubnode() != structure)
	{
		return (kDataExtraneousSubstructure);
	}

	const DataStructure<RefDataType> *dataStructure = static_cast<const DataStructure<RefDataType> *>(structure);
	if (dataStructure->GetDataElementCount() != 0)
	{
		Structure *objectStructure = dataDescription->FindStructure(dataStructure->GetDataElement(0));
		if (objectStructure)
		{
			targetStructure = objectStructure;
			return (kDataOkay);
		}
	}

	return (kDataBrokenRef);
}


// Material Ref Structure
MaterialRefStructure::MaterialRefStructure() : OpenGexStructure(kStructureMaterialRef)
{
	materialIndex = 0;
	targetStructure = nullptr;
}

MaterialRefStructure::~MaterialRefStructure()
{
}

bool MaterialRefStructure::ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value)
{
	if (identifier == "index")
	{
		*type = kDataUnsignedInt32;
		*value = &materialIndex;
		return (true);
	}

	return (false);
}

bool MaterialRefStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	if (structure->GetStructureType() == kDataRef)
	{
		return (true);
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult MaterialRefStructure::ProcessData(DataDescription *dataDescription)
{
	const Structure *structure = GetFirstCoreSubnode();
	if (!structure)
	{
		return (kDataMissingSubstructure);
	}

	if (GetLastCoreSubnode() != structure)
	{
		return (kDataExtraneousSubstructure);
	}

	const DataStructure<RefDataType> *dataStructure = static_cast<const DataStructure<RefDataType> *>(structure);
	if (dataStructure->GetDataElementCount() != 0)
	{
		const Structure *materialStructure = dataDescription->FindStructure(dataStructure->GetDataElement(0));
		if (materialStructure)
		{
			if (materialStructure->GetStructureType() != kStructureMaterial)
			{
				return (kDataOpenGexInvalidMaterialRef);
			}

			targetStructure = static_cast<const MaterialStructure *>(materialStructure);
			return (kDataOkay);
		}
	}

	return (kDataBrokenRef);
}


// Animatable Structure
AnimatableStructure::AnimatableStructure(StructureType type) : OpenGexStructure(type)
{
}

AnimatableStructure::~AnimatableStructure()
{
}


// Matrix Structure
MatrixStructure::MatrixStructure(StructureType type) : AnimatableStructure(type)
{
	SetBaseStructureType(kStructureMatrix);

	objectFlag = false;
}

MatrixStructure::~MatrixStructure()
{
}

bool MatrixStructure::ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value)
{
	if (identifier == "object")
	{
		*type = kDataBool;
		*value = &objectFlag;
		return (true);
	}

	return (false);
}


// Transform Structure
TransformStructure::TransformStructure() : MatrixStructure(kStructureTransform)
{
}

TransformStructure::~TransformStructure()
{
}

bool TransformStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	if (structure->GetStructureType() == kDataFloat)
	{
		const PrimitiveStructure *primitiveStructure = static_cast<const PrimitiveStructure *>(structure);
		return (primitiveStructure->GetArraySize() == 16);
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult TransformStructure::ProcessData(DataDescription *dataDescription)
{
	const Structure *structure = GetFirstCoreSubnode();
	if (!structure)
	{
		return (kDataMissingSubstructure);
	}

	if (GetLastCoreSubnode() != structure)
	{
		return (kDataExtraneousSubstructure);
	}

	const DataStructure<FloatDataType> *dataStructure = static_cast<const DataStructure<FloatDataType> *>(structure);

	transformCount = dataStructure->GetDataElementCount() / 16;
	if (transformCount == 0)
	{
		return (kDataInvalidDataFormat);
	}

	transformArray = &dataStructure->GetDataElement(0);
	return (kDataOkay);
}


// Translation Structure
TranslationStructure::TranslationStructure() :
		MatrixStructure(kStructureTranslation),
		translationKind("xyz")
{
}

TranslationStructure::~TranslationStructure()
{
}

bool TranslationStructure::ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value)
{
	if (identifier == "kind")
	{
		*type = kDataString;
		*value = &translationKind;
		return (true);
	}

	return (false);
}

bool TranslationStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	if (structure->GetStructureType() == kDataFloat)
	{
		const PrimitiveStructure *primitiveStructure = static_cast<const PrimitiveStructure *>(structure);
		unsigned_int32 arraySize = primitiveStructure->GetArraySize();
		return ((arraySize == 0) || (arraySize == 3));
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult TranslationStructure::ProcessData(DataDescription *dataDescription)
{
	const Structure *structure = GetFirstCoreSubnode();
	if (!structure)
	{
		return (kDataMissingSubstructure);
	}

	if (GetLastCoreSubnode() != structure)
	{
		return (kDataExtraneousSubstructure);
	}

	const DataStructure<FloatDataType> *dataStructure = static_cast<const DataStructure<FloatDataType> *>(structure);
	unsigned_int32 arraySize = dataStructure->GetArraySize();

	if ((translationKind == "x") || (translationKind == "y") || (translationKind == "z"))
	{
		if ((arraySize != 0) || (dataStructure->GetDataElementCount() != 1))
		{
			return (kDataInvalidDataFormat);
		}
	}
	else if (translationKind == "xyz")
	{
		if ((arraySize != 3) || (dataStructure->GetDataElementCount() != 3))
		{
			return (kDataInvalidDataFormat);
		}
	}
	else
	{
		return (kDataOpenGexInvalidTranslationKind);
	}

	const float *data = &dataStructure->GetDataElement(0);

	// Data is 1 or 3 floats depending on kind.
	// Build application-specific transform here.

	return (kDataOkay);
}


// Rotation Structure
RotationStructure::RotationStructure() :
		MatrixStructure(kStructureRotation),
		rotationKind("axis")
{
}

RotationStructure::~RotationStructure()
{
}

bool RotationStructure::ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value)
{
	if (identifier == "kind")
	{
		*type = kDataString;
		*value = &rotationKind;
		return (true);
	}

	return (false);
}

bool RotationStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	if (structure->GetStructureType() == kDataFloat)
	{
		const PrimitiveStructure *primitiveStructure = static_cast<const PrimitiveStructure *>(structure);
		unsigned_int32 arraySize = primitiveStructure->GetArraySize();
		return ((arraySize == 0) || (arraySize == 4));
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult RotationStructure::ProcessData(DataDescription *dataDescription)
{
	const Structure *structure = GetFirstCoreSubnode();
	if (!structure)
	{
		return (kDataMissingSubstructure);
	}

	if (GetLastCoreSubnode() != structure)
	{
		return (kDataExtraneousSubstructure);
	}

	const DataStructure<FloatDataType> *dataStructure = static_cast<const DataStructure<FloatDataType> *>(structure);
	unsigned_int32 arraySize = dataStructure->GetArraySize();

	if ((rotationKind == "x") || (rotationKind == "y") || (rotationKind == "z"))
	{
		if ((arraySize != 0) || (dataStructure->GetDataElementCount() != 1))
		{
			return (kDataInvalidDataFormat);
		}
	}
	else if ((rotationKind == "axis") || (rotationKind == "quaternion"))
	{
		if ((arraySize != 4) || (dataStructure->GetDataElementCount() != 4))
		{
			return (kDataInvalidDataFormat);
		}
	}
	else
	{
		return (kDataOpenGexInvalidRotationKind);
	}

	const float *data = &dataStructure->GetDataElement(0);

	// Data is 1 or 4 floats depending on kind.
	// Build application-specific transform here.

	return (kDataOkay);
}


// Scale Structure
ScaleStructure::ScaleStructure() :
		MatrixStructure(kStructureScale),
		scaleKind("xyz")
{
}

ScaleStructure::~ScaleStructure()
{
}

bool ScaleStructure::ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value)
{
	if (identifier == "kind")
	{
		*type = kDataString;
		*value = &scaleKind;
		return (true);
	}

	return (false);
}

bool ScaleStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	if (structure->GetStructureType() == kDataFloat)
	{
		const PrimitiveStructure *primitiveStructure = static_cast<const PrimitiveStructure *>(structure);
		unsigned_int32 arraySize = primitiveStructure->GetArraySize();
		return ((arraySize == 0) || (arraySize == 3));
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult ScaleStructure::ProcessData(DataDescription *dataDescription)
{
	const Structure *structure = GetFirstCoreSubnode();
	if (!structure)
	{
		return (kDataMissingSubstructure);
	}

	if (GetLastCoreSubnode() != structure)
	{
		return (kDataExtraneousSubstructure);
	}

	const DataStructure<FloatDataType> *dataStructure = static_cast<const DataStructure<FloatDataType> *>(structure);
	unsigned_int32 arraySize = dataStructure->GetArraySize();

	if ((scaleKind == "x") || (scaleKind == "y") || (scaleKind == "z"))
	{
		if ((arraySize != 0) || (dataStructure->GetDataElementCount() != 1))
		{
			return (kDataInvalidDataFormat);
		}
	}
	else if (scaleKind == "xyz")
	{
		if ((arraySize != 3) || (dataStructure->GetDataElementCount() != 3))
		{
			return (kDataInvalidDataFormat);
		}
	}
	else
	{
		return (kDataOpenGexInvalidScaleKind);
	}

	const float *data = &dataStructure->GetDataElement(0);

	// Data is 1 or 3 floats depending on kind.
	// Build application-specific transform here.

	return (kDataOkay);
}


// Morph Weight Structure
MorphWeightStructure::MorphWeightStructure() : AnimatableStructure(kStructureMorphWeight)
{
	morphIndex = 0;
	morphWeight = 0.0F;
}

MorphWeightStructure::~MorphWeightStructure()
{
}

bool MorphWeightStructure::ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value)
{
	if (identifier == "index")
	{
		*type = kDataUnsignedInt32;
		*value = &morphIndex;
		return (true);
	}

	return (false);
}

bool MorphWeightStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	if (structure->GetStructureType() == kDataFloat)
	{
		const DataStructure<FloatDataType> *dataStructure = static_cast<const DataStructure<FloatDataType> *>(structure);
		unsigned_int32 arraySize = dataStructure->GetArraySize();
		return (arraySize == 0);
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult MorphWeightStructure::ProcessData(DataDescription *dataDescription)
{
	const Structure *structure = GetFirstCoreSubnode();
	if (!structure)
	{
		return (kDataMissingSubstructure);
	}

	if (GetLastCoreSubnode() != structure)
	{
		return (kDataExtraneousSubstructure);
	}

	const DataStructure<FloatDataType> *dataStructure = static_cast<const DataStructure<FloatDataType> *>(structure);
	if (dataStructure->GetDataElementCount() == 1)
	{
		morphWeight = dataStructure->GetDataElement(0);
	}
	else
	{
		return (kDataInvalidDataFormat);
	}

	// Do application-specific morph weight processing here.

	return (kDataOkay);
}


// Node Structure
NodeStructure::NodeStructure() : OpenGexStructure(kStructureNode)
{
	SetBaseStructureType(kStructureNode);
}

NodeStructure::NodeStructure(StructureType type) : OpenGexStructure(type)
{
	SetBaseStructureType(kStructureNode);
}

NodeStructure::~NodeStructure()
{
}

bool NodeStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	StructureType type = structure->GetBaseStructureType();
	if ((type == kStructureNode) || (type == kStructureMatrix))
	{
		return (true);
	}

	type = structure->GetStructureType();
	if ((type == kStructureName) || (type == kStructureAnimation))
	{
		return (true);
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult NodeStructure::ProcessData(DataDescription *dataDescription)
{
	DataResult result = Structure::ProcessData(dataDescription);
	if (result != kDataOkay)
	{
		return (result);
	}

	const Structure *structure = GetFirstSubstructure(kStructureName);
	if (structure)
	{
		if (GetLastSubstructure(kStructureName) != structure)
		{
			return (kDataExtraneousSubstructure);
		}

		nodeName = static_cast<const NameStructure *>(structure)->GetName();
	}
	else
	{
		nodeName = nullptr;
	}

	// Do application-specific node processing here.

	return (kDataOkay);
}

// Object Structure
const ObjectStructure *NodeStructure::GetObjectStructure(void) const
{
	return (nullptr);
}


// Bone Node Structure
BoneNodeStructure::BoneNodeStructure() : NodeStructure(kStructureBoneNode)
{
}

BoneNodeStructure::~BoneNodeStructure()
{
}


// Geometry Structure
GeometryNodeStructure::GeometryNodeStructure() : NodeStructure(kStructureGeometryNode)
{
	// The first entry in each of the following arrays indicates whether the flag
	// is specified by a property. If true, then the second entry in the array
	// indicates the actual value that the property specified.

	visibleFlag[0] = false;
	shadowFlag[0] = false;
	motionBlurFlag[0] = false;
}

GeometryNodeStructure::~GeometryNodeStructure()
{
}

bool GeometryNodeStructure::ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value)
{
	if (identifier == "visible")
	{
		*type = kDataBool;
		*value = &visibleFlag[1];
		visibleFlag[0] = true;
		return (true);
	}

	if (identifier == "shadow")
	{
		*type = kDataBool;
		*value = &shadowFlag[1];
		shadowFlag[0] = true;
		return (true);
	}

	if (identifier == "motion_blur")
	{
		*type = kDataBool;
		*value = &motionBlurFlag[1];
		motionBlurFlag[0] = true;
		return (true);
	}

	return (false);
}

bool GeometryNodeStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	StructureType type = structure->GetStructureType();
	if ((type == kStructureObjectRef) || (type == kStructureMaterialRef) || (type == kStructureMorphWeight))
	{
		return (true);
	}

	return (NodeStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult GeometryNodeStructure::ProcessData(DataDescription *dataDescription)
{
	DataResult result = NodeStructure::ProcessData(dataDescription);
	if (result != kDataOkay)
	{
		return (result);
	}

	bool objectFlag = false;
	bool materialFlag[256] = {false};
	int32 maxMaterialIndex = -1;

	Structure *structure = GetFirstSubnode();
	while (structure)
	{
		StructureType type = structure->GetStructureType();
		if (type == kStructureObjectRef)
		{
			if (objectFlag)
			{
				return (kDataExtraneousSubstructure);
			}

			objectFlag = true;

			Structure *objectStructure = static_cast<ObjectRefStructure *>(structure)->GetTargetStructure();
			if (objectStructure->GetStructureType() != kStructureGeometryObject)
			{
				return (kDataOpenGexInvalidObjectRef);
			}

			geometryObjectStructure = static_cast<GeometryObjectStructure *>(objectStructure);
		}
		else if (type == kStructureMaterialRef)
		{
			const MaterialRefStructure *materialRefStructure = static_cast<MaterialRefStructure *>(structure);

			unsigned_int32 index = materialRefStructure->GetMaterialIndex();
			if (index > 255)
			{
				// We only support up to 256 materials.
				return (kDataOpenGexMaterialIndexUnsupported);
			}

			if (materialFlag[index])
			{
				return (kDataOpenGexDuplicateMaterialRef);
			}

			materialFlag[index] = true;
			maxMaterialIndex = Max(maxMaterialIndex, index);
		}

		structure = structure->Next();
	}

	if (!objectFlag)
	{
		return (kDataMissingSubstructure);
	}

	if (maxMaterialIndex >= 0)
	{
		for (machine a = 0; a <= maxMaterialIndex; a++)
		{
			if (!materialFlag[a])
			{
				return (kDataOpenGexMissingMaterialRef);
			}
		}

		materialStructureArray.SetElementCount(maxMaterialIndex + 1);

		structure = GetFirstSubnode();
		while (structure)
		{
			if (structure->GetStructureType() == kStructureMaterialRef)
			{
				const MaterialRefStructure *materialRefStructure = static_cast<const MaterialRefStructure *>(structure);
				materialStructureArray[materialRefStructure->GetMaterialIndex()] = materialRefStructure->GetTargetStructure();
			}

			structure = structure->Next();
		}
	}

	// Do application-specific node processing here.

	return (kDataOkay);
}

const ObjectStructure *GeometryNodeStructure::GetObjectStructure(void) const
{
	return (geometryObjectStructure);
}


// Light Node Structure
LightNodeStructure::LightNodeStructure() : NodeStructure(kStructureLightNode)
{
	// The first entry in the following array indicates whether the flag is
	// specified by a property. If true, then the second entry in the array
	// indicates the actual value that the property specified.

	shadowFlag[0] = false;
}

LightNodeStructure::~LightNodeStructure()
{
}

bool LightNodeStructure::ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value)
{
	if (identifier == "shadow")
	{
		*type = kDataBool;
		*value = &shadowFlag[1];
		shadowFlag[0] = true;
		return (true);
	}

	return (false);
}

bool LightNodeStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	if (structure->GetStructureType() == kStructureObjectRef)
	{
		return (true);
	}

	return (NodeStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult LightNodeStructure::ProcessData(DataDescription *dataDescription)
{
	DataResult result = NodeStructure::ProcessData(dataDescription);
	if (result != kDataOkay)
	{
		return (result);
	}

	bool objectFlag = false;

	const Structure *structure = GetFirstSubnode();
	while (structure)
	{
		if (structure->GetStructureType() == kStructureObjectRef)
		{
			if (objectFlag)
			{
				return (kDataExtraneousSubstructure);
			}

			objectFlag = true;

			const Structure *objectStructure = static_cast<const ObjectRefStructure *>(structure)->GetTargetStructure();
			if (objectStructure->GetStructureType() != kStructureLightObject)
			{
				return (kDataOpenGexInvalidObjectRef);
			}

			lightObjectStructure = static_cast<const LightObjectStructure *>(objectStructure);
		}

		structure = structure->Next();
	}

	if (!objectFlag)
	{
		return (kDataMissingSubstructure);
	}

	// Do application-specific node processing here.

	return (kDataOkay);
}

const ObjectStructure *LightNodeStructure::GetObjectStructure(void) const
{
	return (lightObjectStructure);
}


// Camera Node Structure
CameraNodeStructure::CameraNodeStructure() : NodeStructure(kStructureCameraNode)
{
}

CameraNodeStructure::~CameraNodeStructure()
{
}

bool CameraNodeStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	if (structure->GetStructureType() == kStructureObjectRef)
	{
		return (true);
	}

	return (NodeStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult CameraNodeStructure::ProcessData(DataDescription *dataDescription)
{
	DataResult result = NodeStructure::ProcessData(dataDescription);
	if (result != kDataOkay)
	{
		return (result);
	}

	bool objectFlag = false;

	const Structure *structure = GetFirstSubnode();
	while (structure)
	{
		if (structure->GetStructureType() == kStructureObjectRef)
		{
			if (objectFlag)
			{
				return (kDataExtraneousSubstructure);
			}

			objectFlag = true;

			const Structure *objectStructure = static_cast<const ObjectRefStructure *>(structure)->GetTargetStructure();
			if (objectStructure->GetStructureType() != kStructureCameraObject)
			{
				return (kDataOpenGexInvalidObjectRef);
			}

			cameraObjectStructure = static_cast<const CameraObjectStructure *>(objectStructure);
		}

		structure = structure->Next();
	}

	if (!objectFlag)
	{
		return (kDataMissingSubstructure);
	}

	// Do application-specific node processing here.

	return (kDataOkay);
}

const ObjectStructure *CameraNodeStructure::GetObjectStructure(void) const
{
	return (cameraObjectStructure);
}


// Vertex Array Structure
VertexArrayStructure::VertexArrayStructure() : OpenGexStructure(kStructureVertexArray)
{
	morphIndex = 0;
}

VertexArrayStructure::~VertexArrayStructure()
{
}

bool VertexArrayStructure::ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value)
{
	if (identifier == "attrib")
	{
		*type = kDataString;
		*value = &arrayAttrib;
		return (true);
	}

	if (identifier == "morph")
	{
		*type = kDataUnsignedInt32;
		*value = &morphIndex;
		return (true);
	}

	return (false);
}

bool VertexArrayStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	if (structure->GetStructureType() == kDataFloat)
	{
		return (true);
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult VertexArrayStructure::ProcessData(DataDescription *dataDescription)
{
	const Structure *structure = GetFirstCoreSubnode();
	if (!structure)
	{
		return (kDataMissingSubstructure);
	}

	if (GetLastCoreSubnode() != structure)
	{
		return (kDataExtraneousSubstructure);
	}

	const DataStructure<FloatDataType> *dataStructure = static_cast<const DataStructure<FloatDataType> *>(structure);

	int32 arraySize = dataStructure->GetArraySize();
	int32 elementCount = dataStructure->GetDataElementCount();
	int32 vertexCount = elementCount / arraySize;
	const float *data = &dataStructure->GetDataElement(0);

	// Do something with the vertex data here.

	return (kDataOkay);
}


// Index Array Structure
IndexArrayStructure::IndexArrayStructure() : OpenGexStructure(kStructureIndexArray)
{
	materialIndex = 0;
	restartIndex = 0;
	frontFace = "ccw";
}

IndexArrayStructure::~IndexArrayStructure()
{
}

bool IndexArrayStructure::ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value)
{
	if (identifier == "material")
	{
		*type = kDataUnsignedInt32;
		*value = &materialIndex;
		return (true);
	}

	if (identifier == "restart")
	{
		*type = kDataUnsignedInt64;
		*value = &restartIndex;
		return (true);
	}

	if (identifier == "front")
	{
		*type = kDataString;
		*value = &frontFace;
		return (true);
	}

	return (false);
}

bool IndexArrayStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	StructureType type = structure->GetStructureType();
	if ((type == kDataUnsignedInt8) || (type == kDataUnsignedInt16) || (type == kDataUnsignedInt32) || (type == kDataUnsignedInt64))
	{
		return (true);
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult IndexArrayStructure::ProcessData(DataDescription *dataDescription)
{
	const Structure *structure = GetFirstCoreSubnode();
	if (!structure)
	{
		return (kDataMissingSubstructure);
	}

	if (GetLastCoreSubnode() != structure)
	{
		return (kDataExtraneousSubstructure);
	}

	const PrimitiveStructure *primitiveStructure = static_cast<const PrimitiveStructure *>(structure);
	if (primitiveStructure->GetArraySize() != 3)
	{
		return (kDataInvalidDataFormat);
	}

	// Do something with the index array here.

	return (kDataOkay);
}


// Bone Ref Array Structure
BoneRefArrayStructure::BoneRefArrayStructure() : OpenGexStructure(kStructureBoneRefArray)
{
	boneNodeArray = nullptr;
}

BoneRefArrayStructure::~BoneRefArrayStructure()
{
	delete[] boneNodeArray;
}

bool BoneRefArrayStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	if (structure->GetStructureType() == kDataRef)
	{
		return (true);
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult BoneRefArrayStructure::ProcessData(DataDescription *dataDescription)
{
	const Structure *structure = GetFirstCoreSubnode();
	if (!structure)
	{
		return (kDataMissingSubstructure);
	}

	if (GetLastCoreSubnode() != structure)
	{
		return (kDataExtraneousSubstructure);
	}

	const DataStructure<RefDataType> *dataStructure = static_cast<const DataStructure<RefDataType> *>(structure);
	boneCount = dataStructure->GetDataElementCount();

	if (boneCount != 0)
	{
		boneNodeArray = new const BoneNodeStructure *[boneCount];

		for (machine a = 0; a < boneCount; a++)
		{
			const StructureRef& reference = dataStructure->GetDataElement(a);
			const Structure *boneStructure = dataDescription->FindStructure(reference);
			if (!boneStructure)
			{
				return (kDataBrokenRef);
			}

			if (boneStructure->GetStructureType() != kStructureBoneNode)
			{
				return (kDataOpenGexInvalidBoneRef);
			}

			boneNodeArray[a] = static_cast<const BoneNodeStructure *>(boneStructure);
		}
	}

	return (kDataOkay);
}


// Bone Count Array Structure
BoneCountArrayStructure::BoneCountArrayStructure() : OpenGexStructure(kStructureBoneCountArray)
{
	arrayStorage = nullptr;
}

BoneCountArrayStructure::~BoneCountArrayStructure()
{
	delete[] arrayStorage;
}

bool BoneCountArrayStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	StructureType type = structure->GetStructureType();
	if ((type == kDataUnsignedInt8) || (type == kDataUnsignedInt16) || (type == kDataUnsignedInt32) || (type == kDataUnsignedInt64))
	{
		return (true);
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult BoneCountArrayStructure::ProcessData(DataDescription *dataDescription)
{
	const Structure *structure = GetFirstCoreSubnode();
	if (!structure)
	{
		return (kDataMissingSubstructure);
	}

	if (GetLastCoreSubnode() != structure)
	{
		return (kDataExtraneousSubstructure);
	}

	const PrimitiveStructure *primitiveStructure = static_cast<const PrimitiveStructure *>(structure);
	if (primitiveStructure->GetArraySize() != 0)
	{
		return (kDataInvalidDataFormat);
	}

	StructureType type = primitiveStructure->GetStructureType();
	if (type == kDataUnsignedInt16)
	{
		const DataStructure<UnsignedInt16DataType> *dataStructure = static_cast<const DataStructure<UnsignedInt16DataType> *>(primitiveStructure);
		vertexCount = dataStructure->GetDataElementCount();
		boneCountArray = &dataStructure->GetDataElement(0);
	}
	else if (type == kDataUnsignedInt8)
	{
		const DataStructure<UnsignedInt8DataType> *dataStructure = static_cast<const DataStructure<UnsignedInt8DataType> *>(primitiveStructure);
		vertexCount = dataStructure->GetDataElementCount();

		const unsigned_int8 *data = &dataStructure->GetDataElement(0);
		arrayStorage = new unsigned_int16[vertexCount];
		boneCountArray = arrayStorage;

		for (machine a = 0; a < vertexCount; a++)
		{
			arrayStorage[a] = data[a];
		}
	}
	else if (type == kDataUnsignedInt32)
	{
		const DataStructure<UnsignedInt32DataType> *dataStructure = static_cast<const DataStructure<UnsignedInt32DataType> *>(primitiveStructure);
		vertexCount = dataStructure->GetDataElementCount();

		const unsigned_int32 *data = &dataStructure->GetDataElement(0);
		arrayStorage = new unsigned_int16[vertexCount];
		boneCountArray = arrayStorage;

		for (machine a = 0; a < vertexCount; a++)
		{
			unsigned_int32 index = data[a];
			if (index > 65535)
			{
				// We only support 16-bit counts or smaller.
				return (kDataOpenGexIndexValueUnsupported);
			}

			arrayStorage[a] = (unsigned_int16) index;
		}
	}
	else // must be 64-bit
	{
		const DataStructure<UnsignedInt64DataType> *dataStructure = static_cast<const DataStructure<UnsignedInt64DataType> *>(primitiveStructure);
		vertexCount = dataStructure->GetDataElementCount();

		const unsigned_int64 *data = &dataStructure->GetDataElement(0);
		arrayStorage = new unsigned_int16[vertexCount];
		boneCountArray = arrayStorage;

		for (machine a = 0; a < vertexCount; a++)
		{
			unsigned_int64 index = data[a];
			if (index > 65535)
			{
				// We only support 16-bit counts or smaller.
				return (kDataOpenGexIndexValueUnsupported);
			}

			arrayStorage[a] = (unsigned_int16) index;
		}
	}

	return (kDataOkay);
}


// Bone Index Array Structure
BoneIndexArrayStructure::BoneIndexArrayStructure() : OpenGexStructure(kStructureBoneIndexArray)
{
	arrayStorage = nullptr;
}

BoneIndexArrayStructure::~BoneIndexArrayStructure()
{
	delete[] arrayStorage;
}

bool BoneIndexArrayStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	StructureType type = structure->GetStructureType();
	if ((type == kDataUnsignedInt8) || (type == kDataUnsignedInt16) || (type == kDataUnsignedInt32) || (type == kDataUnsignedInt64))
	{
		return (true);
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult BoneIndexArrayStructure::ProcessData(DataDescription *dataDescription)
{
	const Structure *structure = GetFirstCoreSubnode();
	if (!structure)
	{
		return (kDataMissingSubstructure);
	}

	if (GetLastCoreSubnode() != structure)
	{
		return (kDataExtraneousSubstructure);
	}

	const PrimitiveStructure *primitiveStructure = static_cast<const PrimitiveStructure *>(structure);
	if (primitiveStructure->GetArraySize() != 0)
	{
		return (kDataInvalidDataFormat);
	}

	StructureType type = primitiveStructure->GetStructureType();
	if (type == kDataUnsignedInt16)
	{
		const DataStructure<UnsignedInt16DataType> *dataStructure = static_cast<const DataStructure<UnsignedInt16DataType> *>(primitiveStructure);
		boneIndexCount = dataStructure->GetDataElementCount();
		boneIndexArray = &dataStructure->GetDataElement(0);
	}
	else if (type == kDataUnsignedInt8)
	{
		const DataStructure<UnsignedInt8DataType> *dataStructure = static_cast<const DataStructure<UnsignedInt8DataType> *>(primitiveStructure);
		boneIndexCount = dataStructure->GetDataElementCount();

		const unsigned_int8 *data = &dataStructure->GetDataElement(0);
		arrayStorage = new unsigned_int16[boneIndexCount];
		boneIndexArray = arrayStorage;

		for (machine a = 0; a < boneIndexCount; a++)
		{
			arrayStorage[a] = data[a];
		}
	}
	else if (type == kDataUnsignedInt32)
	{
		const DataStructure<UnsignedInt32DataType> *dataStructure = static_cast<const DataStructure<UnsignedInt32DataType> *>(primitiveStructure);
		boneIndexCount = dataStructure->GetDataElementCount();

		const unsigned_int32 *data = &dataStructure->GetDataElement(0);
		arrayStorage = new unsigned_int16[boneIndexCount];
		boneIndexArray = arrayStorage;

		for (machine a = 0; a < boneIndexCount; a++)
		{
			unsigned_int32 index = data[a];
			if (index > 65535)
			{
				// We only support 16-bit indexes or smaller.
				return (kDataOpenGexIndexValueUnsupported);
			}

			arrayStorage[a] = (unsigned_int16) index;
		}
	}
	else // must be 64-bit
	{
		const DataStructure<UnsignedInt64DataType> *dataStructure = static_cast<const DataStructure<UnsignedInt64DataType> *>(primitiveStructure);
		boneIndexCount = dataStructure->GetDataElementCount();

		const unsigned_int64 *data = &dataStructure->GetDataElement(0);
		arrayStorage = new unsigned_int16[boneIndexCount];
		boneIndexArray = arrayStorage;

		for (machine a = 0; a < boneIndexCount; a++)
		{
			unsigned_int64 index = data[a];
			if (index > 65535)
			{
				// We only support 16-bit indexes or smaller.
				return (kDataOpenGexIndexValueUnsupported);
			}

			arrayStorage[a] = (unsigned_int16) index;
		}
	}

	return (kDataOkay);
}


// Bone Wight Array Structure
BoneWeightArrayStructure::BoneWeightArrayStructure() : OpenGexStructure(kStructureBoneWeightArray)
{
}

BoneWeightArrayStructure::~BoneWeightArrayStructure()
{
}

bool BoneWeightArrayStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	if (structure->GetStructureType() == kDataFloat)
	{
		return (true);
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult BoneWeightArrayStructure::ProcessData(DataDescription *dataDescription)
{
	const Structure *structure = GetFirstCoreSubnode();
	if (!structure)
	{
		return (kDataMissingSubstructure);
	}

	if (GetLastCoreSubnode() != structure)
	{
		return (kDataExtraneousSubstructure);
	}

	const DataStructure<FloatDataType> *dataStructure = static_cast<const DataStructure<FloatDataType> *>(structure);
	if (dataStructure->GetArraySize() != 0)
	{
		return (kDataInvalidDataFormat);
	}

	boneWeightCount = dataStructure->GetDataElementCount();
	boneWeightArray = &dataStructure->GetDataElement(0);
	return (kDataOkay);
}


// Skeleton Structure
SkeletonStructure::SkeletonStructure() : OpenGexStructure(kStructureSkeleton)
{
}

SkeletonStructure::~SkeletonStructure()
{
}

bool SkeletonStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	StructureType type = structure->GetStructureType();
	if ((type == kStructureBoneRefArray) || (type == kStructureTransform))
	{
		return (true);
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult SkeletonStructure::ProcessData(DataDescription *dataDescription)
{
	DataResult result = Structure::ProcessData(dataDescription);
	if (result != kDataOkay)
	{
		return (result);
	}

	const Structure *structure = GetFirstSubstructure(kStructureBoneRefArray);
	if (!structure)
	{
		return (kDataMissingSubstructure);
	}

	if (GetLastSubstructure(kStructureBoneRefArray) != structure)
	{
		return (kDataExtraneousSubstructure);
	}

	boneRefArrayStructure = static_cast<const BoneRefArrayStructure *>(structure);

	structure = GetFirstSubstructure(kStructureTransform);
	if (!structure)
	{
		return (kDataMissingSubstructure);
	}

	if (GetLastSubstructure(kStructureTransform) != structure)
	{
		return (kDataExtraneousSubstructure);
	}

	transformStructure = static_cast<const TransformStructure *>(structure);

	if (boneRefArrayStructure->GetBoneCount() != transformStructure->GetTransformCount())
	{
		return (kDataOpenGexBoneCountMismatch);
	}

	return (kDataOkay);
}


// Skin Structure
SkinStructure::SkinStructure() : OpenGexStructure(kStructureSkin)
{
}

SkinStructure::~SkinStructure()
{
}

bool SkinStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	StructureType type = structure->GetStructureType();
	if ((type == kStructureTransform) || (type == kStructureSkeleton) || (type == kStructureBoneCountArray) || (type == kStructureBoneIndexArray) || (type == kStructureBoneWeightArray))
	{
		return (true);
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult SkinStructure::ProcessData(DataDescription *dataDescription)
{
	DataResult result = Structure::ProcessData(dataDescription);
	if (result != kDataOkay)
	{
		return (result);
	}

	const Structure *structure = GetFirstSubstructure(kStructureTransform);
	if (structure)
	{
		if (GetLastSubstructure(kStructureTransform) != structure)
		{
			return (kDataExtraneousSubstructure);
		}

		// Process skin transform here.
	}

	structure = GetFirstSubstructure(kStructureSkeleton);
	if (!structure)
	{
		return (kDataMissingSubstructure);
	}

	if (GetLastSubstructure(kStructureSkeleton) != structure)
	{
		return (kDataExtraneousSubstructure);
	}

	skeletonStructure = static_cast<const SkeletonStructure *>(structure);

	structure = GetFirstSubstructure(kStructureBoneCountArray);
	if (!structure)
	{
		return (kDataMissingSubstructure);
	}

	if (GetLastSubstructure(kStructureBoneCountArray) != structure)
	{
		return (kDataExtraneousSubstructure);
	}

	boneCountArrayStructure = static_cast<const BoneCountArrayStructure *>(structure);

	structure = GetFirstSubstructure(kStructureBoneIndexArray);
	if (!structure)
	{
		return (kDataMissingSubstructure);
	}

	if (GetLastSubstructure(kStructureBoneIndexArray) != structure)
	{
		return (kDataExtraneousSubstructure);
	}

	boneIndexArrayStructure = static_cast<const BoneIndexArrayStructure *>(structure);

	structure = GetFirstSubstructure(kStructureBoneWeightArray);
	if (!structure)
	{
		return (kDataMissingSubstructure);
	}

	if (GetLastSubstructure(kStructureBoneWeightArray) != structure)
	{
		return (kDataExtraneousSubstructure);
	}

	boneWeightArrayStructure = static_cast<const BoneWeightArrayStructure *>(structure);

	int32 boneIndexCount = boneIndexArrayStructure->GetBoneIndexCount();
	if (boneWeightArrayStructure->GetBoneWeightCount() != boneIndexCount)
	{
		return (kDataOpenGexBoneWeightCountMismatch);
	}

	int32 vertexCount = boneCountArrayStructure->GetVertexCount();
	const unsigned_int16 *boneCountArray = boneCountArrayStructure->GetBoneCountArray();

	int32 boneWeightCount = 0;
	for (machine a = 0; a < vertexCount; a++)
	{
		unsigned_int32 count = boneCountArray[a];
		boneWeightCount += count;
	}

	if (boneWeightCount != boneIndexCount)
	{
		return (kDataOpenGexBoneWeightCountMismatch);
	}

	// Do application-specific skin processing here.

	return (kDataOkay);
}


// Morph Structure
MorphStructure::MorphStructure() : OpenGexStructure(kStructureMorph)
{
	// The value of baseFlag indicates whether the base property was actually
	// specified for the structure.

	morphIndex = 0;
	baseFlag = false;
}

MorphStructure::~MorphStructure()
{
}

bool MorphStructure::ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value)
{
	if (identifier == "index")
	{
		*type = kDataUnsignedInt32;
		*value = &morphIndex;
		return (true);
	}

	if (identifier == "base")
	{
		*type = kDataUnsignedInt32;
		*value = &baseIndex;
		baseFlag = true;
		return (true);
	}

	return (false);
}

bool MorphStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	if (structure->GetStructureType() == kStructureName)
	{
		return (true);
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult MorphStructure::ProcessData(DataDescription *dataDescription)
{
	DataResult result = OpenGexStructure::ProcessData(dataDescription);
	if (result != kDataOkay)
	{
		return (result);
	}

	morphName = nullptr;

	const Structure *structure = GetFirstCoreSubnode();
	if (!structure)
	{
		return (kDataMissingSubstructure);
	}

	if (GetLastCoreSubnode() != structure)
	{
		return (kDataExtraneousSubstructure);
	}

	morphName = static_cast<const NameStructure *>(structure)->GetName();

	// Do application-specific morph processing here.

	return (kDataOkay);
}


MeshStructure::MeshStructure() : OpenGexStructure(kStructureMesh)
{
	meshLevel = 0;

	skinStructure = nullptr;
}

MeshStructure::~MeshStructure()
{
}

bool MeshStructure::ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value)
{
	if (identifier == "lod")
	{
		*type = kDataUnsignedInt32;
		*value = &meshLevel;
		return (true);
	}

	if (identifier == "primitive")
	{
		*type = kDataString;
		*value = &meshPrimitive;
		return (true);
	}

	return (false);
}

bool MeshStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	StructureType type = structure->GetStructureType();
	if ((type == kStructureVertexArray) || (type == kStructureIndexArray) || (type == kStructureSkin))
	{
		return (true);
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult MeshStructure::ProcessData(DataDescription *dataDescription)
{
	DataResult result = Structure::ProcessData(dataDescription);
	if (result != kDataOkay)
	{
		return (result);
	}

	Structure *structure = GetFirstSubnode();
	while (structure)
	{
		StructureType type = structure->GetStructureType();
		if (type == kStructureVertexArray)
		{
			const VertexArrayStructure *vertexArrayStructure = static_cast<const VertexArrayStructure *>(structure);

			// Process vertex array here.
		}
		else if (type == kStructureIndexArray)
		{
			IndexArrayStructure *indexArrayStructure = static_cast<IndexArrayStructure *>(structure);

			// Process index array here.
		}
		else if (type == kStructureSkin)
		{
			if (skinStructure)
			{
				return (kDataExtraneousSubstructure);
			}

			skinStructure = static_cast<SkinStructure *>(structure);
		}

		structure = structure->Next();
	}

	// Do application-specific mesh processing here.

	return (kDataOkay);
}


// Object Structure
ObjectStructure::ObjectStructure(StructureType type) : OpenGexStructure(type)
{
	SetBaseStructureType(kStructureObject);
}

ObjectStructure::~ObjectStructure()
{
}


// Geometry Object Structure
GeometryObjectStructure::GeometryObjectStructure() : ObjectStructure(kStructureGeometryObject)
{
	visibleFlag = true;
	shadowFlag = true;
	motionBlurFlag = true;
}

GeometryObjectStructure::~GeometryObjectStructure()
{
	meshMap.RemoveAll();
}

bool GeometryObjectStructure::ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value)
{
	if (identifier == "visible")
	{
		*type = kDataBool;
		*value = &visibleFlag;
		return (true);
	}

	if (identifier == "shadow")
	{
		*type = kDataBool;
		*value = &shadowFlag;
		return (true);
	}

	if (identifier == "motion_blur")
	{
		*type = kDataBool;
		*value = &motionBlurFlag;
		return (true);
	}

	return (false);
}

bool GeometryObjectStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	StructureType type = structure->GetStructureType();
	if ((type == kStructureMesh) || (type == kStructureMorph))
	{
		return (true);
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult GeometryObjectStructure::ProcessData(DataDescription *dataDescription)
{
	DataResult result = Structure::ProcessData(dataDescription);
	if (result != kDataOkay)
	{
		return (result);
	}

	int32 meshCount = 0;
	int32 skinCount = 0;

	Structure *structure = GetFirstCoreSubnode();
	while (structure)
	{
		StructureType type = structure->GetStructureType();
		if (type == kStructureMesh)
		{
			MeshStructure *meshStructure = static_cast<MeshStructure *>(structure);
			if (!meshMap.Insert(meshStructure))
			{
				return (kDataOpenGexDuplicateLod);
			}

			meshCount++;
			skinCount += (meshStructure->GetSkinStructure() != nullptr);
		}
		else if (type == kStructureMorph)
		{
			MorphStructure *morphStructure = static_cast<MorphStructure *>(structure);
			if (!morphMap.Insert(morphStructure))
			{
				return (kDataOpenGexDuplicateMorph);
			}
		}

		structure = structure->Next();
	}

	if (meshCount == 0)
	{
		return (kDataMissingSubstructure);
	}

	if ((skinCount != 0) && (skinCount != meshCount))
	{
		return (kDataOpenGexMissingLodSkin);
	}

	// Do application-specific object processing here.

	return (kDataOkay);
}


// Light Object Structure
LightObjectStructure::LightObjectStructure() : ObjectStructure(kStructureLightObject)
{
	shadowFlag = true;
}

LightObjectStructure::~LightObjectStructure()
{
}

bool LightObjectStructure::ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value)
{
	if (identifier == "type")
	{
		*type = kDataString;
		*value = &typeString;
		return (true);
	}

	if (identifier == "shadow")
	{
		*type = kDataBool;
		*value = &shadowFlag;
		return (true);
	}

	return (false);
}

bool LightObjectStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	if ((structure->GetBaseStructureType() == kStructureAttrib) || (structure->GetStructureType() == kStructureAtten))
	{
		return (true);
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult LightObjectStructure::ProcessData(DataDescription *dataDescription)
{
	DataResult result = Structure::ProcessData(dataDescription);
	if (result != kDataOkay)
	{
		return (result);
	}

	if (typeString == "infinite")
	{
		// Prepare to handle infinite light here.
	}
	else if (typeString == "point")
	{
		// Prepare to handle point light here.
	}
	else if (typeString == "spot")
	{
		// Prepare to handle spot light here.
	}
	else
	{
		return (kDataOpenGexUndefinedLightType);
	}

	const Structure *structure = GetFirstSubnode();
	while (structure)
	{
		StructureType type = structure->GetStructureType();
		if (type == kStructureColor)
		{
			const ColorStructure *colorStructure = static_cast<const ColorStructure *>(structure);
			if (colorStructure->GetAttribString() == "light")
			{
				// Process light color here.
			}
		}
		else if (type == kStructureParam)
		{
			const ParamStructure *paramStructure = static_cast<const ParamStructure *>(structure);
			if (paramStructure->GetAttribString() == "intensity")
			{
				// Process light intensity here.
			}
		}
		else if (type == kStructureTexture)
		{
			const TextureStructure *textureStructure = static_cast<const TextureStructure *>(structure);
			if (textureStructure->GetAttribString() == "projection")
			{
				const char *textureName = textureStructure->GetTextureName();

				// Process light texture here.
			}
		}
		else if (type == kStructureAtten)
		{
			const AttenStructure *attenStructure = static_cast<const AttenStructure *>(structure);
			const String& attenKind = attenStructure->GetAttenKind();
			const String& curveType = attenStructure->GetCurveType();

			if (attenKind == "distance")
			{
				if ((curveType == "linear") || (curveType == "smooth"))
				{
					float beginParam = attenStructure->GetBeginParam();
					float endParam = attenStructure->GetEndParam();

					// Process linear or smooth attenuation here.
				}
				else if (curveType == "inverse")
				{
					float scaleParam = attenStructure->GetScaleParam();
					float linearParam = attenStructure->GetLinearParam();

					// Process inverse attenuation here.
				}
				else if (curveType == "inverse_square")
				{
					float scaleParam = attenStructure->GetScaleParam();
					float quadraticParam = attenStructure->GetQuadraticParam();

					// Process inverse square attenuation here.
				}
				else
				{
					return (kDataOpenGexUndefinedCurve);
				}
			}
			else if (attenKind == "angle")
			{
				float endParam = attenStructure->GetEndParam();

				// Process angular attenutation here.
			}
			else if (attenKind == "cos_angle")
			{
				float endParam = attenStructure->GetEndParam();

				// Process angular attenutation here.
			}
			else
			{
				return (kDataOpenGexUndefinedAtten);
			}
		}

		structure = structure->Next();
	}

	// Do application-specific object processing here.

	return (kDataOkay);
}


// Camera Object Structure
CameraObjectStructure::CameraObjectStructure() : ObjectStructure(kStructureCameraObject)
{
}

CameraObjectStructure::~CameraObjectStructure()
{
}

bool CameraObjectStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	if (structure->GetStructureType() == kStructureParam)
	{
		return (true);
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult CameraObjectStructure::ProcessData(DataDescription *dataDescription)
{
	DataResult result = Structure::ProcessData(dataDescription);
	if (result != kDataOkay)
	{
		return (result);
	}

	focalLength = 2.0F;
	nearDepth = 0.1F;
	farDepth = 1000.0F;

	const OpenGexDataDescription *openGexDataDescription = static_cast<OpenGexDataDescription *>(dataDescription);
	float distanceScale = openGexDataDescription->GetDistanceScale();
	float angleScale = openGexDataDescription->GetAngleScale();

	const Structure *structure = GetFirstSubnode();
	while (structure)
	{
		if (structure->GetStructureType() == kStructureParam)
		{
			const ParamStructure *paramStructure = static_cast<const ParamStructure *>(structure);
			const String& attribString = paramStructure->GetAttribString();
			float param = paramStructure->GetParam();

			if ((attribString == "fov") || (attribString == "fovx"))
			{
				float t = tanf(param * angleScale * 0.5F);
				if (t > 0.0F)
				{
					focalLength = 1.0F / t;
				}
			}
			else if (attribString == "near")
			{
				if (param > 0.0F)
				{
					nearDepth = param * distanceScale;
				}
			}
			else if (attribString == "far")
			{
				if (param > 0.0F)
				{
					farDepth = param * distanceScale;
				}
			}
		}

		structure = structure->Next();
	}

	// Do application-specific object processing here.

	return (kDataOkay);
}


// Attribute Structure
AttribStructure::AttribStructure(StructureType type) : OpenGexStructure(type)
{
	SetBaseStructureType(kStructureAttrib);
}

AttribStructure::~AttribStructure()
{
}

bool AttribStructure::ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value)
{
	if (identifier == "attrib")
	{
		*type = kDataString;
		*value = &attribString;
		return (true);
	}

	return (false);
}


// Parameter Structure
ParamStructure::ParamStructure() : AttribStructure(kStructureParam)
{
}

ParamStructure::~ParamStructure()
{
}

bool ParamStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	if (structure->GetStructureType() == kDataFloat)
	{
		const DataStructure<FloatDataType> *dataStructure = static_cast<const DataStructure<FloatDataType> *>(structure);
		unsigned_int32 arraySize = dataStructure->GetArraySize();
		return (arraySize == 0);
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult ParamStructure::ProcessData(DataDescription *dataDescription)
{
	const Structure *structure = GetFirstCoreSubnode();
	if (!structure)
	{
		return (kDataMissingSubstructure);
	}

	if (GetLastCoreSubnode() != structure)
	{
		return (kDataExtraneousSubstructure);
	}

	const DataStructure<FloatDataType> *dataStructure = static_cast<const DataStructure<FloatDataType> *>(structure);
	if (dataStructure->GetDataElementCount() == 1)
	{
		param = dataStructure->GetDataElement(0);
	}
	else
	{
		return (kDataInvalidDataFormat);
	}

	return (kDataOkay);
}


// Color Structure
ColorStructure::ColorStructure() : AttribStructure(kStructureColor)
{
}

ColorStructure::~ColorStructure()
{
}

bool ColorStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	if (structure->GetStructureType() == kDataFloat)
	{
		const DataStructure<FloatDataType> *dataStructure = static_cast<const DataStructure<FloatDataType> *>(structure);
		unsigned_int32 arraySize = dataStructure->GetArraySize();
		return ((arraySize >= 3) && (arraySize <= 4));
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult ColorStructure::ProcessData(DataDescription *dataDescription)
{
	const Structure *structure = GetFirstCoreSubnode();
	if (!structure)
	{
		return (kDataMissingSubstructure);
	}

	if (GetLastCoreSubnode() != structure)
	{
		return (kDataExtraneousSubstructure);
	}

	const DataStructure<FloatDataType> *dataStructure = static_cast<const DataStructure<FloatDataType> *>(structure);
	unsigned_int32 arraySize = dataStructure->GetArraySize();
	if (dataStructure->GetDataElementCount() == arraySize)
	{
		const float *data = &dataStructure->GetDataElement(0);

		color[0] = data[0];
		color[1] = data[1];
		color[2] = data[2];

		if (arraySize == 3)
		{
			color[3] = 1.0F;
		}
		else
		{
			color[3] = data[3];
		}
	}
	else
	{
		return (kDataInvalidDataFormat);
	}

	return (kDataOkay);
}


// Texture Structure
TextureStructure::TextureStructure() : AttribStructure(kStructureTexture)
{
	texcoordIndex = 0;
}

TextureStructure::~TextureStructure()
{
}

bool TextureStructure::ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value)
{
	if (identifier == "texcoord")
	{
		*type = kDataUnsignedInt32;
		*value = &texcoordIndex;
		return (true);
	}

	return (AttribStructure::ValidateProperty(dataDescription, identifier, type, value));
}

bool TextureStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	StructureType type = structure->GetStructureType();
	if ((type == kDataString) || (type == kStructureAnimation) || (structure->GetBaseStructureType() == kStructureMatrix))
	{
		return (true);
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult TextureStructure::ProcessData(DataDescription *dataDescription)
{
	DataResult result = AttribStructure::ProcessData(dataDescription);
	if (result != kDataOkay)
	{
		return (result);
	}

	bool nameFlag = false;

	const Structure *structure = GetFirstSubnode();
	while (structure)
	{
		if (structure->GetStructureType() == kDataString)
		{
			if (!nameFlag)
			{
				nameFlag = true;

				const DataStructure<StringDataType> *dataStructure = static_cast<const DataStructure<StringDataType> *>(structure);
				if (dataStructure->GetDataElementCount() == 1)
				{
					textureName = dataStructure->GetDataElement(0);
				}
				else
				{
					return (kDataInvalidDataFormat);
				}
			}
			else
			{
				return (kDataExtraneousSubstructure);
			}
		}
		else if (structure->GetBaseStructureType() == kStructureMatrix)
		{
			const MatrixStructure *matrixStructure = static_cast<const MatrixStructure *>(structure);

			// Process transform matrix here.
		}

		structure = structure->Next();
	}

	if (!nameFlag)
	{
		return (kDataMissingSubstructure);
	}

	return (kDataOkay);
}


// Atten Structure
AttenStructure::AttenStructure() :
		OpenGexStructure(kStructureAtten),
		attenKind("distance"),
		curveType("linear")
{
	beginParam = 0.0F;
	endParam = 1.0F;

	scaleParam = 1.0F;
	offsetParam = 0.0F;

	constantParam = 0.0F;
	linearParam = 0.0F;
	quadraticParam = 1.0F;

	powerParam = 1.0F;
}

AttenStructure::~AttenStructure()
{
}

bool AttenStructure::ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value)
{
	if (identifier == "kind")
	{
		*type = kDataString;
		*value = &attenKind;
		return (true);
	}

	if (identifier == "curve")
	{
		*type = kDataString;
		*value = &curveType;
		return (true);
	}

	return (false);
}

bool AttenStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	if (structure->GetStructureType() == kStructureParam)
	{
		return (true);
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult AttenStructure::ProcessData(DataDescription *dataDescription)
{
	DataResult result = Structure::ProcessData(dataDescription);
	if (result != kDataOkay)
	{
		return (result);
	}

	if (curveType == "inverse")
	{
		linearParam = 1.0F;
	}

	const OpenGexDataDescription *openGexDataDescription = static_cast<OpenGexDataDescription *>(dataDescription);
	float distanceScale = openGexDataDescription->GetDistanceScale();
	float angleScale = openGexDataDescription->GetAngleScale();

	const Structure *structure = GetFirstSubnode();
	while (structure)
	{
		if (structure->GetStructureType() == kStructureParam)
		{
			const ParamStructure *paramStructure = static_cast<const ParamStructure *>(structure);
			const String& attribString = paramStructure->GetAttribString();

			if (attribString == "begin")
			{
				beginParam = paramStructure->GetParam();

				if (attenKind == "distance")
				{
					beginParam *= distanceScale;
				}
				else if (attenKind == "angle")
				{
					beginParam *= angleScale;
				}
			}
			else if (attribString == "end")
			{
				endParam = paramStructure->GetParam();

				if (attenKind == "distance")
				{
					endParam *= distanceScale;
				}
				else if (attenKind == "angle")
				{
					endParam *= angleScale;
				}
			}
			else if (attribString == "scale")
			{
				scaleParam = paramStructure->GetParam();

				if (attenKind == "distance")
				{
					scaleParam *= distanceScale;
				}
				else if (attenKind == "angle")
				{
					scaleParam *= angleScale;
				}
			}
			else if (attribString == "offset")
			{
				offsetParam = paramStructure->GetParam();
			}
			else if (attribString == "constant")
			{
				constantParam = paramStructure->GetParam();
			}
			else if (attribString == "linear")
			{
				linearParam = paramStructure->GetParam();
			}
			else if (attribString == "quadratic")
			{
				quadraticParam = paramStructure->GetParam();
			}
			else if (attribString == "power")
			{
				powerParam = paramStructure->GetParam();
			}
		}

		structure = structure->Next();
	}

	return (kDataOkay);
}


// Material Structure
MaterialStructure::MaterialStructure() : OpenGexStructure(kStructureMaterial)
{
	twoSidedFlag = false;
	materialName = nullptr;
}

MaterialStructure::~MaterialStructure()
{
}

bool MaterialStructure::ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value)
{
	if (identifier == "two_sided")
	{
		*type = kDataBool;
		*value = &twoSidedFlag;
		return (true);
	}

	return (false);
}

bool MaterialStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	if ((structure->GetBaseStructureType() == kStructureAttrib) || (structure->GetStructureType() == kStructureName))
	{
		return (true);
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult MaterialStructure::ProcessData(DataDescription *dataDescription)
{
	DataResult result = Structure::ProcessData(dataDescription);
	if (result != kDataOkay)
	{
		return (result);
	}

	const Structure *structure = GetFirstSubstructure(kStructureName);
	if (structure)
	{
		if (GetLastSubstructure(kStructureName) != structure)
		{
			return (kDataExtraneousSubstructure);
		}
	}

	// Do application-specific material processing here.

	return (kDataOkay);
}


// Key Structure
KeyStructure::KeyStructure() :
		OpenGexStructure(kStructureKey),
		keyKind("value")
{
}

KeyStructure::~KeyStructure()
{
}

bool KeyStructure::ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value)
{
	if (identifier == "kind")
	{
		*type = kDataString;
		*value = &keyKind;
		return (true);
	}

	return (false);
}

bool KeyStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	if (structure->GetStructureType() == kDataFloat)
	{
		return (true);
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult KeyStructure::ProcessData(DataDescription *dataDescription)
{
	DataResult result = Structure::ProcessData(dataDescription);
	if (result != kDataOkay)
	{
		return (result);
	}

	const Structure *structure = GetFirstCoreSubnode();
	if (!structure)
	{
		return (kDataMissingSubstructure);
	}

	if (GetLastCoreSubnode() != structure)
	{
		return (kDataExtraneousSubstructure);
	}

	const DataStructure<FloatDataType> *dataStructure = static_cast<const DataStructure<FloatDataType> *>(structure);
	if (dataStructure->GetDataElementCount() == 0)
	{
		return (kDataOpenGexEmptyKeyStructure);
	}

	if ((keyKind == "value") || (keyKind == "-control") || (keyKind == "+control"))
	{
		scalarFlag = false;
	}
	else if ((keyKind == "tension") || (keyKind == "continuity") || (keyKind == "bias"))
	{
		scalarFlag = true;

		if (dataStructure->GetArraySize() != 0)
		{
			return (kDataInvalidDataFormat);
		}
	}
	else
	{
		return (kDataOpenGexInvalidKeyKind);
	}

	return (kDataOkay);
}


// Curve Structure
CurveStructure::CurveStructure(StructureType type) :
		OpenGexStructure(type),
		curveType("linear")
{
	SetBaseStructureType(kStructureCurve);
}

CurveStructure::~CurveStructure()
{
}

bool CurveStructure::ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value)
{
	if (identifier == "curve")
	{
		*type = kDataString;
		*value = &curveType;
		return (true);
	}

	return (false);
}

bool CurveStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	if (structure->GetStructureType() == kStructureKey)
	{
		return (true);
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult CurveStructure::ProcessData(DataDescription *dataDescription)
{
	DataResult result = Structure::ProcessData(dataDescription);
	if (result != kDataOkay)
	{
		return (result);
	}

	keyValueStructure = nullptr;
	keyControlStructure[0] = nullptr;
	keyControlStructure[1] = nullptr;
	keyTensionStructure = nullptr;
	keyContinuityStructure = nullptr;
	keyBiasStructure = nullptr;

	const Structure *structure = GetFirstSubnode();
	while (structure)
	{
		if (structure->GetStructureType() == kStructureKey)
		{
			const KeyStructure *keyStructure = static_cast<const KeyStructure *>(structure);
			const String& keyKind = keyStructure->GetKeyKind();

			if (keyKind == "value")
			{
				if (!keyValueStructure)
				{
					keyValueStructure = keyStructure;
				}
				else
				{
					return (kDataExtraneousSubstructure);
				}
			}
			else if (keyKind == "-control")
			{
				if (curveType != "bezier")
				{
					return (kDataOpenGexInvalidKeyKind);
				}

				if (!keyControlStructure[0])
				{
					keyControlStructure[0] = keyStructure;
				}
				else
				{
					return (kDataExtraneousSubstructure);
				}
			}
			else if (keyKind == "+control")
			{
				if (curveType != "bezier")
				{
					return (kDataOpenGexInvalidKeyKind);
				}

				if (!keyControlStructure[1])
				{
					keyControlStructure[1] = keyStructure;
				}
				else
				{
					return (kDataExtraneousSubstructure);
				}
			}
			else if (keyKind == "tension")
			{
				if (curveType != "tcb")
				{
					return (kDataOpenGexInvalidKeyKind);
				}

				if (!keyTensionStructure)
				{
					keyTensionStructure = keyStructure;
				}
				else
				{
					return (kDataExtraneousSubstructure);
				}
			}
			else if (keyKind == "continuity")
			{
				if (curveType != "tcb")
				{
					return (kDataOpenGexInvalidKeyKind);
				}

				if (!keyContinuityStructure)
				{
					keyContinuityStructure = keyStructure;
				}
				else
				{
					return (kDataExtraneousSubstructure);
				}
			}
			else if (keyKind == "bias")
			{
				if (curveType != "tcb")
				{
					return (kDataOpenGexInvalidKeyKind);
				}

				if (!keyBiasStructure)
				{
					keyBiasStructure = keyStructure;
				}
				else
				{
					return (kDataExtraneousSubstructure);
				}
			}
		}

		structure = structure->Next();
	}

	if (!keyValueStructure)
	{
		return (kDataMissingSubstructure);
	}

	if (curveType == "bezier")
	{
		if ((!keyControlStructure[0]) || (!keyControlStructure[1]))
		{
			return (kDataMissingSubstructure);
		}
	}
	else if (curveType == "tcb")
	{
		if ((!keyTensionStructure) || (!keyContinuityStructure) || (!keyBiasStructure))
		{
			return (kDataMissingSubstructure);
		}
	}

	return (kDataOkay);
}


// Time Structure
TimeStructure::TimeStructure() : CurveStructure(kStructureTime)
{
}

TimeStructure::~TimeStructure()
{
}

DataResult TimeStructure::ProcessData(DataDescription *dataDescription)
{
	DataResult result = CurveStructure::ProcessData(dataDescription);
	if (result != kDataOkay)
	{
		return (result);
	}

	const String& curveType = GetCurveType();
	if ((curveType != "linear") && (curveType != "bezier"))
	{
		return (kDataOpenGexInvalidCurveType);
	}

	int32 elementCount = 0;

	const Structure *structure = GetFirstSubnode();
	while (structure)
	{
		if (structure->GetStructureType() == kStructureKey)
		{
			const KeyStructure *keyStructure = static_cast<const KeyStructure *>(structure);
			const DataStructure<FloatDataType> *dataStructure = static_cast<DataStructure<FloatDataType> *>(keyStructure->GetFirstCoreSubnode());
			if (dataStructure->GetArraySize() != 0)
			{
				return (kDataInvalidDataFormat);
			}

			int32 count = dataStructure->GetDataElementCount();
			if (elementCount == 0)
			{
				elementCount = count;
			}
			else if (count != elementCount)
			{
				return (kDataOpenGexKeyCountMismatch);
			}
		}

		structure = structure->Next();
	}

	keyDataElementCount = elementCount;
	return (kDataOkay);
}


// Value Structure
ValueStructure::ValueStructure() : CurveStructure(kStructureValue)
{
}

ValueStructure::~ValueStructure()
{
}

DataResult ValueStructure::ProcessData(DataDescription *dataDescription)
{
	DataResult result = CurveStructure::ProcessData(dataDescription);
	if (result != kDataOkay)
	{
		return (result);
	}

	const String& curveType = GetCurveType();
	if ((curveType != "linear") && (curveType != "bezier") && (curveType != "tcb"))
	{
		return (kDataOpenGexInvalidCurveType);
	}

	const AnimatableStructure *targetStructure = static_cast<TrackStructure *>(GetSuperNode())->GetTargetStructure();
	const Structure *targetDataStructure = targetStructure->GetFirstCoreSubnode();
	if ((targetDataStructure) && (targetDataStructure->GetStructureType() == kDataFloat))
	{
		unsigned_int32 targetArraySize = static_cast<const PrimitiveStructure *>(targetDataStructure)->GetArraySize();
		int32 elementCount = 0;

		const Structure *structure = GetFirstSubnode();
		while (structure)
		{
			if (structure->GetStructureType() == kStructureKey)
			{
				const KeyStructure *keyStructure = static_cast<const KeyStructure *>(structure);
				const DataStructure<FloatDataType> *dataStructure = static_cast<DataStructure<FloatDataType> *>(keyStructure->GetFirstCoreSubnode());
				unsigned_int32 arraySize = dataStructure->GetArraySize();

				if ((!keyStructure->GetScalarFlag()) && (arraySize != targetArraySize))
				{
					return (kDataInvalidDataFormat);
				}

				int32 count = dataStructure->GetDataElementCount() / Max(arraySize, 1);
				if (elementCount == 0)
				{
					elementCount = count;
				}
				else if (count != elementCount)
				{
					return (kDataOpenGexKeyCountMismatch);
				}
			}

			structure = structure->Next();
		}

		keyDataElementCount = elementCount;
	}

	return (kDataOkay);
}


// Track Structure
TrackStructure::TrackStructure() : OpenGexStructure(kStructureTrack)
{
	targetStructure = nullptr;
}

TrackStructure::~TrackStructure()
{
}

bool TrackStructure::ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value)
{
	if (identifier == "target")
	{
		*type = kDataRef;
		*value = &targetRef;
		return (true);
	}

	return (false);
}

bool TrackStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	if (structure->GetBaseStructureType() == kStructureCurve)
	{
		return (true);
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult TrackStructure::ProcessData(DataDescription *dataDescription)
{
	if (targetRef.GetGlobalRefFlag())
	{
		return (kDataOpenGexTargetRefNotLocal);
	}

	Structure *target = GetSuperNode()->GetSuperNode()->FindStructure(targetRef);
	if (!target)
	{
		return (kDataBrokenRef);
	}

	if ((target->GetBaseStructureType() != kStructureMatrix) && (target->GetStructureType() != kStructureMorphWeight))
	{
		return (kDataOpenGexInvalidTargetStruct);
	}

	targetStructure = static_cast<AnimatableStructure *>(target);

	timeStructure = nullptr;
	valueStructure = nullptr;

	const Structure *structure = GetFirstSubnode();
	while (structure)
	{
		StructureType type = structure->GetStructureType();
		if (type == kStructureTime)
		{
			if (!timeStructure)
			{
				timeStructure = static_cast<const TimeStructure *>(structure);
			}
			else
			{
				return (kDataExtraneousSubstructure);
			}
		}
		else if (type == kStructureValue)
		{
			if (!valueStructure)
			{
				valueStructure = static_cast<const ValueStructure *>(structure);
			}
			else
			{
				return (kDataExtraneousSubstructure);
			}
		}

		structure = structure->Next();
	}

	if ((!timeStructure) || (!valueStructure))
	{
		return (kDataMissingSubstructure);
	}

	DataResult result = Structure::ProcessData(dataDescription);
	if (result != kDataOkay)
	{
		return (result);
	}

	if (timeStructure->GetKeyDataElementCount() != valueStructure->GetKeyDataElementCount())
	{
		return (kDataOpenGexKeyCountMismatch);
	}

	// Do application-specific track processing here.

	return (kDataOkay);
}


// Animation Structure
AnimationStructure::AnimationStructure() : OpenGexStructure(kStructureAnimation)
{
	clipIndex = 0;
	beginFlag = false;
	endFlag = false;
}

AnimationStructure::~AnimationStructure()
{
}

bool AnimationStructure::ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value)
{
	if (identifier == "clip")
	{
		*type = kDataInt32;
		*value = &clipIndex;
		return (true);
	}

	if (identifier == "begin")
	{
		beginFlag = true;
		*type = kDataFloat;
		*value = &beginTime;
		return (true);
	}

	if (identifier == "end")
	{
		endFlag = true;
		*type = kDataFloat;
		*value = &endTime;
		return (true);
	}

	return (false);
}

bool AnimationStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	if (structure->GetStructureType() == kStructureTrack)
	{
		return (true);
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult AnimationStructure::ProcessData(DataDescription *dataDescription)
{
	DataResult result = Structure::ProcessData(dataDescription);
	if (result != kDataOkay)
	{
		return (result);
	}

	if (!GetFirstSubstructure(kStructureTrack))
	{
		return (kDataMissingSubstructure);
	}

	// Do application-specific animation processing here.

	return (kDataOkay);
}


// Clip Structure
ClipStructure::ClipStructure() : OpenGexStructure(kStructureClip)
{
	clipIndex = 0;
}

ClipStructure::~ClipStructure()
{
}

bool ClipStructure::ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value)
{
	if (identifier == "index")
	{
		*type = kDataUnsignedInt32;
		*value = &clipIndex;
		return (true);
	}

	return (false);
}

bool ClipStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	StructureType type = structure->GetStructureType();
	if ((type == kStructureName) || (type == kStructureParam))
	{
		return (true);
	}

	return (OpenGexStructure::ValidateSubstructure(dataDescription, structure));
}

DataResult ClipStructure::ProcessData(DataDescription *dataDescription)
{
	DataResult result = Structure::ProcessData(dataDescription);
	if (result != kDataOkay)
	{
		return (result);
	}

	frameRate = 0.0F;
	clipName = nullptr;

	const Structure *structure = GetFirstSubnode();
	while (structure)
	{
		StructureType type = structure->GetStructureType();
		if (type == kStructureName)
		{
			if (clipName)
			{
				return (kDataExtraneousSubstructure);
			}

			clipName = static_cast<const NameStructure *>(structure)->GetName();
		}
		else if (type == kStructureParam)
		{
			const ParamStructure *paramStructure = static_cast<const ParamStructure *>(structure);
			if (paramStructure->GetAttribString() == "rate")
			{
				frameRate = paramStructure->GetParam();
			}
		}

		structure = structure->Next();
	}

	return (kDataOkay);
}


// Extension Structure
ExtensionStructure::ExtensionStructure() : OpenGexStructure(kStructureExtension)
{
}

ExtensionStructure::~ExtensionStructure()
{
}

bool ExtensionStructure::ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value)
{
	if (identifier == "applic")
	{
		*type = kDataString;
		*value = &applicationString;
		return (true);
	}

	if (identifier == "type")
	{
		*type = kDataString;
		*value = &typeString;
		return (true);
	}

	return (false);
}

bool ExtensionStructure::ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const
{
	return ((structure->GetBaseStructureType() == kStructurePrimitive) || (structure->GetStructureType() == kStructureExtension));
}


// Open Gex Data Description
OpenGexDataDescription::OpenGexDataDescription()
{
	distanceScale = 1.0F;
	angleScale = 1.0F;
	timeScale = 1.0F;
	upDirection = "z";
	forwardDirection = "x";
}

OpenGexDataDescription::~OpenGexDataDescription()
{
}

Structure *OpenGexDataDescription::CreateStructure(const String& identifier) const
{
	if (identifier == "Metric")
	{
		return (new MetricStructure);
	}

	if (identifier == "Name")
	{
		return (new NameStructure);
	}

	if (identifier == "ObjectRef")
	{
		return (new ObjectRefStructure);
	}

	if (identifier == "MaterialRef")
	{
		return (new MaterialRefStructure);
	}

	if (identifier == "Transform")
	{
		return (new TransformStructure);
	}

	if (identifier == "Translation")
	{
		return (new TranslationStructure);
	}

	if (identifier == "Rotation")
	{
		return (new RotationStructure);
	}

	if (identifier == "Scale")
	{
		return (new ScaleStructure);
	}

	if (identifier == "MorphWeight")
	{
		return (new MorphWeightStructure);
	}

	if (identifier == "Node")
	{
		return (new NodeStructure);
	}

	if (identifier == "BoneNode")
	{
		return (new BoneNodeStructure);
	}

	if (identifier == "GeometryNode")
	{
		return (new GeometryNodeStructure);
	}

	if (identifier == "LightNode")
	{
		return (new LightNodeStructure);
	}

	if (identifier == "CameraNode")
	{
		return (new CameraNodeStructure);
	}

	if (identifier == "VertexArray")
	{
		return (new VertexArrayStructure);
	}

	if (identifier == "IndexArray")
	{
		return (new IndexArrayStructure);
	}

	if (identifier == "BoneRefArray")
	{
		return (new BoneRefArrayStructure);
	}

	if (identifier == "BoneCountArray")
	{
		return (new BoneCountArrayStructure);
	}

	if (identifier == "BoneIndexArray")
	{
		return (new BoneIndexArrayStructure);
	}

	if (identifier == "BoneWeightArray")
	{
		return (new BoneWeightArrayStructure);
	}

	if (identifier == "Skeleton")
	{
		return (new SkeletonStructure);
	}

	if (identifier == "Skin")
	{
		return (new SkinStructure);
	}

	if (identifier == "Morph")
	{
		return (new MorphStructure);
	}

	if (identifier == "Mesh")
	{
		return (new MeshStructure);
	}

	if (identifier == "GeometryObject")
	{
		return (new GeometryObjectStructure);
	}

	if (identifier == "LightObject")
	{
		return (new LightObjectStructure);
	}

	if (identifier == "CameraObject")
	{
		return (new CameraObjectStructure);
	}

	if (identifier == "Param")
	{
		return (new ParamStructure);
	}

	if (identifier == "Color")
	{
		return (new ColorStructure);
	}

	if (identifier == "Texture")
	{
		return (new TextureStructure);
	}

	if (identifier == "Atten")
	{
		return (new AttenStructure);
	}

	if (identifier == "Material")
	{
		return (new MaterialStructure);
	}

	if (identifier == "Key")
	{
		return (new KeyStructure);
	}

	if (identifier == "Time")
	{
		return (new TimeStructure);
	}

	if (identifier == "Value")
	{
		return (new ValueStructure);
	}

	if (identifier == "Track")
	{
		return (new TrackStructure);
	}

	if (identifier == "Animation")
	{
		return (new AnimationStructure);
	}

	if (identifier == "Clip")
	{
		return (new ClipStructure);
	}

	if (identifier == "Extension")
	{
		return (new ExtensionStructure);
	}

	return (nullptr);
}

bool OpenGexDataDescription::ValidateTopLevelStructure(const Structure *structure) const
{
	StructureType type = structure->GetBaseStructureType();
	if ((type == kStructureNode) || (type == kStructureObject))
	{
		return (true);
	}

	type = structure->GetStructureType();
	return ((type == kStructureMetric) || (type == kStructureMaterial) || (type == kStructureClip) || (type == kStructureExtension));
}

DataResult OpenGexDataDescription::ProcessData(void)
{
	DataResult result = DataDescription::ProcessData();
	if (result == kDataOkay)
	{
		Structure *structure = GetRootStructure()->GetFirstSubnode();
		while (structure)
		{
			// Do something with the node here.

			structure = structure->Next();
		}
	}

	return (result);
}

