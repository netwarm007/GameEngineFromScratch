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


#ifndef OpenGEX_h
#define OpenGEX_h


#include "OpenDDL.h"


using namespace ODDL;


namespace OGEX
{
    enum
    {
        kStructureMetric                        = "mtrc"_i32,
        kStructureName                          = "name"_i32,
        kStructureObjectRef                     = "obrf"_i32,
        kStructureMaterialRef                   = "mtrf"_i32,
        kStructureMatrix                        = "mtrx"_i32,
        kStructureTransform                     = "xfrm"_i32,
        kStructureTranslation                   = "xslt"_i32,
        kStructureRotation                      = "rota"_i32,
        kStructureScale                         = "scal"_i32,
        kStructureMorphWeight                   = "mwgt"_i32,
        kStructureNode                          = "node"_i32,
        kStructureBoneNode                      = "bnnd"_i32,
        kStructureGeometryNode                  = "gmnd"_i32,
        kStructureLightNode                     = "ltnd"_i32,
        kStructureCameraNode                    = "cmnd"_i32,
        kStructureVertexArray                   = "vert"_i32,
        kStructureIndexArray                    = "indx"_i32,
        kStructureBoneRefArray                  = "bref"_i32,
        kStructureBoneCountArray                = "bcnt"_i32,
        kStructureBoneIndexArray                = "bidx"_i32,
        kStructureBoneWeightArray               = "bwgt"_i32,
        kStructureSkeleton                      = "skel"_i32,
        kStructureSkin                          = "skin"_i32,
        kStructureMorph                         = "mrph"_i32,
        kStructureMesh                          = "mesh"_i32,
        kStructureObject                        = "objc"_i32,
        kStructureGeometryObject                = "gmob"_i32,
        kStructureLightObject                   = "ltob"_i32,
        kStructureCameraObject                  = "cmob"_i32,
        kStructureAttrib                        = "attr"_i32,
        kStructureParam                         = "parm"_i32,
        kStructureColor                         = "colr"_i32,
        kStructureTexture                       = "txtr"_i32,
        kStructureAtten                         = "attn"_i32,
        kStructureMaterial                      = "matl"_i32,
        kStructureKey                           = "key "_i32,
        kStructureCurve                         = "curv"_i32,
        kStructureTime                          = "time"_i32,
        kStructureValue                         = "valu"_i32,
        kStructureTrack                         = "trac"_i32,
        kStructureAnimation                     = "anim"_i32,
        kStructureClip                          = "clip"_i32,
        kStructureExtension                     = "extn"_i32
    };


    enum
    {
        kDataOpenGexInvalidUpDirection          = "ivud"_i32,
        kDataOpenGexInvalidForwardDirection     = "ivfd"_i32,
        kDataOpenGexInvalidTranslationKind      = "ivtk"_i32,
        kDataOpenGexInvalidRotationKind         = "ivrk"_i32,
        kDataOpenGexInvalidScaleKind            = "ivsk"_i32,
        kDataOpenGexDuplicateLod                = "dlod"_i32,
        kDataOpenGexMissingLodSkin              = "mlsk"_i32,
        kDataOpenGexMissingLodMorph             = "mlmp"_i32,
        kDataOpenGexDuplicateMorph              = "dmph"_i32,
        kDataOpenGexUndefinedLightType          = "ivlt"_i32,
        kDataOpenGexUndefinedCurve              = "udcv"_i32,
        kDataOpenGexUndefinedAtten              = "udan"_i32,
        kDataOpenGexDuplicateVertexArray        = "dpva"_i32,
        kDataOpenGexPositionArrayRequired       = "parq"_i32,
        kDataOpenGexVertexCountUnsupported      = "vcus"_i32,
        kDataOpenGexIndexValueUnsupported       = "ivus"_i32,
        kDataOpenGexIndexArrayRequired          = "iarq"_i32,
        kDataOpenGexVertexCountMismatch         = "vcmm"_i32,
        kDataOpenGexBoneCountMismatch           = "bcmm"_i32,
        kDataOpenGexBoneWeightCountMismatch     = "bwcm"_i32,
        kDataOpenGexInvalidBoneRef              = "ivbr"_i32,
        kDataOpenGexInvalidObjectRef            = "ivor"_i32,
        kDataOpenGexInvalidMaterialRef          = "ivmr"_i32,
        kDataOpenGexMaterialIndexUnsupported    = "mius"_i32,
        kDataOpenGexDuplicateMaterialRef        = "dprf"_i32,
        kDataOpenGexMissingMaterialRef          = "msrf"_i32,
        kDataOpenGexTargetRefNotLocal           = "trnl"_i32,
        kDataOpenGexInvalidTargetStruct         = "ivts"_i32,
        kDataOpenGexInvalidKeyKind              = "ivkk"_i32,
        kDataOpenGexInvalidCurveType            = "ivct"_i32,
        kDataOpenGexKeyCountMismatch            = "kycm"_i32,
        kDataOpenGexEmptyKeyStructure           = "emky"_i32
    };


    class MaterialStructure;
    class ObjectStructure;
    class GeometryObjectStructure;
    class LightObjectStructure;
    class CameraObjectStructure;
    class OpenGexDataDescription;


    class OpenGexStructure : public Structure
    {
        protected:

            OpenGexStructure(StructureType type);

        public:

            ~OpenGexStructure();

            Structure *GetFirstCoreSubnode(void) const;
            Structure *GetLastCoreSubnode(void) const;
            Structure *GetFirstExtensionSubnode(void) const;

            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
    };


    class MetricStructure : public OpenGexStructure
    {
        private:

            String      metricKey;

        public:

            MetricStructure();
            ~MetricStructure();

            const String& GetMetricKey(void) const
            {
                return (metricKey);
            }

            bool ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value);
            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class NameStructure : public OpenGexStructure
    {
        private:

            const char      *name;

        public:

            NameStructure();
            ~NameStructure();

            const char *GetName(void) const
            {
                return (name);
            }

            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class ObjectRefStructure : public OpenGexStructure
    {
        private:

            Structure       *targetStructure;

        public:

            ObjectRefStructure();
            ~ObjectRefStructure();

            Structure *GetTargetStructure(void) const
            {
                return (targetStructure);
            }

            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class MaterialRefStructure : public OpenGexStructure
    {
        private:

            unsigned_int32              materialIndex;
            const MaterialStructure     *targetStructure;

        public:

            MaterialRefStructure();
            ~MaterialRefStructure();

            unsigned_int32 GetMaterialIndex(void) const
            {
                return (materialIndex);
            }

            const MaterialStructure *GetTargetStructure(void) const
            {
                return (targetStructure);
            }

            bool ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value);
            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class AnimatableStructure : public OpenGexStructure
    {
        protected:

            AnimatableStructure(StructureType type);
            ~AnimatableStructure();
    };


    class MatrixStructure : public AnimatableStructure
    {
        private:

            bool        objectFlag;

        protected:

            MatrixStructure(StructureType type);

        public:

            ~MatrixStructure();

            bool GetObjectFlag(void) const
            {
                return (objectFlag);
            }

            bool ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value);
    };


    class TransformStructure : public MatrixStructure
    {
        private:

            int32           transformCount;
            const float     *transformArray;

        public:

            TransformStructure();
            ~TransformStructure();

            int32 GetTransformCount(void) const
            {
                return (transformCount);
            }

            const float *GetTransform(int32 index = 0) const
            {
                return (&transformArray[index * 16]);
            }

            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class TranslationStructure : public MatrixStructure
    {
        private:

            String      translationKind;
            const float* translationArray;     

        public:

            TranslationStructure();
            ~TranslationStructure();

            const String& GetTranslationKind(void) const
            {
                return (translationKind);
            }

            const float* GetTranslation(void) const
            {
                return (translationArray);
            }

            bool ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value);
            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class RotationStructure : public MatrixStructure
    {
        private:

            String      rotationKind;
            const float* rotationArray;     

        public:

            RotationStructure();
            ~RotationStructure();

            const String& GetRotationKind(void) const
            {
                return (rotationKind);
            }

            const float* GetRotation(void) const
            {
                return (rotationArray);
            }

            bool ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value);
            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class ScaleStructure : public MatrixStructure
    {
        private:

            String      scaleKind;
            const float* scaleArray;     

        public:

            ScaleStructure();
            ~ScaleStructure();

            const String& GetScaleKind(void) const
            {
                return (scaleKind);
            }

            const float* GetScale(void) const
            {
                return (scaleArray);
            }

            bool ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value);
            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class MorphWeightStructure : public AnimatableStructure
    {
        private:

            unsigned_int32      morphIndex;
            float               morphWeight;

        public:

            MorphWeightStructure();
            ~MorphWeightStructure();

            unsigned_int32 GetMorphIndex(void) const
            {
                return (morphIndex);
            }

            float GetMorphWeight(void) const
            {
                return (morphWeight);
            }

            bool ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value);
            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class NodeStructure : public OpenGexStructure
    {
        private:

            const char      *nodeName;

            void CalculateTransforms(const OpenGexDataDescription *dataDescription);

        protected:

            NodeStructure(StructureType type);

        public:

            NodeStructure();
            ~NodeStructure();

            const char *GetNodeName(void) const
            {
                return (nodeName);
            }

            virtual const ObjectStructure *GetObjectStructure(void) const;

            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class BoneNodeStructure : public NodeStructure
    {
        public:

            BoneNodeStructure();
            ~BoneNodeStructure();
    };


    class GeometryNodeStructure : public NodeStructure
    {
        private:

            bool        visibleFlag[2];
            bool        shadowFlag[2];
            bool        motionBlurFlag[2];

            GeometryObjectStructure                 *geometryObjectStructure;
            Array<const MaterialStructure *, 4>     materialStructureArray;

        public:

            GeometryNodeStructure();
            ~GeometryNodeStructure();

            const ObjectStructure *GetObjectStructure(void) const;
            auto GetMaterialStructureArray(void) const -> decltype(materialStructureArray);
			bool GetVisibleFlag(void) const;
			bool GetShadowFlag(void) const;
			bool GetMotionBlurFlag(void) const;

            bool ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value);
            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class LightNodeStructure : public NodeStructure
    {
        private:

            bool        shadowFlag[2];

            const LightObjectStructure      *lightObjectStructure;

        public:

            LightNodeStructure();
            ~LightNodeStructure();

            const ObjectStructure *GetObjectStructure(void) const;
            bool GetShadowFlag(void) const;

            bool ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value);
            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class CameraNodeStructure : public NodeStructure
    {
        private:

            const CameraObjectStructure     *cameraObjectStructure;

        public:

            CameraNodeStructure();
            ~CameraNodeStructure();

            const ObjectStructure *GetObjectStructure(void) const;

            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class VertexArrayStructure : public OpenGexStructure
    {
        private:

            String              arrayAttrib;
            unsigned_int32      morphIndex;

        public:

            VertexArrayStructure();
            ~VertexArrayStructure();

            const String& GetArrayAttrib(void) const
            {
                return (arrayAttrib);
            }

            unsigned_int32 GetMorphIndex(void) const
            {
                return (morphIndex);
            }

            bool ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value);
            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class IndexArrayStructure : public OpenGexStructure
    {
        private:

            unsigned_int32          materialIndex;
            unsigned_int64          restartIndex;
            String                  frontFace;

        public:

            IndexArrayStructure();
            ~IndexArrayStructure();

            unsigned_int32 GetMaterialIndex(void) const
            {
                return (materialIndex);
            }

            unsigned_int64 GetRestartIndex(void) const
            {
                return (restartIndex);
            }

            const String& GetFrontFace(void) const
            {
                return (frontFace);
            }

            bool ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value);
            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class BoneRefArrayStructure : public OpenGexStructure
    {
        private:

            int32                       boneCount;
            const BoneNodeStructure     **boneNodeArray;

        public:

            BoneRefArrayStructure();
            ~BoneRefArrayStructure();

            int32 GetBoneCount(void) const
            {
                return (boneCount);
            }

            const BoneNodeStructure *const *GetBoneNodeArray(void) const
            {
                return (boneNodeArray);
            }

            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class BoneCountArrayStructure : public OpenGexStructure
    {
        private:

            int32                   vertexCount;
            const unsigned_int16    *boneCountArray;
            unsigned_int16          *arrayStorage;

        public:

            BoneCountArrayStructure();
            ~BoneCountArrayStructure();

            int32 GetVertexCount(void) const
            {
                return (vertexCount);
            }

            const unsigned_int16 *GetBoneCountArray(void) const
            {
                return (boneCountArray);
            }

            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class BoneIndexArrayStructure : public OpenGexStructure
    {
        private:

            int32                   boneIndexCount;
            const unsigned_int16    *boneIndexArray;
            unsigned_int16          *arrayStorage;

        public:

            BoneIndexArrayStructure();
            ~BoneIndexArrayStructure();

            int32 GetBoneIndexCount(void) const
            {
                return (boneIndexCount);
            }

            const unsigned_int16 *GetBoneIndexArray(void) const
            {
                return (boneIndexArray);
            }

            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class BoneWeightArrayStructure : public OpenGexStructure
    {
        private:

            int32           boneWeightCount;
            const float     *boneWeightArray;

        public:

            BoneWeightArrayStructure();
            ~BoneWeightArrayStructure();

            int32 GetBoneWeightCount(void) const
            {
                return (boneWeightCount);
            }

            const float *GetBoneWeightArray(void) const
            {
                return (boneWeightArray);
            }

            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class SkeletonStructure : public OpenGexStructure
    {
        private:

            const BoneRefArrayStructure     *boneRefArrayStructure;
            const TransformStructure        *transformStructure;

        public:

            SkeletonStructure();
            ~SkeletonStructure();

            const BoneRefArrayStructure *GetBoneRefArrayStructure(void) const
            {
                return (boneRefArrayStructure);
            }

            const TransformStructure *GetTransformStructure(void) const
            {
                return (transformStructure);
            }

            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class SkinStructure : public OpenGexStructure
    {
        private:

            const SkeletonStructure             *skeletonStructure;
            const BoneCountArrayStructure       *boneCountArrayStructure;
            const BoneIndexArrayStructure       *boneIndexArrayStructure;
            const BoneWeightArrayStructure      *boneWeightArrayStructure;

        public:

            SkinStructure();
            ~SkinStructure();

            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class MorphStructure : public OpenGexStructure, public MapElement<MorphStructure>
    {
        private:

            unsigned_int32      morphIndex;

            bool                baseFlag;
            unsigned_int32      baseIndex;

            const char          *morphName;

        public:

            typedef unsigned_int32 KeyType;

            MorphStructure();
            ~MorphStructure();

            using MapElement<MorphStructure>::Previous;
            using MapElement<MorphStructure>::Next;

            KeyType GetKey(void) const
            {
                return (morphIndex);
            }

            unsigned_int32 GetMorphIndex(void) const
            {
                return (morphIndex);
            }

            bool GetBaseFlag(void) const
            {
                return (baseFlag);
            }

            unsigned_int32 GetBaseIndex(void) const
            {
                return (baseIndex);
            }

            const char *GetMorphName(void) const
            {
                return (morphName);
            }

            bool ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value);
            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class MeshStructure : public OpenGexStructure, public MapElement<MeshStructure>
    {
        private:

            unsigned_int32          meshLevel;
            String                  meshPrimitive;

            SkinStructure           *skinStructure;

        public:

            typedef unsigned_int32 KeyType;

            MeshStructure();
            ~MeshStructure();

            using MapElement<MeshStructure>::Previous;
            using MapElement<MeshStructure>::Next;

            KeyType GetKey(void) const
            {
                return (meshLevel);
            }

            unsigned_int32 GetMeshLevel(void) const
            {
                return (meshLevel);
            }

            const String& GetMeshPrimitive(void) const
            {
                return (meshPrimitive);
            }

            SkinStructure *GetSkinStructure(void) const
            {
                return (skinStructure);
            }

            bool ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value);
            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class ObjectStructure : public OpenGexStructure
    {
        protected:

            ObjectStructure(StructureType type);

        public:

            ~ObjectStructure();
    };


    class GeometryObjectStructure : public ObjectStructure
    {
        private:

            bool                    visibleFlag;
            bool                    shadowFlag;
            bool                    motionBlurFlag;

            Map<MeshStructure>      meshMap;
            Map<MorphStructure>     morphMap;

        public:

            GeometryObjectStructure();
            ~GeometryObjectStructure();

            bool GetVisibleFlag(void) const
            {
                return (visibleFlag);
            }

            bool GetShadowFlag(void) const
            {
                return (shadowFlag);
            }

            bool GetMotionBlurFlag(void) const
            {
                return (motionBlurFlag);
            }

            const Map<MeshStructure> *GetMeshMap(void) const
            {
                return (&meshMap);
            }

            const Map<MorphStructure> *GetMorphMap(void) const
            {
                return (&morphMap);
            }

            bool ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value);
            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class LightObjectStructure : public ObjectStructure
    {
        private:

            String      typeString;
            bool        shadowFlag;

        public:

            LightObjectStructure();
            ~LightObjectStructure();

            const String& GetTypeString(void) const
            {
                return (typeString);
            }

            bool GetShadowFlag(void) const
            {
                return (shadowFlag);
            }

            bool ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value);
            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class CameraObjectStructure : public ObjectStructure
    {
        private:

            float       focalLength;
            float       nearDepth;
            float       farDepth;

        public:

            CameraObjectStructure();
            ~CameraObjectStructure();

            float GetFocalLength(void) const
            {
                return (focalLength);
            }

            float GetNearDepth(void) const
            {
                return (nearDepth);
            }

            float GetFarDepth(void) const
            {
                return (farDepth);
            }

            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class AttribStructure : public OpenGexStructure
    {
        private:

            String      attribString;

        protected:

            AttribStructure(StructureType type);

        public:

            ~AttribStructure();

            const String& GetAttribString(void) const
            {
                return (attribString);
            }

            bool ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value);
    };


    class ParamStructure : public AttribStructure
    {
        private:

            float       param;

        public:

            ParamStructure();
            ~ParamStructure();

            float GetParam(void) const
            {
                return (param);
            }

            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class ColorStructure : public AttribStructure
    {
        private:

            float       color[4];

        public:

            ColorStructure();
            ~ColorStructure();

            const float *GetColor(void) const
            {
                return (color);
            }

            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class TextureStructure : public AttribStructure
    {
        private:

            String              textureName;
            unsigned_int32      texcoordIndex;

        public:

            TextureStructure();
            ~TextureStructure();

            const String& GetTextureName(void) const
            {
                return (textureName);
            }

            unsigned_int32 GetTexcoordIndex(void) const
            {
                return (texcoordIndex);
            }

            bool ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value);
            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class AttenStructure : public OpenGexStructure
    {
        private:

            String          attenKind;
            String          curveType;

            float           beginParam;
            float           endParam;

            float           scaleParam;
            float           offsetParam;

            float           constantParam;
            float           linearParam;
            float           quadraticParam;

            float           powerParam;

        public:

            AttenStructure();
            ~AttenStructure();

            const String& GetAttenKind(void) const
            {
                return (attenKind);
            }

            const String& GetCurveType(void) const
            {
                return (curveType);
            }

            float GetBeginParam(void) const
            {
                return (beginParam);
            }

            float GetEndParam(void) const
            {
                return (endParam);
            }

            float GetScaleParam(void) const
            {
                return (scaleParam);
            }

            float GetOffsetParam(void) const
            {
                return (offsetParam);
            }

            float GetConstantParam(void) const
            {
                return (constantParam);
            }

            float GetLinearParam(void) const
            {
                return (linearParam);
            }

            float GetQuadraticParam(void) const
            {
                return (quadraticParam);
            }

            float GetPowerParam(void) const
            {
                return (powerParam);
            }

            bool ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value);
            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class MaterialStructure : public OpenGexStructure
    {
        private:

            bool            twoSidedFlag;
            const char      *materialName;

        public:

            MaterialStructure();
            ~MaterialStructure();

            bool GetTwoSidedFlag(void) const
            {
                return (twoSidedFlag);
            }

            const char *GetMaterialName(void) const
            {
                return (materialName);
            }

            bool ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value);
            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class KeyStructure : public OpenGexStructure
    {
        private:

            String          keyKind;
            bool            scalarFlag;

        public:

            KeyStructure();
            ~KeyStructure();

            const String& GetKeyKind(void) const
            {
                return (keyKind);
            }

            bool GetScalarFlag(void) const
            {
                return (scalarFlag);
            }

            bool ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value);
            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class CurveStructure : public OpenGexStructure
    {
        private:

            String                  curveType;

            const KeyStructure      *keyValueStructure;
            const KeyStructure      *keyControlStructure[2];
            const KeyStructure      *keyTensionStructure;
            const KeyStructure      *keyContinuityStructure;
            const KeyStructure      *keyBiasStructure;

        protected:

            int32                   keyDataElementCount;

            CurveStructure(StructureType type);

        public:

            ~CurveStructure();

            const String& GetCurveType(void) const
            {
                return (curveType);
            }

            const KeyStructure *GetKeyValueStructure(void) const
            {
                return (keyValueStructure);
            }

            const KeyStructure *GetKeyControlStructure(int32 index) const
            {
                return (keyControlStructure[index]);
            }

            const KeyStructure *GetKeyTensionStructure(void) const
            {
                return (keyTensionStructure);
            }

            const KeyStructure *GetKeyContinuityStructure(void) const
            {
                return (keyContinuityStructure);
            }

            const KeyStructure *GetKeyBiasStructure(void) const
            {
                return (keyBiasStructure);
            }

            int32 GetKeyDataElementCount(void) const
            {
                return (keyDataElementCount);
            }

            bool ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value);
            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class TimeStructure : public CurveStructure
    {
        public:

            TimeStructure();
            ~TimeStructure();

            DataResult ProcessData(DataDescription *dataDescription);
    };


    class ValueStructure : public CurveStructure
    {
        public:

            ValueStructure();
            ~ValueStructure();

            DataResult ProcessData(DataDescription *dataDescription);
    };


    class TrackStructure : public OpenGexStructure
    {
        private:

            StructureRef            targetRef;

            AnimatableStructure     *targetStructure;
            const TimeStructure     *timeStructure;
            const ValueStructure    *valueStructure;

        public:

            TrackStructure();
            ~TrackStructure();

            const StructureRef& GetTargetRef(void) const
            {
                return (targetRef);
            }

            AnimatableStructure *GetTargetStructure(void) const
            {
                return (targetStructure);
            }

            const TimeStructure *GetTimeStructure(void) const
            {
                return (timeStructure);
            }

            const ValueStructure *GetValueStructure(void) const
            {
                return (valueStructure);
            }

            bool ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value);
            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class AnimationStructure : public OpenGexStructure
    {
        private:

            int32       clipIndex;

            bool        beginFlag;
            bool        endFlag;
            float       beginTime;
            float       endTime;

        public:

            AnimationStructure();
            ~AnimationStructure();

            int32 GetClipIndex(void) const
            {
                return (clipIndex);
            }

            bool ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value);
            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class ClipStructure : public OpenGexStructure
    {
        private:

            unsigned_int32      clipIndex;

            float               frameRate;
            const char          *clipName;

        public:

            ClipStructure();
            ~ClipStructure();

            unsigned_int32 GetClipIndex(void) const
            {
                return (clipIndex);
            }

            float GetFrameRate(void) const
            {
                return (frameRate);
            }

            const char *GetClipName(void) const
            {
                return (clipName);
            }

            bool ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value);
            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
            DataResult ProcessData(DataDescription *dataDescription);
    };


    class ExtensionStructure : public OpenGexStructure
    {
        private:

            String      applicationString;
            String      typeString;

        public:

            ExtensionStructure();
            ~ExtensionStructure();

            const String& GetApplicationString(void) const
            {
                return (applicationString);
            }

            const String& GetTypeString(void) const
            {
                return (typeString);
            }

            bool ValidateProperty(const DataDescription *dataDescription, const String& identifier, DataType *type, void **value);
            bool ValidateSubstructure(const DataDescription *dataDescription, const Structure *structure) const;
    };


    class OpenGexDataDescription : public DataDescription
    {
        private:

            float           distanceScale;
            float           angleScale;
            float           timeScale;
            ODDL::String    upDirection;
            ODDL::String    forwardDirection;

            DataResult ProcessData(void);

        public:

            OpenGexDataDescription();
            ~OpenGexDataDescription();

            float GetDistanceScale(void) const
            {
                return (distanceScale);
            }

            void SetDistanceScale(float scale)
            {
                distanceScale = scale;
            }

            float GetAngleScale(void) const
            {
                return (angleScale);
            }

            void SetAngleScale(float scale)
            {
                angleScale = scale;
            }

            float GetTimeScale(void) const
            {
                return (timeScale);
            }

            void SetTimeScale(float scale)
            {
                timeScale = scale;
            }

            const ODDL::String& GetUpDirection(void) const
            {
                return (upDirection);
            }

            void SetUpDirection(const char *direction)
            {
                upDirection = direction;
            }

            const ODDL::String& GetForwardDirection(void) const
            {
                return (forwardDirection);
            }

            void SetForwardDirection(const char *direction)
            {
                forwardDirection = direction;
            }

            Structure *CreateStructure(const String& identifier) const;
            bool ValidateTopLevelStructure(const Structure *structure) const;
    };
}


#endif
