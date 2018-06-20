#pragma once
#include <functional>
#include "BaseSceneObject.hpp"
#include "SceneObjectTypeDef.hpp"

namespace My {
    ENUM(AttenCurveType) {
        kNone = 0,
        kLinear = 1,
        kSmooth = 2,
        kInverse = 3,
        kInverseSquare = 4
    };

    struct AttenCurve {
        AttenCurveType type;
        union AttenCurveParams {
            struct LinearParam { float begin_atten; float end_atten; } linear_params;
            struct SmoothParam { float begin_atten; float end_atten; } smooth_params;
            struct InverseParam { float scale; float offset; float kl; float kc; } inverse_params;
            struct InverseSquareParam { float scale; float offset; float kq; float kl; float kc; } inverse_squre_params;
        } u;

        AttenCurve() : type(AttenCurveType::kNone)
                       {}
    };
    
    class SceneObjectLight : public BaseSceneObject
    {
        protected:
            Color       m_LightColor;
            float       m_fIntensity;
            AttenCurve  m_LightDistanceAttenuation;
            bool        m_bCastShadows;
            std::string m_strTexture;

        public:
			void SetIfCastShadow(bool shadow) { m_bCastShadows = shadow; };

            void SetColor(std::string& attrib, Vector4f& color) 
            { 
                if(attrib == "light") {
                    m_LightColor = Color(color); 
                }
            };

            void SetParam(std::string& attrib, float param) 
            { 
                if(attrib == "intensity") {
                    m_fIntensity = param; 
                }
            };

            void SetTexture(std::string& attrib, std::string& textureName) 
            { 
                if(attrib == "projection") {
                    m_strTexture = textureName;
                }
            };

            void SetDistanceAttenuation(AttenCurve curve)
            {
                m_LightDistanceAttenuation = curve;
            }

            const AttenCurve& GetDistanceAttenuation(void)
            {
                return m_LightDistanceAttenuation;
            }

            const Color& GetColor() { return m_LightColor; };
            float GetIntensity() { return m_fIntensity; };

        protected:
            // can only be used as base class of delivered lighting objects
            SceneObjectLight(const SceneObjectType type) : BaseSceneObject(type), 
                m_LightColor(Vector4f(1.0f)), m_fIntensity(1.0f), 
                m_bCastShadows(false) {};

        friend std::ostream& operator<<(std::ostream& out, const SceneObjectLight& obj);
    };

    class SceneObjectOmniLight : public SceneObjectLight
    {
        public:
            SceneObjectOmniLight(void) : SceneObjectLight(SceneObjectType::kSceneObjectTypeLightOmni) {}

        friend std::ostream& operator<<(std::ostream& out, const SceneObjectOmniLight& obj);
    };

    class SceneObjectSpotLight : public SceneObjectLight
    {
        protected:
            float   m_fConeBeginAngle;
            float   m_fConeEndAngle;
            AttenCurve  m_LightAngleAttenuation;

        public:
            SceneObjectSpotLight(void) : SceneObjectLight(SceneObjectType::kSceneObjectTypeLightSpot)
                            {};

            void SetAngleAttenuation(AttenCurve curve)
            {
                m_LightAngleAttenuation = curve;
            }

            const AttenCurve& GetAngleAttenuation(void)
            {
                return m_LightAngleAttenuation;
            }

        friend std::ostream& operator<<(std::ostream& out, const SceneObjectSpotLight& obj);
    };

    class SceneObjectInfiniteLight : public SceneObjectLight
    {
        public:
            SceneObjectInfiniteLight(void) : SceneObjectLight(SceneObjectType::kSceneObjectTypeLightInfi) {}

        friend std::ostream& operator<<(std::ostream& out, const SceneObjectInfiniteLight& obj);
    };

}