/* Author: Morten S. Mikkelsen
 * Freely available for any type of use.
 */

#ifndef __ILLUM_H__
#define __ILLUM_H__

#ifndef M_PI
	#define M_PI 3.1415926535897932384626433832795
#endif

float3 FixNormal(float3 vN, float3 vV);
float toBeckmannParam(const float n);
float toNPhong(const float m);


// Schlick's Fresnel approximation
float fresnelReflectance( float VdotH, float F0 )
{
	float base = 1-VdotH;
	float exponential = pow(base, 5.0);
	return saturate(exponential + F0 * (1 - exponential));
}

#define FLT_EPSILON     1.192092896e-07F        // smallest such that 1.0+FLT_EPSILON != 1.0
#define FLT_MAX         3.402823466e+38F        // max value
#define FLT_MIN         1.175494351e-38F        // min positive value



// The Torrance-Sparrow visibility factor, G,
// as described by Jim Blinn but divided by VdotN
float VisibDiv(float LdotN, float VdotN, float VdotH, float HdotN)
{	
	// VdotH should never be zero. Only possible if
	// L and V end up in the same plane (unlikely).
	const float denom = max( VdotH, FLT_EPSILON );	
										
	float numL = min(VdotN, LdotN);
	const float numR = 2*HdotN;
	if((numL*numR)<=denom)	// min(x,1) = x
	{
		numL = numL == VdotN ? 1.0 : (LdotN / VdotN);	// VdotN is > 0 if this division is used
		return (numL*numR) / denom;
	}
	else					// min(x,1) = 1				this branch is taken when H and N are "close" (see fig. 3)
		return 1.0 / VdotN;
		// VdotN >= HdotN*VdotN >= HdotN*min(VdotN, LdotN) >= FLT_EPSILON/2
}


// this is a normalized Phong model used in the Torrance-Sparrow model
float3 BRDF_ts_nphong(float3 vN, float3 vL, float3 vV, float3 Cd, float3 Cs, float n=32, float F0=0.2)
{
	// reflect hack when view vector is occluded
	// (not needed).
	//vN = FixNormal(vN, vV);

	// microfacet normal
	float3 vH = normalize(vV+vL);

	// the various dot products
	const float LdotN = saturate(dot(vL, vN));
	const float VdotN = saturate(dot(vV, vN));
	const float VdotH = saturate(dot(vV, vH));
	const float HdotN = saturate(dot(vH, vN));
	
	// diffuse
	float fDiff = LdotN;

	// D is a surface distribution function and obeys:
	// D(vH)*HdotN is normalized (over half-spere)
	// Specifically, this is the normalized phong model
	const float D = ((n+2)/(2*M_PI))*pow(HdotN, n);

	// torrance-sparrow visibility term divided by VdotN
	const float fVdivDots = VisibDiv(LdotN, VdotN, VdotH, HdotN);
	
	// Schlick's approximation
	const float fFres = fresnelReflectance(VdotH, F0);

	// torrance-sparrow:
	// (F * G * D) / (4 * LdotN * VdotN)
	// Division by VdotN is done in VisibDiv()
	// and division by LdotN is removed since 
	// outgoing radiance is determined by:
	// BRDF * LdotN * L()
	float fSpec = (fFres * fVdivDots * D) / 4;
	
	// sum up: diff + spec
	// technically fDiff should be divided by pi.
	// Instead, we choose here to scale Cs by pi
	// which makes the final result scaled by pi.
	// We do this to keep the output intensity range
	// at a level which is more "familiar".
	float3 res = Cd * fDiff + M_PI * Cs * fSpec;
	return res;
}

float3 BRDF2_ts_nphong(float3 vN, float3 vN2, float3 vL, float3 vV, float3 Cd, float3 Cs, float n=32, float F0=0.2)
{
	float3 res = BRDF_ts_nphong(vN, vL, vV, Cd, Cs, n, F0);
	return saturate(4*dot(vL, vN2)) * res;
}

// this is the Torrance-Sparrow model but using the Beckmann distribution
float3 BRDF_ts_beckmann(float3 vN, float3 vL, float3 vV, float3 Cd, float3 Cs, float m=0.22, float F0=0.2)
{
	// reflect hack when view vector is occluded
	// (not needed).
	//vN = FixNormal(vN, vV);

	// microfacet normal
	float3 vH = normalize(vV+vL);

	// the various dot products
	const float LdotN = saturate(dot(vL, vN));
	const float VdotN = saturate(dot(vV, vN));
	const float VdotH = saturate(dot(vV, vH));
	const float HdotN = saturate(dot(vH, vN));
	
	// diffuse
	float fDiff = LdotN;
	
	// D is a surface distribution function and obeys:
	// D(vH)*HdotN is normalized (over half-spere)
	// Specifically, this is the Beckmann surface distribution function
	// D = exp(-tan^2(\theta_h)/m^2) / (pi * m^2 * cos^4(\theta_h));
	// where \theta_h = acos(HdotN)
	const float fSqCSnh = HdotN*HdotN;
	const float fSqCSnh_m2 = fSqCSnh*m*m;
	//const float numerator = exp(-pow(tan(acos(HdotN))/m,2));
	const float numerator = exp(-((1-fSqCSnh)/max(fSqCSnh_m2, FLT_EPSILON)));		// faster than tan+acos
	const float D = numerator / (M_PI*max(fSqCSnh_m2*fSqCSnh, FLT_EPSILON));

	// torrance-sparrow visibility term divided by VdotN
	const float fVdivDots = VisibDiv(LdotN, VdotN, VdotH, HdotN);
	
	// Schlick's approximation
	const float fFres = fresnelReflectance(VdotH, F0);

	// torrance-sparrow:
	// (F * G * D) / (4 * LdotN * VdotN)
	// Division by VdotN is done in VisibDiv()
	// and division by LdotN is removed since 
	// outgoing radiance is determined by:
	// BRDF * LdotN * L()
	float fSpec = (fFres * fVdivDots * D) / 4;
	
	// sum up: diff + spec
	// technically fDiff should be divided by pi.
	// Instead, we choose here to scale Cs by pi
	// which makes the final result scaled by pi.
	// We do this to keep the output intensity range
	// at a level which is more "familiar".
	float3 res = Cd * fDiff + M_PI * Cs * fSpec;
	return res;
}

float3 BRDF2_ts_beckmann(float3 vN, float3 vN2, float3 vL, float3 vV, float3 Cd, float3 Cs, float m=0.22, float F0=0.2)
{
	float3 res = BRDF_ts_beckmann(vN, vL, vV, Cd, Cs, m, F0);
	return saturate(4*dot(vL, vN2)) * res;
}


float3 FixNormal(float3 vN, float3 vV)
{
	const float VdotN = dot(vV,vN);
	if(VdotN<=0)
	{
		vN = vN - 2*vV * VdotN;
	}
	return vN;
}

float toBeckmannParam(const float n)
{
	// remap to beckmann roughness parameter by matching
	// the normalization constants in the surface distribution functions.
	float m = sqrt(2 / (n+2));
	return m;
}

float toNPhong(const float m)
{
	// remap to normalized phong roughness parameter by matching
	// the normalization constants in the surface distribution functions.
	float n = (2 / (m*m)) - 2;
	return n;
}







// disabled fresnel on the normalized Phong model used in the Torrance-Sparrow model
float3 BRDF_ts_nphong_nofr(float3 vN, float3 vL, float3 vV, float3 Cd, float3 Cs, float n=32)
{
	// microfacet normal
	float3 vH = normalize(vV+vL);

	// the various dot products
	const float LdotN = saturate(dot(vL, vN));
	const float VdotN = saturate(dot(vV, vN));
	const float VdotH = saturate(dot(vV, vH));
	const float HdotN = saturate(dot(vH, vN));
	
	// diffuse
	float fDiff = LdotN;

	const float D = ((n+2)/(2*M_PI))*pow(HdotN, n);
	const float fVdivDots = VisibDiv(LdotN, VdotN, VdotH, HdotN);
	float fSpec = (fVdivDots * D) / 4;
	
	float3 res = Cd * fDiff + M_PI * Cs * fSpec;
	return res;
}

float3 BRDF2_ts_nphong_nofr(float3 vN, float3 vN2, float3 vL, float3 vV, float3 Cd, float3 Cs, float n=32)
{
	float3 res = BRDF_ts_nphong_nofr(vN, vL, vV, Cd, Cs, n);
	return saturate(4*dot(vL, vN2)) * res;
}




// traditional Blinn-Phong model
float3 BRDF(float3 vN, float3 vL, float3 vV, float3 Cd, float3 Cs, float rough=32)
{
	float3 myhalf = normalize(vV+vL);
	
	float fDiff = saturate(dot(vL, vN));
	float fSpec = pow(saturate(dot(myhalf, vN)), rough);
	
	float3 res = Cd * fDiff + Cs * fSpec;	
	return res;
}

float3 BRDF2(float3 vN, float3 vN2, float3 vL, float3 vV, float3 Cd, float3 Cs, float rough=32)
{
	float3 res = BRDF(vN, vL, vV, Cd, Cs, rough);
	return saturate(4*dot(vL, vN2)) * res;
}

float3 BRDF2_subsurf(float3 vN, float3 vN2, float3 vL, float3 vV, float3 Cd, float3 Cs, float rough=32, float RollOff = 0.2, float3 SubColor = 0.6*float3(1.0f, 0.2f, 0.2f))
{
	float3 myhalf = normalize(vV+vL);
	
	
	float ldn = dot(vL, vN);
	float fDiff = saturate(ldn);
	float fSpec = pow(saturate(dot(myhalf, vN)), rough);
	
	
	float subLamb = smoothstep(-RollOff,1.0,ldn) - smoothstep(0.0,1.0,ldn);
    subLamb = max(0.0,subLamb);
    float3 subContrib = subLamb * SubColor;
	
	float3 res = saturate(4*dot(vL, vN2)) * (subContrib + Cd * fDiff + Cs * fSpec);	
	return res;
}


#endif