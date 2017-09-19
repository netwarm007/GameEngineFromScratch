#pragma once
#include <stdint.h>

#if __cplusplus >= 201103L && !defined(__ORBIS__)
#define ENUM(e) enum struct e : uint32_t
#else
#define ENUM(e) enum e
#endif

