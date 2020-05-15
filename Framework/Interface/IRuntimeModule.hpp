#pragma once

#include "Interface.hpp"
#include <config.h>

namespace My {
	Interface IRuntimeModule{
public:
	virtual ~IRuntimeModule() = default;

	virtual int Initialize() = 0;
	virtual void Finalize() = 0;

	virtual void Tick() = 0;

#ifdef DEBUG
	virtual void DrawDebugInfo() {};
#endif
	};

}

