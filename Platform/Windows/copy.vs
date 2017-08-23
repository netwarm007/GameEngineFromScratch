#include "cbuffer.h"
#include "vsoutput.hs"

v2p main(a2v input) {
	v2p output;
	output.position = float4(input.position, 1.0);
	output.color = input.color;

	return output;
}

