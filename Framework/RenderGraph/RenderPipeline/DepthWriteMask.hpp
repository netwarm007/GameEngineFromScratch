#pragma once
namespace My::RenderGraph {
    namespace DepthWriteMask {
        static inline int Count = 2;
        enum Enum { 
            Zero = 0,
            All = 1
        };

        static const char* s_value_names[] = {
            "Zero",
            "All"
        };

        static const char* ToString( Enum e ) {
            return s_value_names[(int)e];
        }
    } // namespace DepthWriteMask
} // namespace My::RenderGraph
