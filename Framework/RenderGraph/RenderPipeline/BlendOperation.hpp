#pragma once
namespace My::RenderGraph {
    namespace BlendOperation {
        static inline int Count = 5;
        enum Enum { 
            Add = 0,
            Subtract = 1,
            RevSubtract = 2,
            Min = 3,
            Max = 4
        };

        static const char* s_value_names[] = {
            "Add",
            "Subtract",
            "RevSubtract",
            "Min",
            "Max"
        };

        static const char* ToString( Enum e ) {
            return s_value_names[(int)e];
        }
    } // namespace BlendOperation
} // namespace My::RenderGraph
