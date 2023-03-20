#pragma once
namespace My::RenderGraph {
    namespace ComparisonFunction {
        static inline int Count = 8;
        enum Enum { 
            Never = 0,
            Less = 1,
            Equal = 2,
            LessEqual = 3,
            Greater = 4,
            NotEqual = 5,
            GreaterEqual = 6,
            Always = 7
        };

        static const char* s_value_names[] = {
            "Never",
            "Less",
            "Equal",
            "LessEqual",
            "Greater",
            "NotEqual",
            "GreaterEqual",
            "Always"
        };

        static const char* ToString( Enum e ) {
            return s_value_names[(int)e];
        }
    } // namespace ComparisonFunction
} // namespace My::RenderGraph
