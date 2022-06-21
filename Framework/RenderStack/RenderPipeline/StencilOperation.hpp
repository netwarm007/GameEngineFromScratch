namespace My {
    namespace StencilOperation {
        int Count = 8;
        enum Enum { 
            Keep = 0,
            Zero = 1,
            Replace = 2,
            IncrSat = 3,
            DecrSat = 4,
            Invert = 5,
            Incr = 6,
            Decr = 7
        };

        static const char* s_value_names[] = {
            "Keep",
            "Zero",
            "Replace",
            "IncrSat",
            "DecrSat",
            "Invert",
            "Incr",
            "Decr"
        };

        static const char* ToString( Enum e ) {
            return s_value_names[(int)e];
        }
    } // namespace StencilOperation
} // namespace My
