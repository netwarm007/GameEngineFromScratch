namespace My::RenderGraph {
    namespace RenderTargetLoadStoreAction {
        int Count = 3;
        enum Enum { 
            DontCare = 0,
            Clear = 1,
            Keep = 2
        };

        static const char* s_value_names[] = {
            "DontCare",
            "Clear",
            "Keep"
        };

        static const char* ToString( Enum e ) {
            return s_value_names[(int)e];
        }
    } // namespace RenderTargetLoadStoreAction
} // namespace My::RenderGraph
