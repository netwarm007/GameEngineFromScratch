#pragma once
#include "IRuntimeModule.hpp"

namespace My {
    class InputManager : implements IRuntimeModule
    {
        public:
            virtual int Initialize();
            virtual void Finalize();
            virtual void Tick();

            // keyboard handling
            void UpArrowKeyDown();
            void UpArrowKeyUp();
            void DownArrowKeyDown();
            void DownArrowKeyUp();
            void LeftArrowKeyDown();
            void LeftArrowKeyUp();
            void RightArrowKeyDown();
            void RightArrowKeyUp();

            void ResetKeyDown();
            void ResetKeyUp();

#ifdef DEBUG
            void DebugKeyDown();
            void DebugKeyUp();
#endif
            // mouse handling
            void LeftMouseButtonDown();
            void LeftMouseButtonUp();
            void LeftMouseDrag(float deltaX, float deltaY);

        protected:
            bool m_bUpKeyPressed    = false;
            bool m_bDownKeyPressed  = false;
            bool m_bLeftKeyPressed  = false;
            bool m_bRightKeyPressed = false;
    };

    extern InputManager* g_pInputManager;
}

