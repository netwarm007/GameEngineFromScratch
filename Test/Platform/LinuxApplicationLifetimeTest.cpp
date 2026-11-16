#include <iostream>

#include "OpenGLApplication.hpp"

using namespace My;

namespace {
class ContextCheckingModule : public IMemoryManager {
   public:
    explicit ContextCheckingModule(OpenGLApplication& app) : m_App(app) {}

    int Initialize() override {
        ++initializeCount;
        initializedWithContext = glXGetCurrentContext() != nullptr;
        return initializedWithContext ? 0 : 17;
    }
    void Finalize() override {
        valid = m_App.GetMainWindowHandler() != nullptr &&
                glXGetCurrentContext() != nullptr;
        finalized = true;
    }
    void Tick() override { ++tickCount; }
    void* AllocatePage(size_t) override { return nullptr; }
    void FreePage(void*) override {}

    bool finalized = false;
    bool initializedWithContext = false;
    bool valid = false;
    int initializeCount = 0;
    int tickCount = 0;

   private:
    OpenGLApplication& m_App;
};

enum class FailurePoint { Drawable, DrawableError, Context, Bind, CoreContext };

class GlxFailureScope {
   public:
    explicit GlxFailureScope(FailurePoint point) : point(point) {
        current = this;
        glXCreateWindow = CreateWindow;
        glXCreateContextAttribsARB = CreateCoreContext;
        glXCreateNewContext = CreateLegacyContext;
        glXMakeContextCurrent = MakeCurrent;
        glXDestroyContext = DestroyContext;
        glXDestroyWindow = DestroyWindow;
        previousHandler = XSetErrorHandler(ErrorHandler);
    }

    ~GlxFailureScope() {
        XSetErrorHandler(previousHandler);
        glXCreateWindow = createWindow;
        glXCreateContextAttribsARB = createCoreContext;
        glXCreateNewContext = createLegacyContext;
        glXMakeContextCurrent = makeCurrent;
        glXDestroyContext = destroyContext;
        glXDestroyWindow = destroyWindow;
        current = nullptr;
    }

    bool HandlerRestored() {
        return XSetErrorHandler(ErrorHandler) == ErrorHandler;
    }

    FailurePoint point;
    int windowsCreated = 0, windowsDestroyed = 0;
    int contextsCreated = 0, contextsDestroyed = 0;
    int coreAttempts = 0, legacyAttempts = 0, unexpectedErrors = 0;

   private:
    static int ErrorHandler(Display*, XErrorEvent*) {
        ++current->unexpectedErrors;
        return 0;
    }

    static GLXWindow CreateWindow(Display* dpy, GLXFBConfig config,
                                  Window win, const int* attribs) {
        if (current->point == FailurePoint::Drawable) return 0;
        const auto result = current->createWindow(dpy, config, win, attribs);
        if (result) ++current->windowsCreated;
        if (current->point == FailurePoint::DrawableError) {
            // Queue a real asynchronous X error after allocating the GLX window.
            XDestroyWindow(dpy, None);
        }
        return result;
    }

    static GLXContext CreateCoreContext(Display* dpy, GLXFBConfig config,
                                        GLXContext share, Bool direct,
                                        const int* attribs) {
        ++current->coreAttempts;
        if (current->point == FailurePoint::Context ||
            current->point == FailurePoint::CoreContext)
            return nullptr;
        const auto result =
            current->createCoreContext(dpy, config, share, direct, attribs);
        if (result) ++current->contextsCreated;
        return result;
    }

    static GLXContext CreateLegacyContext(Display* dpy, GLXFBConfig config,
                                          int type, GLXContext share, Bool direct) {
        ++current->legacyAttempts;
        if (current->point == FailurePoint::Context) return nullptr;
        const auto result =
            current->createLegacyContext(dpy, config, type, share, direct);
        if (result) ++current->contextsCreated;
        return result;
    }

    static Bool MakeCurrent(Display* dpy, GLXDrawable draw, GLXDrawable read,
                             GLXContext context) {
        if (current->point == FailurePoint::Bind && context) return False;
        return current->makeCurrent(dpy, draw, read, context);
    }

    static void DestroyContext(Display* dpy, GLXContext context) {
        ++current->contextsDestroyed;
        current->destroyContext(dpy, context);
    }

    static void DestroyWindow(Display* dpy, GLXWindow window) {
        ++current->windowsDestroyed;
        current->destroyWindow(dpy, window);
    }

    inline static GlxFailureScope* current = nullptr;
    PFNGLXCREATEWINDOWPROC createWindow = glXCreateWindow;
    PFNGLXCREATECONTEXTATTRIBSARBPROC createCoreContext = glXCreateContextAttribsARB;
    PFNGLXCREATENEWCONTEXTPROC createLegacyContext = glXCreateNewContext;
    PFNGLXMAKECONTEXTCURRENTPROC makeCurrent = glXMakeContextCurrent;
    PFNGLXDESTROYCONTEXTPROC destroyContext = glXDestroyContext;
    PFNGLXDESTROYWINDOWPROC destroyWindow = glXDestroyWindow;
    XErrorHandler previousHandler = nullptr;
};

bool TestInitializationFailure(FailurePoint point) {
    GfxConfiguration config;
    config.screenWidth = config.screenHeight = 64;
    OpenGLApplication app(config);
    ContextCheckingModule module(app);
    app.CreateMainWindow();
    app.RegisterManagerModule(&module);
    GlxFailureScope injection(point);
    const int result = app.Initialize();
    app.Tick();
    const bool success = point == FailurePoint::CoreContext;
    const bool initializationValid = success
        ? result == 0 && module.initializedWithContext &&
              module.initializeCount == 1 && module.tickCount == 1 &&
              injection.coreAttempts == 1 && injection.legacyAttempts == 1
        : result != 0 && module.initializeCount == 0 && module.tickCount == 0 &&
              !glXGetCurrentContext() &&
              injection.windowsCreated == injection.windowsDestroyed &&
              injection.contextsCreated == injection.contextsDestroyed;
    const bool nativeWindowAlive = app.GetMainWindowHandler() != nullptr;
    const bool handlerRestored = injection.HandlerRestored();
    app.Finalize();
    app.Finalize();
    if (!initializationValid || !nativeWindowAlive || !handlerRestored ||
        injection.unexpectedErrors != 0 || app.GetMainWindowHandler() ||
        glXGetCurrentContext() ||
        injection.windowsCreated != injection.windowsDestroyed ||
        injection.contextsCreated != injection.contextsDestroyed ||
        module.finalized != success) {
        std::cerr << "GLX failure boundary " << static_cast<int>(point)
                  << " did not preserve initialization/cleanup invariants\n";
        return false;
    }
    return true;
}
}  // namespace

int main() {
    for (auto point : {FailurePoint::Drawable, FailurePoint::DrawableError,
                       FailurePoint::Context, FailurePoint::Bind,
                       FailurePoint::CoreContext}) {
        if (!TestInitializationFailure(point)) return 1;
    }
    GfxConfiguration config;
    config.screenWidth = 64;
    config.screenHeight = 64;
    OpenGLApplication app(config);
    ContextCheckingModule module(app);
    app.CreateMainWindow();
    if (!app.GetMainWindowHandler() || glXGetCurrentContext()) {
        std::cerr << "Window creation must not initialize the GLX context\n";
        app.Finalize();
        return 1;
    }
    app.RegisterManagerModule(&module);
    const int result = app.Initialize();
    app.Finalize();
    if (result != 0 || !module.initializedWithContext ||
        !module.finalized || !module.valid ||
        app.GetMainWindowHandler() != nullptr ||
        glXGetCurrentContext() != nullptr) {
        std::cerr << "Linux application did not finalize modules with a live "
                     "context before releasing the backend and window\n";
        return 1;
    }
    app.Finalize();
    return 0;
}
