#include <wrl/client.h>
#include <iostream>
#include <thread>
#include <vector>

#include "D3d12Application.hpp"

using namespace My;

namespace {
class FailingApplication : public D3d12Application {
   public:
    using D3d12Application::D3d12Application;
    int failAt = 0;
    int attempts = 0;
    std::vector<HANDLE> events;

   protected:
    HANDLE CreateFenceEvent() override {
        if (attempts++ == failAt) {
            SetLastError(ERROR_NOT_ENOUGH_MEMORY);
            return nullptr;
        }
        HANDLE event = D3d12Application::CreateFenceEvent();
        if (event) events.push_back(event);
        return event;
    }
};

class RecordingModule : public IMemoryManager {
   public:
    int Initialize() override {
        ++initialized;
        return 0;
    }
    void Finalize() override { ++finalized; }
    void Tick() override { ++ticks; }
    void* AllocatePage(size_t) override { return nullptr; }
    void FreePage(void*) override {}
    int initialized = 0, finalized = 0, ticks = 0;
};

class SubmittingModule : public RecordingModule {
   public:
    explicit SubmittingModule(D3d12Application& app) : app(app) {}
    ~SubmittingModule() {
        if (unblock.joinable()) unblock.join();
    }
    int Initialize() override {
        auto* device = app.GetRHI().GetDevice();
        auto* queue = app.GetRHI().GetGraphicsCommandQueue();
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                       IID_PPV_ARGS(&gate))) ||
            FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                       IID_PPV_ARGS(&completion))) ||
            FAILED(device->CreateCommandAllocator(
                D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) ||
            FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                             allocator.Get(), nullptr,
                                             IID_PPV_ARGS(&list))) ||
            FAILED(list->Close()) || FAILED(queue->Wait(gate.Get(), 1))) {
            std::cerr << "Cannot prepare native GPU rollback test\n";
            return -1;
        }
        ID3D12CommandList* lists[] = {list.Get()};
        queue->ExecuteCommandLists(1, lists);
        const HRESULT result = queue->Signal(completion.Get(), 1);
        unblock = std::thread([this] {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (FAILED(gate->Signal(1))) {
                std::cerr << "Cannot unblock native GPU rollback test\n";
            }
        });
        return SUCCEEDED(result) ? 0 : -1;
    }
    void Finalize() override {
        completedBeforeRollback =
            completion->GetCompletedValue() >= 1 && app.GetRHI().GetDevice();
        unblock.join();
        list.Reset();
        allocator.Reset();
        completion.Reset();
        gate.Reset();
        RecordingModule::Finalize();
    }
    bool completedBeforeRollback = false;

   private:
    D3d12Application& app;
    Microsoft::WRL::ComPtr<ID3D12Fence> gate, completion;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> list;
    std::thread unblock;
};

class FailingModule : public RecordingModule {
   public:
    int Initialize() override { return -23; }
};
}  // namespace

int main() {
    for (int failAt = 0; failAt < 3; ++failAt) {
        GfxConfiguration config(8, 8, 8, 8, 24, 8, 1, 64, 64,
                                "D3D12 initialization failure test");
        FailingApplication app(config);
        RecordingModule module;
        app.failAt = failAt;
        app.CreateMainWindow();
        app.RegisterManagerModule(&module);
        const int result = app.Initialize();
        if (result != ERROR_NOT_ENOUGH_MEMORY || module.initialized ||
            app.GetRHI().GetDevice() || !app.GetMainWindowHandler() ||
            app.events.size() != static_cast<size_t>(failAt)) {
            std::cerr << "D3D12 event failure did not unwind native setup\n";
            return 1;
        }
        for (HANDLE event : app.events) {
            DWORD flags;
            if (GetHandleInformation(event, &flags)) {
                std::cerr << "Partial initialization leaked a fence event\n";
                return 1;
            }
        }
        app.Tick();
        app.Finalize();
        // A newly allocated event may reuse a released fence handle. Repeated
        // cleanup must not close that unrelated handle.
        HANDLE sentinel = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (!sentinel) {
            std::cerr << "Cannot allocate sentinel event\n";
            return 1;
        }
        app.Finalize();
        const bool valid = SetEvent(sentinel) &&
                           WaitForSingleObject(sentinel, 0) == WAIT_OBJECT_0;
        CloseHandle(sentinel);
        if (!valid || app.GetMainWindowHandler() || module.finalized ||
            module.ticks) {
            std::cerr << "Repeated failure cleanup changed live state\n";
            return 1;
        }
        // Recreate the window after explicit cleanup, then use normal events.
        app.failAt = -1;
        app.CreateMainWindow();
        app.RegisterManagerModule(&module);
        if (app.Initialize() != 0 || !app.GetRHI().GetDevice() ||
            module.initialized != 1) {
            std::cerr << "D3D12 initialization after cleanup failed\n";
            return 1;
        }
        int callbacks = 0;
        app.GetRHI().DestroyResourceCB([&] { ++callbacks; });
        app.Finalize();
        app.Finalize();
        if (callbacks != 1 || module.finalized != 1) {
            std::cerr << "D3D12 resource cleanup repeated\n";
            return 1;
        }
    }
    D3d12Application app;
    SubmittingModule submitting(app);
    FailingModule failing;
    app.CreateMainWindow();
    app.RegisterManagerModule(&submitting);
    app.RegisterManagerModule(&failing);
    if (app.Initialize() != -23 || !submitting.completedBeforeRollback ||
        submitting.finalized != 1 || failing.finalized) {
        std::cerr << "Native GPU work was not drained before module rollback\n";
        return 1;
    }
    app.Finalize();
    return 0;
}
